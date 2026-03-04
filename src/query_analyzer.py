import json
import os
import re
import pickle
from difflib import SequenceMatcher
from threading import RLock
from typing import Dict, Any, List, Optional, Tuple, Set

from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from .config import settings


# Default schema fields for GEO data
DEFAULT_SCHEMA_FIELDS = [
    "organism", "tissue", "disease", "assay_type", "library_strategy",
    "library_source", "library_selection", "molecule", "platform_id",
    "instrument_model", "status", "genotype", "cell_line", "cell_type"
]

VALID_QUERY_TYPES = {
    "SEMANTIC", "MODALITY", "METADATA", "EXPERIMENTAL",
    "STUDY_DESIGN", "MULTI_CONSTRAINT", "PLATFORM"
}

_TOKEN_CLEAN_RE = re.compile(r"\s+")
_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


class QueryAnalyzer:
    """
    Query Analyzer:
      1) Builds a metadata vocabulary index from Qdrant payloads
      2) Extracts candidates from query (substring)
      3) Uses Groq LLM to parse into structured filters (STRICT JSON)
      4) Grounds/validates filters against vocabulary (exact/fuzzy)
      5) Computes confidence + strategy
    """

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Please set GROQ_API_KEY in environment.")

        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.metadata_index: Dict[str, Set[str]] = {f: set() for f in DEFAULT_SCHEMA_FIELDS}
        self._lock = RLock()

    # ---------------- normalization & parsing ----------------
    def _normalize(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = s.replace("_", " ").replace("-", " ").replace("/", " ")
        s = _TOKEN_CLEAN_RE.sub(" ", s)
        # Keep + (CD4+), remove other punctuation
        s = re.sub(r"[^\w\s\+]", "", s)
        return s.strip()

    def _safe_json_load(self, text: str) -> Optional[dict]:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            m = _JSON_OBJ_RE.search(text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

    def _fuzzy_match(self, value: str, candidates: Set[str], cutoff: float = 0.85) -> Optional[Tuple[str, float]]:
        v = self._normalize(value)
        best_val: Optional[str] = None
        best_score = 0.0
        for c in candidates:
            score = SequenceMatcher(None, v, c).ratio()
            if score > best_score:
                best_val, best_score = c, score
        if best_val and best_score >= cutoff:
            return best_val, best_score
        return None

    # ---------------- cache helpers ----------------
    def _cache_path(self) -> Optional[str]:
        cache_dir = os.getenv("METADATA_CACHE_DIR", "")
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "metadata_index.pkl")

    def load_metadata_cache(self) -> bool:
        p = self._cache_path()
        if not p or not os.path.exists(p):
            return False
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            idx = obj.get("metadata_index", {})
            # ensure schema keys exist
            self.metadata_index = {f: set(idx.get(f, set())) for f in DEFAULT_SCHEMA_FIELDS}
            print(f"[QueryAnalyzer] Loaded metadata index cache from {p}")
            return True
        except Exception as e:
            print(f"[QueryAnalyzer] Failed to load metadata cache: {e}")
            return False

    def save_metadata_cache(self) -> None:
        p = self._cache_path()
        if not p:
            return
        try:
            with open(p, "wb") as f:
                pickle.dump({"metadata_index": self.metadata_index}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[QueryAnalyzer] Saved metadata index cache to {p}")
        except Exception as e:
            print(f"[QueryAnalyzer] Failed to save metadata cache: {e}")

    # ---------------- metadata index building ----------------
    def build_metadata_index(
        self,
        qdrant_client: QdrantClient,
        collections: List[str],
        force_rebuild: bool = False
    ) -> None:
        """
        Build vocabulary index from Qdrant Cloud data.
        - Tries to load cache (unless force_rebuild=True)
        - Scrolls collections and collects values for DEFAULT_SCHEMA_FIELDS
        """
        with self._lock:
            if not force_rebuild and self.load_metadata_cache():
                return

            print("[QueryAnalyzer] Building metadata value index from Qdrant Cloud...")
            self.metadata_index = {f: set() for f in DEFAULT_SCHEMA_FIELDS}

            for col in collections:
                offset = None
                while True:
                    points, offset = qdrant_client.scroll(
                        collection_name=col,
                        limit=1000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )

                    for pt in points:
                        payload = pt.payload or {}
                        meta = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}

                        for field in DEFAULT_SCHEMA_FIELDS:
                            v = payload.get(field) or meta.get(field)
                            if not v:
                                continue

                            if isinstance(v, list):
                                for item in v:
                                    nv = self._normalize(str(item))
                                    if nv:
                                        self.metadata_index[field].add(nv)
                            else:
                                nv = self._normalize(str(v))
                                if nv:
                                    self.metadata_index[field].add(nv)

                    if offset is None:
                        break

            for field, vals in self.metadata_index.items():
                print(f"  - {field}: {len(vals)} unique values")

            self.save_metadata_cache()

    # ---------------- candidate extraction ----------------
    def extract_candidates(self, query: str) -> Dict[str, List[str]]:
        """
        Find candidate values from query using substring matching
        (fast heuristic using built vocab).
        """
        qn = self._normalize(query)
        candidates: Dict[str, List[str]] = {}

        for field, vocab in self.metadata_index.items():
            if not vocab:
                continue
            hits: List[str] = []
            for val in vocab:
                if val and val in qn:
                    hits.append(val)
                    if len(hits) >= 5:
                        break
            if hits:
                candidates[field] = hits

        return candidates

    # ---------------- LLM parse ----------------
    def llm_parse(self, query: str, candidates: Dict[str, List[str]]) -> dict:
        schema_text = ",".join(sorted(DEFAULT_SCHEMA_FIELDS))

        cand_lines = [f"- {f}: {', '.join(vals[:3])}" for f, vals in candidates.items()]
        cand_text = "\n".join(cand_lines) if cand_lines else "(none)"

        prompt = f"""You are a biomedical dataset query parser for GEO-like metadata.

Allowed metadata fields: {schema_text}

Dataset-grounded candidate values found in this query:
{cand_text}

Return STRICT JSON only with this schema:
{{
  "query_type": "SEMANTIC|MODALITY|METADATA|EXPERIMENTAL|STUDY_DESIGN|MULTI_CONSTRAINT|PLATFORM",
  "filters": {{"<field>": "<value>", "...": "..."}},
  "requires_replicates": true/false,
  "logical_operator": "AND|OR|NOT|null"
}}

Rules:
- Only use metadata fields when query explicitly refers to dataset properties
- Only use filter values from the candidate list provided (do not invent)
- If unsure about a filter value, omit it
- For multiple constraints, use MULTI_CONSTRAINT
- logical_operator reflects explicit boolean words only; otherwise null

Query: \"\"\"{query}\"\"\""""

        try:
            resp = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
            text = resp.choices[0].message.content.strip()
            parsed = self._safe_json_load(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            print(f"[QueryAnalyzer] LLM parse error: {e}")

        return {
            "query_type": "SEMANTIC",
            "filters": {},
            "requires_replicates": False,
            "logical_operator": None,
        }

    # ---------------- grounding ----------------
    def ground_filters(self, filters: Dict[str, str]) -> Tuple[Dict[str, str], List[dict], float]:
        grounded: Dict[str, str] = {}
        dropped: List[dict] = []

        for field, raw_value in (filters or {}).items():
            if field not in DEFAULT_SCHEMA_FIELDS:
                dropped.append({"field": field, "value": raw_value, "reason": "field_not_allowed"})
                continue

            v = self._normalize(raw_value)
            vocab = self.metadata_index.get(field, set())

            if v in vocab:
                grounded[field] = v
                continue

            fm = self._fuzzy_match(v, vocab)
            if fm:
                grounded[field] = fm[0]
                continue

            dropped.append({"field": field, "value": raw_value, "reason": "value_not_grounded"})

        confidence = len(grounded) / len(filters) if filters else 1.0
        return grounded, dropped, confidence

    # ---------------- analysis pipeline ----------------
    def analyze(self, query: str) -> dict:
        # Step 1: Extract candidates
        candidates = self.extract_candidates(query)

        # Step 2: LLM parse
        parsed = self.llm_parse(query, candidates)

        # Step 2.5: Validate query_type
        qt = (parsed.get("query_type") or "SEMANTIC").upper()
        if qt not in VALID_QUERY_TYPES:
            qt = "SEMANTIC"

        # Step 3: Ground filters
        grounded, dropped, ground_conf = self.ground_filters(parsed.get("filters", {}) or {})

        # Step 4: Confidence
        parse_conf = 0.95 if qt else 0.3
        schema_conf = len(grounded) / max(len(parsed.get("filters", {})) or 0, 1)

        qn = self._normalize(query)
        rep_mentioned = any(r in qn for r in ["replicate", "replicates", "n="])
        consistency = 0.75 if (rep_mentioned and not parsed.get("requires_replicates")) else 0.9

        overall = 0.35 * parse_conf + 0.25 * schema_conf + 0.30 * ground_conf + 0.10 * consistency

        if overall >= 0.85:
            strategy = "HARD_FILTER"
        elif overall >= 0.60:
            strategy = "SOFT_BOOST"
        else:
            strategy = "SEMANTIC_ONLY"

        return {
            "query_type": qt,
            "filters": grounded,
            "requires_replicates": bool(parsed.get("requires_replicates", False)),
            "logical_operator": parsed.get("logical_operator"),
            "confidence": {
                "parse_confidence": round(parse_conf, 2),
                "schema_match_confidence": round(schema_conf, 2),
                "value_grounding_confidence": round(ground_conf, 2),
                "consistency_confidence": round(consistency, 2),
                "overall_confidence": round(overall, 2),
            },
            "strategy": strategy,
            "candidates": candidates,
            "dropped_filters": dropped,
        }


def get_analyzer() -> QueryAnalyzer:
    """
    Factory instead of global singleton.
    Importing this module will not create Groq clients automatically.
    """
    return QueryAnalyzer()