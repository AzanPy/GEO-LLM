# retriever.py
from __future__ import annotations

import os
import re
import math
import time
import pickle
from pathlib import Path
from threading import RLock
from typing import List, Dict, Optional, Tuple, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from rank_bm25 import BM25Okapi

from .config import settings


# Biomedical-friendly tokenization:
# keeps tokens like: "IL-6", "CD4+", "scRNA-seq", "T-cell"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-]+")


class BM25Index:
    """
    In-memory BM25 index built from Qdrant payloads (text field).
    Supports optional disk cache (pickle).

    Stored:
      - doc_ids: List[str]
      - doc_meta: List[dict]  (optional, useful for post-filtering)
      - corpus_tokens: List[List[str]]
      - bm25: BM25Okapi
    """

    def __init__(self, name: str, cache_dir: Optional[str] = None):
        self.name = name
        self.cache_dir = cache_dir or os.getenv("BM25_CACHE_DIR", "")
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self._built_at: Optional[float] = None

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    def _cache_path(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        d = Path(self.cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"bm25_{self.name}.pkl"

    def is_ready(self) -> bool:
        return self.bm25 is not None and len(self.doc_ids) > 0

    def load_cache(self) -> bool:
        p = self._cache_path()
        if not p or not p.exists():
            return False

        try:
            with p.open("rb") as f:
                obj = pickle.load(f)

            self.doc_ids = obj["doc_ids"]
            self.doc_meta = obj.get("doc_meta", [])
            self.corpus_tokens = obj["corpus_tokens"]
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self._built_at = obj.get("built_at")

            print(f"[BM25] Loaded cache '{self.name}' ({len(self.doc_ids)} docs) from {p}")
            return True
        except Exception as e:
            print(f"[BM25] Cache load failed '{self.name}': {e}")
            return False

    def save_cache(self) -> None:
        p = self._cache_path()
        if not p or not self.corpus_tokens or not self.doc_ids:
            return

        try:
            payload = {
                "doc_ids": self.doc_ids,
                "doc_meta": self.doc_meta,
                "corpus_tokens": self.corpus_tokens,
                "built_at": self._built_at or time.time(),
            }
            with p.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[BM25] Saved cache '{self.name}' ({len(self.doc_ids)} docs) to {p}")
        except Exception as e:
            print(f"[BM25] Cache save failed '{self.name}': {e}")

    def build_from_qdrant(
        self,
        client: QdrantClient,
        collection_name: str,
        text_key: str = "text",
        limit: int = 1000,
    ) -> None:
        """
        Build BM25 by scrolling all points in a Qdrant collection (Cloud-friendly).
        NOTE: This can be heavy for large collections.
        """
        self.doc_ids = []
        self.doc_meta = []
        self.corpus_tokens = []
        self.bm25 = None

        offset = None

        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for p in points:
                payload = p.payload or {}
                text = payload.get(text_key, "") or ""
                toks = self._tokenize(text)
                if not toks:
                    continue

                self.doc_ids.append(str(p.id))
                self.doc_meta.append(
                    {
                        "gse_id": payload.get("gse_id"),
                        "gsm_id": payload.get("gsm_id"),
                        "field": payload.get("field"),
                        "level": payload.get("level"),
                    }
                )
                self.corpus_tokens.append(toks)

            if offset is None:
                break

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self._built_at = time.time()
            print(f"[BM25] Built '{self.name}' from '{collection_name}': {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.is_ready():
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked]


class HybridRetriever:
    """
    Hybrid retrieval: Qdrant vector search + BM25 keyword search.

    - Lazy BM25 build (no heavy work in __init__)
    - Optional disk cache for BM25
    - Post-filter BM25-only hits (best-effort) when qdrant_filter is provided
    - Qdrant Cloud friendly (query_points + retrieve) with fallback to search()
    """

    def __init__(self, qdrant_client: QdrantClient):
        self.client = qdrant_client

        self.bm25_study = BM25Index(name="study")
        self.bm25_sample = BM25Index(name="sample")

        self._lock = RLock()
        self._study_ready = False
        self._sample_ready = False

    def _ensure_bm25(self, which: str) -> None:
        with self._lock:
            if which == "study" and self._study_ready:
                return
            if which == "sample" and self._sample_ready:
                return

            idx = self.bm25_study if which == "study" else self.bm25_sample
            col = settings.STUDY_COLLECTION if which == "study" else settings.SAMPLE_COLLECTION

            if idx.load_cache():
                if which == "study":
                    self._study_ready = True
                else:
                    self._sample_ready = True
                return

            print(f"[HybridRetriever] Building BM25 '{which}' from Qdrant collection '{col}' ...")
            idx.build_from_qdrant(self.client, col)
            idx.save_cache()

            if which == "study":
                self._study_ready = True
            else:
                self._sample_ready = True

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        vmin, vmax = min(scores), max(scores)
        if math.isclose(vmin, vmax):
            return [0.5] * len(scores)
        return [(v - vmin) / (vmax - vmin) for v in scores]

    def _get_payload_value(self, payload: Dict[str, Any], key: str):
        """PATCH #2: support nested payload['metadata'] values too."""
        if not payload:
            return None
        if key in payload:
            return payload.get(key)
        meta = payload.get("metadata")
        if isinstance(meta, dict):
            return meta.get(key)
        return None

    def _payload_satisfies_filter(self, payload: Dict[str, Any], flt: Filter) -> bool:
        if payload is None or flt is None:
            return True

        must = getattr(flt, "must", None)
        if not must:
            return True

        for cond in must:
            if not isinstance(cond, FieldCondition):
                continue

            key = cond.key
            match = cond.match
            val = self._get_payload_value(payload, key)

            if isinstance(match, MatchValue):
                if val != match.value:
                    return False
            elif isinstance(match, MatchAny):
                allowed = getattr(match, "any", None)
                if allowed is None:
                    continue
                if val not in allowed:
                    return False

        return True

    def hybrid_search(
        self,
        collection: str,
        query_text: str,
        query_vector: List[float],
        bm25_index: BM25Index,
        top_k: int = 10,
        qdrant_filter: Optional[Filter] = None,
        bm25_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        bm25_k = bm25_k or (top_k * 3)

        # 1) Vector search (PATCH #3: fallback for client-version compatibility)
        try:
            vec_points = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            ).points
        except Exception:
            vec_points = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )

        vec_map: Dict[str, float] = {str(p.id): float(p.score) for p in (vec_points or [])}

        # 2) BM25 search
        bm25_hits = bm25_index.search(query_text, top_k=bm25_k)
        bm25_map: Dict[str, float] = {doc_id: float(score) for doc_id, score in bm25_hits}

        # 3) Union IDs (PATCH #1: deterministic ordering)
        union_ids = sorted(set(vec_map.keys()) | set(bm25_map.keys()))
        if not union_ids:
            return []

        # 4) Fetch payloads for union candidates
        fetched = self.client.retrieve(
            collection_name=collection,
            ids=union_ids,
            with_payload=True,
            with_vectors=False,
        )
        id_to_payload: Dict[str, Dict[str, Any]] = {str(p.id): (p.payload or {}) for p in (fetched or [])}

        # 5) Optional post-filter for BM25-only / union (best-effort)
        if qdrant_filter is not None:
            filtered_ids = []
            for pid in union_ids:
                payload = id_to_payload.get(pid)
                if payload is None:
                    continue
                if self._payload_satisfies_filter(payload, qdrant_filter):
                    filtered_ids.append(pid)
            union_ids = filtered_ids
            if not union_ids:
                return []

        # 6) Normalize scores within candidate set
        vec_scores = [vec_map.get(i, 0.0) for i in union_ids]
        bm_scores = [bm25_map.get(i, 0.0) for i in union_ids]
        vec_norm = self._normalize_scores(vec_scores)
        bm_norm = self._normalize_scores(bm_scores)

        alpha = float(getattr(settings, "ALPHA_VECTOR", 0.65))
        beta = float(getattr(settings, "BETA_BM25", 0.35))

        # 7) Merge + rank
        out: List[Dict[str, Any]] = []
        for idx, pid in enumerate(union_ids):
            payload = id_to_payload.get(pid)
            if payload is None:
                continue
            out.append(
                {
                    "id": pid,
                    "final_score": alpha * vec_norm[idx] + beta * bm_norm[idx],
                    "vector_score": float(vec_map.get(pid, 0.0)),
                    "bm25_score": float(bm25_map.get(pid, 0.0)),
                    "payload": payload,
                }
            )

        out.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
        return out[:top_k]

    def search_study(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
        qdrant_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_bm25("study")
        top_k = top_k or int(getattr(settings, "TOP_K_STUDY", 10))
        return self.hybrid_search(
            collection=settings.STUDY_COLLECTION,
            query_text=query_text,
            query_vector=query_vector,
            bm25_index=self.bm25_study,
            top_k=top_k,
            qdrant_filter=qdrant_filter,
        )

    def search_sample(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
        qdrant_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_bm25("sample")
        top_k = top_k or int(getattr(settings, "TOP_K_SAMPLE", 10))
        return self.hybrid_search(
            collection=settings.SAMPLE_COLLECTION,
            query_text=query_text,
            query_vector=query_vector,
            bm25_index=self.bm25_sample,
            top_k=top_k,
            qdrant_filter=qdrant_filter,
        )