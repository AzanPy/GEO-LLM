from typing import List, Dict, Any, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny  # ✅ FIX: MatchAny added

from groq import Groq

from .config import settings
from .retriever import HybridRetriever
from .query_analyzer import get_analyzer
from .embedder import embedder


# -----------------------------
# State type for LangGraph
# -----------------------------
class GeoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: List[Dict[str, Any]]
    top_gse_ids: List[str]


class RAGChain:
    def __init__(self, qdrant_client: QdrantClient):
        self.client = qdrant_client

        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Please set GROQ_API_KEY in environment.")
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

        # Hybrid retriever (lazy BM25 inside)
        self.retriever = HybridRetriever(qdrant_client)

        # Query analyzer (factory, no singleton)
        self.analyzer = get_analyzer()

        # Build metadata index once (can be cached via METADATA_CACHE_DIR)
        self.analyzer.build_metadata_index(
            qdrant_client,
            [settings.STUDY_COLLECTION, settings.SAMPLE_COLLECTION],
            force_rebuild=False,
        )

    # -----------------------------
    # helpers
    # -----------------------------
    def _get_last_user_text(self, messages: List[BaseMessage]) -> str:
        for msg in reversed(messages or []):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
                return (msg.content or "").strip()
        return ""

    def _get_payload_value(self, payload: dict, field: str):
        """Safely get value from payload or nested metadata."""
        if not payload:
            return None
        if field in payload and payload[field] not in (None, "", []):
            return payload[field]
        meta = payload.get("metadata")
        if isinstance(meta, dict):
            v = meta.get(field)
            return v if v not in (None, "", []) else None
        return None

    def _apply_soft_boost(self, hits: List[Dict[str, Any]], filters: Dict[str, str], bonus: float = 0.03) -> List[Dict[str, Any]]:
        """Boost final_score based on metadata matches."""
        if not hits or not filters:
            return hits

        for h in hits:
            p = h.get("payload") or {}
            match_bonus = 0.0
            for f, v in filters.items():
                pv = self._get_payload_value(p, f)
                if pv and str(pv).strip().lower() == str(v).strip().lower():
                    match_bonus += bonus
            h["final_score"] = float(h.get("final_score", 0.0)) + match_bonus

        hits.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
        return hits

    def _pack_hit(self, hit: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Format hit for context."""
        p = hit.get("payload") or {}
        return {
            "source": source,
            "final_score": float(hit.get("final_score", 0.0)),
            "vector_score": float(hit.get("vector_score", 0.0)),
            "bm25_score": float(hit.get("bm25_score", 0.0)),
            "gse_id": p.get("gse_id"),
            "gsm_id": p.get("gsm_id"),
            "field": p.get("field"),
            "organism": self._get_payload_value(p, "organism"),
            "assay_type": self._get_payload_value(p, "assay_type"),
            "tissue": self._get_payload_value(p, "tissue"),
            "library_strategy": self._get_payload_value(p, "library_strategy"),
            "library_source": self._get_payload_value(p, "library_source"),
            "library_selection": self._get_payload_value(p, "library_selection"),
            "molecule": self._get_payload_value(p, "molecule"),
            "genotype": self._get_payload_value(p, "genotype"),
            "cell_line": self._get_payload_value(p, "cell_line"),
            "cell_type": self._get_payload_value(p, "cell_type"),
            "platform_id": self._get_payload_value(p, "platform_id"),
            "instrument_model": self._get_payload_value(p, "instrument_model"),
            "bioproject_url": self._get_payload_value(p, "bioproject_url"),
            "biosample_url": self._get_payload_value(p, "biosample_url"),
            "sra_url": self._get_payload_value(p, "sra_url"),
            "text": (p.get("text") or "")[:1200],
        }

    def _unique_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # -----------------------------
    # main run
    # -----------------------------
    def run(self, state: GeoState) -> GeoState:
        messages = state.get("messages", [])
        user_text = self._get_last_user_text(messages)

        if not user_text:
            return {
                "messages": messages,
                "retrieved_context": [],
                "top_gse_ids": [],
            }

        print(f"\n🔍 Query: {user_text[:120]}")

        # 1) Analyze query
        query_info = self.analyzer.analyze(user_text)
        strategy: str = query_info.get("strategy", "SEMANTIC_ONLY")
        filters: Dict[str, str] = query_info.get("filters", {}) or {}
        conf = query_info.get("confidence", {}) or {}

        print(
            f"   Type: {query_info.get('query_type')} | Strategy: {strategy} | "
            f"OverallConf: {conf.get('overall_confidence')}"
        )

        # 2) Embed query
        qvec = embedder.encode_single(user_text, normalize=True)

        # 3) Build Qdrant filter for HARD_FILTER
        study_filter: Optional[Filter] = None
        if strategy == "HARD_FILTER" and filters:
            study_filter = Filter(
                must=[FieldCondition(key=f, match=MatchValue(value=v)) for f, v in filters.items()]
            )

        # 4) Dynamic top_k
        effective_topk = int(getattr(settings, "TOP_K_STUDY", 10))
        overall_conf = float(conf.get("overall_confidence", 0.0) or 0.0)
        if overall_conf < 0.6:
            effective_topk = max(effective_topk, 25)

        # 5) Retrieve studies
        study_hits = self.retriever.search_study(
            query_text=user_text,
            query_vector=qvec,
            top_k=effective_topk,
            qdrant_filter=study_filter,
        )

        if strategy == "SOFT_BOOST" and filters:
            study_hits = self._apply_soft_boost(study_hits, filters, bonus=0.03)

        # Similarity gate
        sim_th = float(getattr(settings, "SIM_THRESHOLD", 0.6))
        gated_studies = [h for h in study_hits if float(h.get("vector_score", 0.0)) >= sim_th]

        # Relaxation: HARD_FILTER -> SOFT_BOOST
        if strategy == "HARD_FILTER" and not gated_studies:
            print("   Relaxing: HARD_FILTER → SOFT_BOOST")
            study_hits = self.retriever.search_study(
                query_text=user_text,
                query_vector=qvec,
                top_k=effective_topk,
                qdrant_filter=None,
            )
            if filters:
                study_hits = self._apply_soft_boost(study_hits, filters, bonus=0.03)
            gated_studies = [h for h in study_hits if float(h.get("vector_score", 0.0)) >= sim_th]

        # Final fallback: semantic only (no gate)
        if not gated_studies:
            print("   Relaxing: → SEMANTIC_ONLY (no similarity gate)")
            gated_studies = self.retriever.search_study(
                query_text=user_text,
                query_vector=qvec,
                top_k=max(effective_topk, 35),
                qdrant_filter=None,
            )

        # 6) Top GSE IDs
        gse_ids = []
        for h in gated_studies:
            gse = (h.get("payload") or {}).get("gse_id")
            if gse:
                gse_ids.append(str(gse))
        top_gse_ids = self._unique_keep_order(gse_ids)[:10]

        # 7) Optional: sample retrieval filtered by top GSE IDs
        sample_context: List[Dict[str, Any]] = []
        if top_gse_ids:
            sample_filter = Filter(
                must=[FieldCondition(key="gse_id", match=MatchAny(any=top_gse_ids))]
            )
            sample_hits = self.retriever.search_sample(
                query_text=user_text,
                query_vector=qvec,
                top_k=int(getattr(settings, "TOP_K_SAMPLE", 10)),
                qdrant_filter=sample_filter,
            )
            sample_context = [self._pack_hit(h, "sample") for h in sample_hits[:15]]

        # 8) Pack final retrieved context
        study_context = [self._pack_hit(h, "study") for h in gated_studies[:25]]
        retrieved_context = study_context + sample_context

        return {
            "messages": messages,
            "retrieved_context": retrieved_context,
            "top_gse_ids": top_gse_ids,
        }