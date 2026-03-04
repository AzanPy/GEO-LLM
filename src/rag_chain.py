from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from groq import Groq

from .config import settings
from .retriever import HybridRetriever
from .query_analyzer import get_analyzer
from .embedder import embedder


# -----------------------------
# State type for LangGraph
# -----------------------------
class GeoState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: List[Dict[str, Any]]
    top_gse_ids: List[str]
    answer: str
    debug: Dict[str, Any]


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

    def _apply_soft_boost(
        self,
        hits: List[Dict[str, Any]],
        filters: Dict[str, str],
        bonus: float = 0.03
    ) -> List[Dict[str, Any]]:
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
        max_chars = int(getattr(settings, "MAX_CONTEXT_CHARS", 1200))

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
            "text": (p.get("text") or "")[:max_chars],
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
    # Evidence + Answer generation
    # -----------------------------
    def _format_evidence_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """
        Turn top-N context into compact evidence blocks.
        Each block begins with citation token like: [GSE293983]
        """
        max_evidence = int(getattr(settings, "MAX_EVIDENCE", 5))
        max_chars = int(getattr(settings, "MAX_CONTEXT_CHARS", 1200))

        blocks: List[str] = []
        for item in (context or [])[:max_evidence]:
            gse = item.get("gse_id") or "UNKNOWN"
            field = item.get("field") or item.get("source") or "study"
            organism = item.get("organism")
            tissue = item.get("tissue")
            assay = item.get("assay_type")

            header_bits = [f"[{gse}]", f"field={field}"]
            if organism:
                header_bits.append(f"organism={organism}")
            if tissue:
                header_bits.append(f"tissue={tissue}")
            if assay:
                header_bits.append(f"assay={assay}")

            txt = (item.get("text") or "")[:max_chars]
            blocks.append(" | ".join(header_bits) + "\n" + txt)

        return "\n\n---\n\n".join(blocks)

    def generate_answer(self, question: str, context: List[Dict[str, Any]], top_gse_ids: List[str]) -> str:
        evidence_text = self._format_evidence_for_llm(context)
        gse_list = ", ".join(top_gse_ids[: int(getattr(settings, "MAX_EVIDENCE", 5))]) if top_gse_ids else "(none)"

        prompt = f"""
You are a biomedical research assistant specialized in GEO datasets.

You MUST answer using ONLY the evidence provided below.
If evidence is insufficient, say what is missing and what to search for next.

Hard rules:
- Keep it concise.
- Provide a short Summary (3-4 lines max).
- Provide Key Findings as bullet points.
- Include citations in square brackets like [GSE293983] ONLY (no other citation formats).
- Do NOT invent dataset IDs.
- Use at most {int(getattr(settings, "MAX_EVIDENCE", 5))} datasets.

Question:
\"\"\"{question}\"\"\"

Top candidate GSE IDs (from retrieval):
{gse_list}

Evidence:
{evidence_text}

Return EXACTLY in this format:

Summary:
...

Key Findings:
- ...
- ...

Relevant Datasets:
- [GSE...] one-line reason
- [GSE...] one-line reason

Next Steps (if needed):
- ...
""".strip()

        try:
            resp = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=650,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Error generating answer: {type(e).__name__}: {e}"

    def _build_debug_payload(
        self,
        query_info: Dict[str, Any],
        strategy: str,
        filters: Dict[str, str],
        effective_topk: int,
        sim_th: float,
        study_hits_count: int,
        gated_studies_count: int,
        sample_hits_count: int,
        top_sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "query_analysis": query_info,
            "strategy": strategy,
            "filters": filters,
            "effective_topk": effective_topk,
            "similarity_threshold": sim_th,
            "counts": {
                "study_hits": study_hits_count,
                "gated_studies": gated_studies_count,
                "sample_hits": sample_hits_count,
            },
            "top_sources_preview": [
                {
                    "gse_id": s.get("gse_id"),
                    "source": s.get("source"),
                    "final_score": s.get("final_score"),
                    "vector_score": s.get("vector_score"),
                    "bm25_score": s.get("bm25_score"),
                    "field": s.get("field"),
                }
                for s in (top_sources or [])[: int(getattr(settings, "MAX_EVIDENCE", 5))]
            ],
        }

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
                "answer": "",
                "debug": {},
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

        # Similarity gate (vector-based)
        sim_th = float(getattr(settings, "SIM_THRESHOLD", 0.6))
        gated_studies = [h for h in (study_hits or []) if float(h.get("vector_score", 0.0)) >= sim_th]

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
            gated_studies = [h for h in (study_hits or []) if float(h.get("vector_score", 0.0)) >= sim_th]

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
        gse_ids: List[str] = []
        for h in (gated_studies or []):
            gse = (h.get("payload") or {}).get("gse_id")
            if gse:
                gse_ids.append(str(gse))
        top_gse_ids = self._unique_keep_order(gse_ids)[:10]

        # 7) Optional: sample retrieval filtered by top GSE IDs
        sample_context: List[Dict[str, Any]] = []
        sample_hits_count = 0
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
            sample_hits_count = len(sample_hits or [])
            sample_context = [self._pack_hit(h, "sample") for h in (sample_hits or [])[:10]]

        # 8) Pack final retrieved context (FIXED: study+sample actually mix)
        max_ev = int(getattr(settings, "MAX_EVIDENCE", 5))
        study_context = [self._pack_hit(h, "study") for h in (gated_studies or [])[:max_ev]]

        # ✅ MUST FIX: previously sample almost never included.
        # Mix study + sample, then take top max_ev
        merged_context = (study_context + sample_context)[:max_ev]
        retrieved_context = merged_context

        # 9) Generate answer (concise + bullets + citations)
        answer = self.generate_answer(user_text, retrieved_context, top_gse_ids)

        # 10) Debug payload (optional)
        debug_payload: Dict[str, Any] = {}
        if bool(getattr(settings, "DEBUG_RETRIEVAL", False)):
            debug_payload = self._build_debug_payload(
                query_info=query_info,
                strategy=strategy,
                filters=filters,
                effective_topk=effective_topk,
                sim_th=sim_th,
                study_hits_count=len(study_hits or []),
                gated_studies_count=len(gated_studies or []),
                sample_hits_count=sample_hits_count,
                top_sources=retrieved_context,
            )

        return {
            "messages": messages,
            "retrieved_context": retrieved_context,
            "top_gse_ids": top_gse_ids,
            "answer": answer,
            "debug": debug_payload,
        }


# -----------------------------
# Factory function (MUST HAVE)
# -----------------------------
def create_rag_chain(qdrant_client: QdrantClient) -> "RAGChain":
    return RAGChain(qdrant_client)