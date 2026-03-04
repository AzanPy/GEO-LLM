import json
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Import from src
from src.config import settings
from src.rag_chain import create_rag_chain, GeoState


# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="GEO-LLM RAG Chatbot",
    description="Biomedical dataset search using Qdrant Cloud + Groq LLM",
    version="1.0.0",
)

# CORS (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# -----------------------------
# Clients
# -----------------------------
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

rag_chain = create_rag_chain(qdrant_client)


# -----------------------------
# Pydantic models
# -----------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query_info: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
async def health_check():
    try:
        collections = qdrant_client.get_collections()
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": [c.name for c in collections.collections],
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# -----------------------------
# Web UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# -----------------------------
# API endpoint (non-streaming)
# -----------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        state: GeoState = {
            "messages": [HumanMessage(content=req.message)],
            "retrieved_context": [],
            "top_gse_ids": [],
        }

        result = rag_chain.run(state)

        max_ev = int(getattr(settings, "MAX_EVIDENCE", 5))
        answer = (result.get("answer") or "").strip()
        sources = (result.get("retrieved_context") or [])[:max_ev]
        top_gse_ids = result.get("top_gse_ids") or []

        payload: Dict[str, Any] = {
            "answer": answer,
            "sources": sources,
            "query_info": {"top_gse_ids": top_gse_ids},
        }

        # Only include debug when enabled
        if bool(getattr(settings, "DEBUG_RETRIEVAL", False)):
            payload["debug"] = result.get("debug") or {}

        return ChatResponse(**payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# API endpoint (streaming SSE)
# -----------------------------
@app.post("/api/chat/stream")
async def chat_endpoint_stream(req: ChatRequest):
    """
    SSE streaming.
    Note: This streams AFTER the final answer is produced (chunked),
    so it works even if Groq token-streaming isn't enabled.
    """

    async def event_gen():
        try:
            state: GeoState = {
                "messages": [HumanMessage(content=req.message)],
                "retrieved_context": [],
                "top_gse_ids": [],
            }

            result = rag_chain.run(state)

            max_ev = int(getattr(settings, "MAX_EVIDENCE", 5))
            answer = (result.get("answer") or "").strip()
            sources = (result.get("retrieved_context") or [])[:max_ev]
            top_gse_ids = result.get("top_gse_ids") or []
            debug = result.get("debug") or {}

            # 1) Send metadata first
            meta: Dict[str, Any] = {
                "type": "meta",
                "top_gse_ids": top_gse_ids,
                "sources": sources,
            }
            if bool(getattr(settings, "DEBUG_RETRIEVAL", False)):
                meta["debug"] = debug

            yield f"data: {json.dumps(meta)}\n\n"

            # 2) Stream answer chunks
            if not bool(getattr(settings, "ENABLE_STREAMING", True)):
                yield f"data: {json.dumps({'type': 'chunk', 'text': answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            chunk_size = 40
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i : i + chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0.02)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# -----------------------------
# Form endpoint (web UI submit)
# -----------------------------
@app.post("/chat", response_class=HTMLResponse)
async def chat_form(request: Request, message: str = Form(...)):
    try:
        state: GeoState = {
            "messages": [HumanMessage(content=message)],
            "retrieved_context": [],
            "top_gse_ids": [],
        }

        result = rag_chain.run(state)

        max_ev = int(getattr(settings, "MAX_EVIDENCE", 5))
        answer = (result.get("answer") or "").strip()
        sources = (result.get("retrieved_context") or [])[:max_ev]
        debug = result.get("debug") if bool(getattr(settings, "DEBUG_RETRIEVAL", False)) else None

        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "query": message,
                "answer": answer,
                "sources": sources,
                "debug": debug,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "chat.html",
            {"request": request, "error": str(e)},
        )


# -----------------------------
# Local run
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)