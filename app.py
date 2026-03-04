from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from groq import Groq
from langchain_core.messages import HumanMessage

from typing import List, Dict, Any
from pydantic import BaseModel

# Import from src
from src.config import settings
from src.rag_chain import RAGChain, GeoState  # ✅ FIX: use RAGChain class, not create_rag_chain


# Initialize FastAPI
app = FastAPI(
    title="GEO-LLM RAG Chatbot",
    description="Biomedical dataset search using Qdrant Cloud + Groq LLM",
    version="1.0.0",
)

# CORS for team access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize clients
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)
groq_client = Groq(api_key=settings.GROQ_API_KEY)

# Initialize RAG chain
rag_chain = RAGChain(qdrant_client)  # ✅ FIX: instantiate the class


# Pydantic models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query_info: Dict[str, Any]


# Health check
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
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# Web UI
@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# API endpoint for chat
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        state: GeoState = {
            "messages": [HumanMessage(content=request.message)],
            "retrieved_context": [],
            "top_gse_ids": [],
        }

        # Run RAG
        result = rag_chain.run(state)

        # Generate answer using Groq
        answer = generate_answer(request.message, result["retrieved_context"])

        return ChatResponse(
            answer=answer,
            sources=result["retrieved_context"][:5],
            query_info={"top_gse_ids": result["top_gse_ids"]},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Form endpoint for web UI
@app.post("/chat", response_class=HTMLResponse)
async def chat_form(request: Request, message: str = Form(...)):
    try:
        state: GeoState = {
            "messages": [HumanMessage(content=message)],
            "retrieved_context": [],
            "top_gse_ids": [],
        }

        result = rag_chain.run(state)
        answer = generate_answer(message, result["retrieved_context"])

        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "query": message,
                "answer": answer,
                "sources": result["retrieved_context"][:5],
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "error": str(e),
            },
        )


def generate_answer(query: str, context: List[Dict[str, Any]]) -> str:
    """Generate final answer using Groq"""
    if not context:
        return "No relevant datasets found for this query. Please try rephrasing your question."

    evidence_lines = []
    for i, item in enumerate(context[:10], 1):
        evidence_lines.append(
            f"""Evidence {i}:
Source: {item.get('source')}
GSE: {item.get('gse_id')}
GSM: {item.get('gsm_id', 'N/A')}
Field: {item.get('field')}
Organism: {item.get('organism')}
Tissue: {item.get('tissue')}
Assay: {item.get('assay_type')}
Score: {round(float(item.get('vector_score', 0.0)), 3)}
Text: {(item.get('text', '') or '')[:500]}
"""
        )

    evidence_block = "\n".join(evidence_lines)

    prompt = f"""You are a biomedical dataset analysis assistant. Answer the user query using ONLY the provided evidence. Do NOT hallucinate.

Structure your answer:
1. Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Relevant GSE Datasets (list with brief descriptions)
4. Experimental Details (if available)

User Query: \"\"\"{query}\"\"\"

Retrieved Evidence:
{evidence_block}
"""

    try:
        response = groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}\n\nRaw context available: {len(context)} items retrieved."


# For local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)