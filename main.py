from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import logging

import config
from ingest import ingest_pdf_to_pinecone
from rag_graph import run_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context: Auto-ingest PDF into Pinecone on server startup if needed."""
    logger.info("Initializing Agentic AI RAG Backend...")
    try:
        if config.OPENAI_API_KEY and config.PINECONE_API_KEY:
            logger.info("Checking Pinecone index & auto-ingesting PDF if empty...")
            total_chunks = ingest_pdf_to_pinecone()
            logger.info(f"RAG Backend Ready! ({total_chunks} vectors in Pinecone)")
        else:
            logger.warning("API keys missing in .env. Skipping auto-ingestion.")
    except Exception as e:
        logger.error(f"Auto-ingestion check on startup failed: {e}")
    yield
    logger.info("Shutting down RAG Backend...")

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot API",
    description="LangGraph & Pinecone RAG API strictly grounded in the Agentic AI eBook knowledge base.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / Response Models ---
class QueryRequest(BaseModel):
    question: str

class ChunkResponse(BaseModel):
    content: str
    page: int
    source: str
    similarity_score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[ChunkResponse]
    confidence_score: float

class IngestResponse(BaseModel):
    message: str
    total_chunks: int

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check():
    return {
        "status": "healthy",
        "pinecone_index": config.PINECONE_INDEX_NAME,
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL
    }

@app.post("/ingest", response_model=IngestResponse, summary="Ingest Agentic AI eBook PDF into Pinecone")
def ingest_pdf():
    """Triggers downloading the Agentic AI eBook, splitting into chunks, embedding, and indexing in Pinecone."""
    try:
        total_chunks = ingest_pdf_to_pinecone()
        return IngestResponse(
            message=f"Successfully ingested and indexed PDF into Pinecone index '{config.PINECONE_INDEX_NAME}'.",
            total_chunks=total_chunks
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, summary="Query the RAG Pipeline")
def query_rag(request: QueryRequest):
    """Executes the 3-node LangGraph pipeline to retrieve chunks, grade relevance, and generate a grounded answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = run_rag_pipeline(request.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            retrieved_chunks=[ChunkResponse(**c) for c in result["retrieved_chunks"]],
            confidence_score=result["confidence_score"]
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
