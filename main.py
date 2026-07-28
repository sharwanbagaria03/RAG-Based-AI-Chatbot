from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
from contextlib import asynccontextmanager

import config
from ingest import ingest_pdf_to_pinecone
from rag_graph import run_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG Backend...")
    try:
        if config.OPENAI_API_KEY and config.PINECONE_API_KEY:
            total_chunks = ingest_pdf_to_pinecone()
            logger.info(f"RAG Backend ready with {total_chunks} vectors.")
        else:
            logger.warning("API keys missing in environment.")
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
    yield

app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pinecone_index": config.PINECONE_INDEX_NAME,
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL
    }

@app.post("/ingest", response_model=IngestResponse)
def ingest_pdf():
    try:
        total_chunks = ingest_pdf_to_pinecone()
        return IngestResponse(
            message=f"Indexed PDF into index '{config.PINECONE_INDEX_NAME}'.",
            total_chunks=total_chunks
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
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
