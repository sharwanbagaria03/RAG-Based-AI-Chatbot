import os
import requests
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_pdf(url: str, output_path: str) -> str:
    """Download the target PDF if it does not already exist locally."""
    if os.path.exists(output_path):
        logger.info(f"PDF already exists locally at: {output_path}")
        return output_path
    
    logger.info(f"Downloading PDF from {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    logger.info(f"PDF successfully downloaded and saved to: {output_path}")
    return output_path

def ensure_pinecone_index(api_key: str, index_name: str, dimension: int = 1536):
    """Ensure Pinecone index exists using official v3 SDK."""
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is missing. Please configure it in .env file.")
    
    pc = Pinecone(api_key=api_key)
    existing_indexes = [index_info.name for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index '{index_name}' (dimension={dimension}, metric='cosine')...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"Index '{index_name}' created successfully.")
    else:
        logger.info(f"Pinecone index '{index_name}' already exists.")

def ingest_pdf_to_pinecone(force: bool = False):
    """Main ingestion workflow: Download PDF, split into chunks with preserved metadata, and index in Pinecone."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is missing. Please configure it in .env file.")
    
    # Ensure Pinecone index exists
    ensure_pinecone_index(config.PINECONE_API_KEY, config.PINECONE_INDEX_NAME)
    
    # Check if vectors already exist in Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    vector_count = stats.total_vector_count or 0

    if vector_count > 0 and not force:
        logger.info(f"Pinecone index '{config.PINECONE_INDEX_NAME}' already contains {vector_count} vectors. Skipping re-embedding.")
        return vector_count

    pdf_path = download_pdf(config.PDF_URL, config.LOCAL_PDF_PATH)
    
    logger.info(f"Loading PDF via PyPDFLoader: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    logger.info(f"Loaded {len(raw_docs)} pages from PDF.")

    logger.info(f"Chunking document (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    # Preserve and verify metadata for all chunks
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        # Ensure page number metadata is 1-indexed for display
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    logger.info(f"Created {len(chunks)} text chunks with preserved metadata (e.g. source, page, page_number, start_index).")

    # Generate embeddings and store in Pinecone
    logger.info(f"Generating embeddings using '{config.EMBEDDING_MODEL}' and uploading vectors to Pinecone...")
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
        pinecone_api_key=config.PINECONE_API_KEY
    )

    logger.info("PDF Ingestion completed successfully!")
    return len(chunks)

if __name__ == "__main__":
    import sys
    force_ingest = "--force" in sys.argv
    try:
        total_chunks = ingest_pdf_to_pinecone(force=force_ingest)
        print(f"\n✅ Ingestion complete! Total vectors in '{config.PINECONE_INDEX_NAME}': {total_chunks}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
