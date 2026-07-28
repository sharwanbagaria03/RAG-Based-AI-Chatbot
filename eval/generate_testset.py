import os
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure parent directory is in sys.path for importing config and rag_graph
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import types
import pickle
import logging
import requests
import warnings
from typing import List

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import langchain_community.chat_models.vertexai
except (ImportError, ModuleNotFoundError):
    mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI:
        pass
    mock_vertex.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_pdf_if_needed(url: str, local_path: str) -> str:
    if os.path.exists(local_path):
        return local_path
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path

def get_or_build_knowledge_graph(docs, generator_llm, generator_embeddings):
    cache_file = os.path.join(os.path.dirname(__file__), "kg_cache.pkl")
    
    if os.path.exists(cache_file):
        logger.info("Loading Knowledge Graph from local cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
            
    logger.info("Building Knowledge Graph from loaded document nodes...")
    kg = KnowledgeGraph()
    for doc in docs:
        node = Node(
            type="document",
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata
            }
        )
        kg.nodes.append(node)

    transforms = default_transforms(
        documents=docs,
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    apply_transforms(kg, transforms)
    
    with open(cache_file, "wb") as f:
        pickle.dump(kg, f)
        
    return kg

def generate_synthetic_testset(output_path: str = None, test_size: int = 30):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "testset.json")

    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required in .env file.")

    pdf_path = download_pdf_if_needed(config.PDF_URL, config.LOCAL_PDF_PATH)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=config.LLM_MODEL, openai_api_key=config.OPENAI_API_KEY)
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY)
    )

    kg = get_or_build_knowledge_graph(docs, generator_llm, generator_embeddings)

    safe_config = RunConfig(max_workers=4, max_retries=10)
    
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.50),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25)
    ]

    logger.info(f"Generating synthetic test dataset ({test_size} samples)...")
    dataset = generator.generate(
        testset_size=test_size,
        query_distribution=query_distribution,
        run_config=safe_config
    )

    export_samples = []
    for sample in dataset:
        dict_sample = sample.to_dict() if hasattr(sample, "to_dict") else sample
        export_samples.append(dict_sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_samples, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported testset ({len(export_samples)} samples) to '{output_path}'.")
    return export_samples

if __name__ == "__main__":
    generate_synthetic_testset()
