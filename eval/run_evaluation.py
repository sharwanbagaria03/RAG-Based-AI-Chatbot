import os
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure parent directory is in sys.path for importing config and rag_graph
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import types
import logging
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import langchain_community.chat_models.vertexai
except (ImportError, ModuleNotFoundError):
    mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI:
        pass
    mock_vertex.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.run_config import RunConfig
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import config
from rag_graph import run_rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def is_chunk_ground_truth_relevant(chunk_text: str, ground_truth_contexts: List[str], embeddings_model: OpenAIEmbeddings, threshold: float = 0.65) -> bool:
    if not ground_truth_contexts or not chunk_text.strip():
        return False
    
    for gt in ground_truth_contexts:
        if chunk_text in gt or gt in chunk_text:
            return True
            
    try:
        chunk_vec = np.array(embeddings_model.embed_query(chunk_text))
        for gt in ground_truth_contexts:
            gt_vec = np.array(embeddings_model.embed_query(gt))
            sim = cosine_similarity(chunk_vec, gt_vec)
            if sim >= threshold:
                return True
    except Exception as e:
        logger.warning(f"Embedding similarity check error: {e}")
        
    return False

def calculate_grader_metrics(retrieved_chunks: List[Dict[str, Any]], graded_chunks: List[Dict[str, Any]], ground_truth_contexts: List[str], embeddings_model: OpenAIEmbeddings) -> Dict[str, float]:
    if not retrieved_chunks:
        return {"grader_precision": 0.0, "grader_recall": 0.0}

    rel_retrieved_count = sum(
        1 for c in retrieved_chunks
        if is_chunk_ground_truth_relevant(c["content"], ground_truth_contexts, embeddings_model)
    )

    rel_graded_count = sum(
        1 for c in graded_chunks
        if is_chunk_ground_truth_relevant(c["content"], ground_truth_contexts, embeddings_model)
    )

    total_graded_count = len(graded_chunks)

    grader_precision = (rel_graded_count / total_graded_count) if total_graded_count > 0 else 1.0
    grader_recall = (rel_graded_count / rel_retrieved_count) if rel_retrieved_count > 0 else 1.0

    return {
        "grader_precision": round(grader_precision, 4),
        "grader_recall": round(grader_recall, 4)
    }

async def run_pipeline_evaluation(testset_path: str = None, output_csv: str = None):
    eval_dir = os.path.dirname(__file__)
    if testset_path is None:
        testset_path = os.path.join(eval_dir, "testset.json")
    if output_csv is None:
        output_csv = os.path.join(eval_dir, "evaluation_report.csv")

    if not os.path.exists(testset_path):
        raise FileNotFoundError(f"Testset file '{testset_path}' not found.")

    with open(testset_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    logger.info(f"Loaded {len(test_samples)} test samples from '{testset_path}'. Executing pipeline...")

    eval_llm = LangchainLLMWrapper(ChatOpenAI(model=config.LLM_MODEL, openai_api_key=config.OPENAI_API_KEY))
    eval_embeddings_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY))
    raw_embeddings_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY)

    single_turn_samples = []
    custom_grader_metrics_list = []
    sample_records = []

    for idx, sample in enumerate(test_samples, 1):
        eval_data = sample.get("eval_sample", sample)
        
        question = eval_data.get("user_input") or eval_data.get("question") or ""
        reference = eval_data.get("reference") or eval_data.get("ground_truth") or ""
        reference_contexts = eval_data.get("reference_contexts") or eval_data.get("contexts") or []
        
        if not question:
            logger.warning(f"Skipping Sample {idx}: No question found.")
            continue

        logger.info(f"[{idx}/{len(test_samples)}] Invoking CRAG Pipeline for: '{question[:60]}...'")

        pipeline_output = run_rag_pipeline(question)
        answer = pipeline_output["answer"]
        graded_chunks = pipeline_output["retrieved_chunks"]
        retrieved_contexts = [c["content"] for c in graded_chunks]

        single_sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts if retrieved_contexts else ["No relevant context found."],
            reference=reference,
            reference_contexts=reference_contexts
        )
        single_turn_samples.append(single_sample)

        grader_metrics = calculate_grader_metrics(
            retrieved_chunks=pipeline_output["initial_retrieved_chunks"],
            graded_chunks=graded_chunks,
            ground_truth_contexts=reference_contexts,
            embeddings_model=raw_embeddings_model
        )
        custom_grader_metrics_list.append(grader_metrics)

        sample_records.append({
            "question": question,
            "answer": answer,
            "ground_truth": reference,
            "grader_precision": grader_metrics["grader_precision"],
            "grader_recall": grader_metrics["grader_recall"]
        })

    eval_dataset = EvaluationDataset(samples=single_turn_samples)

    metrics = [
        Faithfulness(llm=eval_llm),
        ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings_wrapper),
        LLMContextPrecisionWithReference(llm=eval_llm),
        LLMContextRecall(llm=eval_llm)
    ]

    logger.info("Executing Ragas Evaluation framework...")
    safe_config = RunConfig(max_workers=4, max_retries=10)
    ragas_results = evaluate(dataset=eval_dataset, metrics=metrics, run_config=safe_config)
    
    df_ragas = ragas_results.to_pandas()
    df_custom = pd.DataFrame(custom_grader_metrics_list)

    df_combined = pd.concat([df_ragas, df_custom], axis=1)
    df_combined.to_csv(output_csv, index=False)
    logger.info(f"Evaluation report saved to '{output_csv}'.")

    summary_scores = {
        "faithfulness": round(float(df_combined["faithfulness"].mean()), 4) if "faithfulness" in df_combined else 0.0,
        "response_relevance": round(float(df_combined["response_relevance"].mean() if "response_relevance" in df_combined else (df_combined["answer_relevancy"].mean() if "answer_relevancy" in df_combined else 0.0)), 4),
        "context_precision": round(float(df_combined["llm_context_precision_with_reference"].mean() if "llm_context_precision_with_reference" in df_combined else (df_combined["context_precision"].mean() if "context_precision" in df_combined else 0.0)), 4),
        "context_recall": round(float(df_combined["context_recall"].mean()), 4) if "context_recall" in df_combined else 0.0,
        "grader_precision": round(float(df_combined["grader_precision"].mean()), 4),
        "grader_recall": round(float(df_combined["grader_recall"].mean()), 4)
    }

    print("\n" + "=" * 60)
    print("EVALUATION SCORES SUMMARY")
    print("=" * 60)
    print(json.dumps(summary_scores, indent=4))
    print("=" * 60 + "\n")

    return summary_scores

if __name__ == "__main__":
    asyncio.run(run_pipeline_evaluation())
