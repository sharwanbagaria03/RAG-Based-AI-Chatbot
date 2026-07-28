import json
import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_graph import run_rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    sample_file = os.path.join(os.path.dirname(__file__), "sample_queries.json")
    if not os.path.exists(sample_file):
        print(f"Error: {sample_file} not found.")
        sys.exit(1)

    with open(sample_file, "r") as f:
        queries = json.load(f)

    print("=" * 80)
    print("SAMPLE QUERIES EVALUATION SUITE")
    print("=" * 80)

    for item in queries:
        qid = item["id"]
        category = item["category"]
        question = item["question"]

        print(f"\n--------------------------------------------------------------------------------")
        print(f"Query #{qid} [{category}]: {question}")
        print(f"--------------------------------------------------------------------------------")

        try:
            res = run_rag_pipeline(question)
            
            print(f"ANSWER:")
            print(res["answer"])
            print(f"\nMETRICS & RETRIEVAL SUMMARY:")
            print(f"   • Confidence Score: {res['confidence_score']:.2f} ({res['confidence_score']:.0%})")
            print(f"   • Context Chunks Retrieved & Graded: {len(res['retrieved_chunks'])}")
            
            for idx, chunk in enumerate(res["retrieved_chunks"], 1):
                print(f"     [{idx}] Page {chunk['page']} | Score: {chunk['similarity_score']:.2f} | Chunk Snippet: {chunk['content'][:120].strip()}...")
        except Exception as e:
            print(f"Error processing query #{qid}: {e}")

    print("\n" + "=" * 80)
    print("Evaluation Completed for all sample queries!")
    print("=" * 80)

if __name__ == "__main__":
    main()
