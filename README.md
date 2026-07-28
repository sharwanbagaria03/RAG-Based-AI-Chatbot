# Corrective RAG (CRAG) Chatbot & Evaluation Suite

A production-ready Corrective Retrieval-Augmented Generation (CRAG) pipeline built with **LangGraph**, **Pinecone**, **FastAPI**, **Streamlit**, and **OpenAI**. The system is strictly grounded in the knowledge base of the **[Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)**.

Includes a complete automated evaluation pipeline powered by **Ragas v0.2+** to generate synthetic test datasets and benchmark system performance across standard RAGAS and custom grader metrics.

---

## Deliverables Summary

- **Grounded CRAG Pipeline**: 3-node LangGraph workflow (`retrieve` -> `grade_contexts` -> `generate_answer`).
- **REST API & Web UI**: FastAPI backend with Swagger docs (`/docs`) and a Streamlit frontend with confidence scoring and source chunk inspectability.
- **Sample Benchmark Queries**: 6 representative test queries covering direct facts, multi-concept synthesis, and out-of-scope edge cases.
- **Automated Evaluation Suite**: Synthetic testset generator and evaluation runner using Ragas v0.2+ with concurrency controls and Knowledge Graph caching.

---

## Evaluation Benchmark Results

Evaluated against a synthetic dataset generated from the source PDF using Ragas evolutionary synthesizers:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Faithfulness** | `0.9037` | Measures if the generated response is strictly derived from retrieved context |
| **Response Relevance** | `0.8002` | Measures how directly the answer addresses the user query |
| **Context Precision** | `0.9140` | Measures signal-to-noise ratio of retrieved chunks |
| **Context Recall** | `0.8285` | Measures if all necessary ground-truth contexts were retrieved |
| **Grader Precision** | `0.9140` | Custom metric measuring relevance of chunks passed by the LLM grader |
| **Grader Recall** | `0.7608` | Custom metric measuring proportion of relevant candidate chunks retained |

---

## Architecture Overview

```
                         +--------------------------+
                         |   Ebook-Agentic-AI.pdf   |
                         +------------+-------------+
                                      | (ingest.py)
                                      v
                         +--------------------------+
                         | PyPDFLoader & Splitter   | Preserves page & source metadata
                         +------------+-------------+
                                      | OpenAI Embeddings
                                      v
                         +--------------------------+
                         |   Pinecone Vector DB     |
                         +------------+-------------+
                                      ^
                                      | Similarity Search
                                      |
+---------------+     POST /query     +---+----------------------+     Response Payload
|  Streamlit UI | ------------------> |  LangGraph StateGraph    | ───────────────────>
| / REST Client | <------------------ | 1. retrieve_node         | (Answer, Chunks,
+---------------+                     | 2. grade_contexts_node   |  Confidence Score)
                                      | 3. generate_answer_node  |
                                      +--------------------------+
```

### Pipeline Flow
1. **`retrieve_node`**: Queries Pinecone vector store for top candidate chunks using Cosine similarity.
2. **`grade_contexts_node`**: Structured LLM evaluation filters out non-relevant candidate chunks and computes a normalized confidence score (0.0 to 1.0).
3. **`generate_answer_node`**: Formulates a concise answer grounded strictly in verified context chunks. If confidence is below threshold or no relevant context exists, it cleanly refuses.

---

## Project Structure

```
├── config.py                 # Centralized configuration & environment loader
├── ingest.py                 # Automatic PDF downloader, splitter & Pinecone indexer
├── rag_graph.py              # 3-Node LangGraph state graph pipeline
├── main.py                   # FastAPI REST server
├── streamlit_app.py          # Streamlit chat interface
├── requirements.txt          # Dependency manifests
├── .env.example              # Environment key template
└── eval/                     # Evaluation Suite
    ├── generate_testset.py   # Ragas synthetic testset generator
    ├── run_evaluation.py     # Automated evaluation runner & report exporter
    ├── run_sample_queries.py # CLI sample query runner
    ├── sample_queries.json   # Benchmark sample queries
    ├── testset.json          # Generated test dataset
    ├── evaluation_report.csv # Metric results CSV export
    └── kg_cache.pkl            # Knowledge Graph cache
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key
- Pinecone API Key

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file from the provided `.env.example`:

**PowerShell (Windows)**:
```powershell
Copy-Item .env.example .env
```

**Bash (Linux / macOS / Git Bash)**:
```bash
cp .env.example .env
```

Then configure your API keys inside `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-ebook
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

---

## Running the Application

Auto-ingestion checks Pinecone on startup and populates vectors automatically if the index is empty.

### 1. Start FastAPI Backend
```bash
uvicorn main:app --reload
```
- API Endpoint: `http://localhost:8000`
- Interactive API Docs: `http://localhost:8000/docs`

### 2. Start Streamlit UI
```bash
streamlit run streamlit_app.py
```
- Web Application: `http://localhost:8501`

---

## Running the Evaluation Suite

### 1. Execute Sample Benchmark Queries (CLI)
```bash
python eval/run_sample_queries.py
```

### 2. Generate Synthetic Test Dataset
```bash
python eval/generate_testset.py
```
Generates synthetic questions using Ragas `KnowledgeGraph` and stores them in `eval/testset.json` (uses local pickle caching for zero redundant API usage).

### 3. Run Pipeline Evaluation
```bash
python eval/run_evaluation.py
```
Executes the dataset through the CRAG pipeline and outputs metric calculations to `eval/evaluation_report.csv`.

---

## Sample Queries

The system includes 6 pre-configured evaluation queries (`eval/sample_queries.json`) accessible directly in the Streamlit UI and CLI runner:

1. *"What is Agentic AI and how does it differ from traditional generative AI models?"*
2. *"What are the core architectural components of an Autonomous AI Agent?"*
3. *"How do multi-agent frameworks manage task decomposition, orchestration, and collaboration?"*
4. *"What role do Vector Databases and long-term memory play in Agentic Workflows?"*
5. *"What security, alignment, and governance challenges are associated with Agentic AI?"*
6. *"What future trends and enterprise applications of Agentic Workflows are highlighted in the book?"*
