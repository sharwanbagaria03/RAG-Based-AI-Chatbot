# 📖 Agentic AI RAG Chatbot

An enterprise-grade, grounded RAG (Retrieval-Augmented Generation) chatbot system built in Python using **LangGraph**, **Pinecone (v3 SDK)**, **OpenAI Embeddings & LLM**, **FastAPI**, and **Streamlit**. 

The chatbot answers user queries strictly grounded in the knowledge base of the **[Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)**.

---

## 🌟 Key Features

1. **Strictly Grounded RAG Pipeline (LangGraph)**:
   - Uses a **3-node StateGraph** workflow (`Retrieve` $\rightarrow$ `Grade & Score Contexts` $\rightarrow$ `Grounded Generator`).
   - Prevents hallucinations by grading chunk relevance and refusing cleanly if sufficient facts are not present in the PDF.
   - Calculates a normalized **Confidence Score** for every response.

2. **Pinecone Vector Database (Official `pinecone` v3 SDK)**:
   - Preserves complete document metadata (`source`, `page_number`, `chunk_id`, `start_index`) during PyPDF loading and text splitting.
   - Uses `text-embedding-3-small` embeddings with Cosine similarity.

3. **Multi-Interface Access**:
   - **FastAPI REST API**: Endpoints for PDF Ingestion (`POST /ingest`), Health check (`GET /health`), and Querying (`POST /query`).
   - **Streamlit Web UI**: Interactive chat interface with real-time confidence badges, expandable context chunk drawers, and quick-click sample query buttons.
   - **Sample Queries CLI Evaluator**: Automated execution script for 6 representative test queries.

---

## 🏗️ Architecture Explanation

```
                          ┌───────────────────────────┐
                          │   Ebook-Agentic-AI.pdf    │
                          └─────────────┬─────────────┘
                                        │ (ingest.py)
                                        ▼
                          ┌───────────────────────────┐
                          │ PyPDFLoader & Splitter    │ (Preserves Metadata: page, source)
                          └─────────────┬─────────────┘
                                        │ OpenAI Embeddings
                                        ▼
                          ┌───────────────────────────┐
                          │   Pinecone Vector DB      │
                          └─────────────┬─────────────┘
                                        ▲
                                        │ Similarity Search
                                        │
┌───────────────┐     POST /query     ┌─┴────────────────────────┐     Response Payload
│  Streamlit UI │ ──────────────────► │  LangGraph StateGraph   │ ───────────────────────►
│ / REST Client │ ◄────────────────── │ 1. Retrieve              │ (Answer + Context Chunks
└───────────────┘                     │ 2. Grade & Score Context │  + Confidence Score)
                                      │ 3. Grounded Generator    │
                                      └──────────────────────────┘
```

### LangGraph Workflow Nodes:
1. **Retrieve Node**: Connects to Pinecone to query top $K=4$ candidate document chunks with cosine relevance scores.
2. **Grade Contexts Node**: Evaluates document relevance against the user query, filters out non-relevant chunks, and calculates an overall normalized `confidence_score` (0.0 to 1.0).
3. **Generate Grounded Answer Node**: Uses `gpt-4o-mini` with a strict system prompt to formulate the final answer using *only* verified PDF chunks. If context is lacking or confidence is below threshold, returns a clean refusal message.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key
- Pinecone API Key (Free tier supported)

### 2. Clone & Setup Virtual Environment
```bash
# Navigate to project directory
cd "c:\Users\91916\OneDrive\Desktop\Appening infotech"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
In `.env`:
```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=agentic-ai-ebook
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

---

## 🚀 Running the Application

You only need **2 terminals** to run the complete system. Ingestion, PDF downloading, and Pinecone index checks are handled **automatically on startup**!

### Terminal 1: Run Backend API
```powershell
uvicorn main:app --reload
```
- Server starts on `http://localhost:8000`.
- Automatically downloads PDF & indexes vectors in Pinecone if not already present.
- Swagger API docs available at `http://localhost:8000/docs`.

### Terminal 2: Run Frontend Streamlit UI
```powershell
streamlit run streamlit_app.py
```
- Opens interactive Chat UI at `http://localhost:8501`.

---

## 🧪 Running Sample Queries Evaluation Suite

To test the RAG chatbot against 6 representative questions from the Agentic AI eBook:
```bash
python run_sample_queries.py
```

### 📌 Included Sample Queries (`sample_queries.json`):
1. **Core Concept**: *"What is Agentic AI and how does it differ from traditional generative AI models?"*
2. **Architecture**: *"What are the core architectural components of an Autonomous AI Agent?"*
3. **Multi-Agent Systems**: *"How do multi-agent frameworks manage task decomposition, orchestration, and collaboration?"*
4. **Memory & Vector DBs**: *"What role do Vector Databases and long-term memory play in Agentic Workflows?"*
5. **Governance & Security**: *"What security, alignment, and governance challenges are associated with Agentic AI?"*
6. **Future Outlook**: *"What future trends and enterprise applications of Agentic Workflows are highlighted in the book?"*

---

## 📂 Project Structure

```
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies (official pinecone v3 SDK)
├── config.py                 # Configuration loader & constants
├── ingest.py                 # PDF Downloader, Metadata-Preserved Chunking & Pinecone Indexing
├── rag_graph.py              # 3-Node LangGraph RAG StateGraph Pipeline
├── main.py                   # FastAPI REST API Application
├── streamlit_app.py          # Interactive Streamlit Web UI
├── sample_queries.json       # 6 Representative Evaluation Queries
├── run_sample_queries.py     # CLI evaluation runner script
└── README.md                 # Project documentation & architecture overview
```
