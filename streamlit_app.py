import streamlit as st
import json
import os
import requests
import config
from rag_graph import run_rag_pipeline

# Page Config
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .confidence-badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .confidence-badge-medium {
        background-color: #FEF08A;
        color: #713F12;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .confidence-badge-low {
        background-color: #FDE8E8;
        color: #9B1C1C;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .chunk-card {
        border-left: 3px solid #6366F1;
        background-color: #F9FAFB;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load sample queries
@st.cache_data
def load_sample_queries():
    if os.path.exists("sample_queries.json"):
        with open("sample_queries.json", "r") as f:
            return json.load(f)
    return []

sample_queries = load_sample_queries()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-Ingest Check on Initial Load
if "ingested" not in st.session_state:
    try:
        from ingest import ingest_pdf_to_pinecone
        ingest_pdf_to_pinecone()
        st.session_state["ingested"] = True
    except Exception as e:
        pass

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Settings & Ingestion")
    st.markdown("---")
    
    st.subheader("⚙️ System Status")
    openai_ok = bool(config.OPENAI_API_KEY)
    pinecone_ok = bool(config.PINECONE_API_KEY)
    
    st.write(f"**OpenAI API Key:** {'✅ Configured' if openai_ok else '❌ Missing'}")
    st.write(f"**Pinecone API Key:** {'✅ Configured' if pinecone_ok else '❌ Missing'}")
    st.write(f"**Index:** `{config.PINECONE_INDEX_NAME}`")
    st.write(f"**LLM:** `{config.LLM_MODEL}`")
    
    st.markdown("---")
    st.subheader("📥 Ingestion Control")
    if st.button("Trigger PDF Ingestion", use_container_width=True):
        with st.spinner("Downloading PDF and indexing into Pinecone..."):
            try:
                from ingest import ingest_pdf_to_pinecone
                total = ingest_pdf_to_pinecone()
                st.success(f"Ingested {total} chunks into Pinecone!")
            except Exception as err:
                st.error(f"Ingestion Error: {err}")
                
    st.markdown("---")
    st.subheader("💡 Sample Queries")
    st.caption("Click any query below to run it:")
    
    selected_sample = None
    for sq in sample_queries:
        if st.button(f"📌 {sq['question']}", key=f"sq_{sq['id']}", use_container_width=True):
            selected_sample = sq['question']

# Header
st.markdown('<div class="main-header">📖 Agentic AI eBook Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">LangGraph & Pinecone RAG Pipeline strictly grounded in '
    '<a href="https://konverge.ai/pdf/Ebook-Agentic-AI.pdf" target="_blank">Ebook-Agentic-AI.pdf</a></div>',
    unsafe_allow_html=True
)

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "confidence_score" in msg:
            score = msg["confidence_score"]
            if score >= 0.75:
                badge_html = f'<span class="confidence-badge-high">🟢 High Confidence ({score:.0%})</span>'
            elif score >= 0.4:
                badge_html = f'<span class="confidence-badge-medium">🟡 Medium Confidence ({score:.0%})</span>'
            else:
                badge_html = f'<span class="confidence-badge-low">🔴 Low Confidence ({score:.0%})</span>'
                
            st.markdown(badge_html, unsafe_allow_html=True)
            
            # Display Context Chunks Drawer
            if msg.get("retrieved_chunks"):
                with st.expander(f"📚 View Retrieved Context Chunks ({len(msg['retrieved_chunks'])} sources)"):
                    for idx, chunk in enumerate(msg["retrieved_chunks"], 1):
                        st.markdown(f"**Source {idx} (Page {chunk['page']}, Similarity Score: {chunk['similarity_score']:.2f})**")
                        st.text(chunk["content"])
                        st.markdown("---")

# Process Prompt Input (User text input or Sample Query selection)
user_prompt = st.chat_input("Ask a question about Agentic AI...")
if selected_sample:
    user_prompt = selected_sample

if user_prompt:
    # Add User Message to State & Display
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process via RAG Graph
    with st.chat_message("assistant"):
        with st.spinner("Retrieving contexts & generating grounded response..."):
            try:
                # Run RAG Pipeline directly
                res = run_rag_pipeline(user_prompt)
                
                answer = res["answer"]
                confidence = res["confidence_score"]
                chunks = res["retrieved_chunks"]
                
                st.markdown(answer)
                
                # Confidence badge
                if confidence >= 0.75:
                    badge_html = f'<span class="confidence-badge-high">🟢 High Confidence ({confidence:.0%})</span>'
                elif confidence >= 0.4:
                    badge_html = f'<span class="confidence-badge-medium">🟡 Medium Confidence ({confidence:.0%})</span>'
                else:
                    badge_html = f'<span class="confidence-badge-low">🔴 Low Confidence ({confidence:.0%})</span>'
                st.markdown(badge_html, unsafe_allow_html=True)
                
                if chunks:
                    with st.expander(f"📚 View Retrieved Context Chunks ({len(chunks)} sources)"):
                        for idx, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Source {idx} (Page {chunk['page']}, Similarity Score: {chunk['similarity_score']:.2f})**")
                            st.text(chunk["content"])
                            st.markdown("---")
                            
                # Save assistant response to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "confidence_score": confidence,
                    "retrieved_chunks": chunks
                })
            except Exception as e:
                st.error(f"Error generating answer: {e}")
