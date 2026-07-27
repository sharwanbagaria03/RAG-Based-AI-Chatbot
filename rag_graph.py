from typing import List, Dict, Any, TypedDict
import logging
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
import config

logger = logging.getLogger(__name__)

# --- State Definition ---
class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Dict[str, Any]]
    graded_docs: List[Dict[str, Any]]
    confidence_score: float
    final_answer: str

# Pydantic Schema for LLM Grading
class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="True if the context chunk is relevant to the question, False otherwise.")
    relevance_reason: str = Field(description="Short rationale for the relevance evaluation.")

def get_vector_store():
    """Helper to initialize Pinecone vector store."""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=config.PINECONE_API_KEY
    )

# --- Node 1: Retrieve ---
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    logger.info(f"[LangGraph Node: Retrieve] Fetching context for query: '{question}'")
    
    try:
        vector_store = get_vector_store()
        # Retrieve top k documents with relevance scores
        results = vector_store.similarity_search_with_relevance_scores(question, k=config.TOP_K_RESULTS)
        
        retrieved_docs = []
        for doc, score in results:
            # Format doc metadata cleanly
            page_num = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
            source = doc.metadata.get("source", config.LOCAL_PDF_PATH)
            
            retrieved_docs.append({
                "content": doc.page_content,
                "page": page_num,
                "source": source,
                "similarity_score": round(float(score), 4) if score is not None else 0.0,
                "metadata": doc.metadata
            })
            
        logger.info(f"[LangGraph Node: Retrieve] Retrieved {len(retrieved_docs)} chunks.")
        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"retrieved_docs": []}

# --- Node 2: Grade Contexts & Score Confidence ---
def grade_contexts_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])
    
    logger.info(f"[LangGraph Node: Grade] Evaluating {len(retrieved_docs)} chunks for relevance...")
    
    if not retrieved_docs:
        return {
            "graded_docs": [],
            "confidence_score": 0.0
        }
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0.0,
        openai_api_key=config.OPENAI_API_KEY
    ).with_structured_output(RelevanceGrade)
    
    graded_docs = []
    relevance_scores = []
    
    for doc in retrieved_docs:
        prompt = f"""You are a strict relevance grader evaluating retrieved chunks from the Agentic AI eBook.
User Question: {question}
Retrieved Chunk Context:
{doc['content']}

Evaluate if this chunk contains facts, concepts, or details relevant to answering the question.
"""
        try:
            grade: RelevanceGrade = llm.invoke(prompt)
            if grade.is_relevant:
                graded_docs.append(doc)
                relevance_scores.append(doc["similarity_score"])
                logger.info(f"Chunk (Page {doc['page']}) graded RELEVANT. Reason: {grade.relevance_reason}")
            else:
                logger.info(f"Chunk (Page {doc['page']}) graded IRRELEVANT. Reason: {grade.relevance_reason}")
        except Exception as err:
            logger.warning(f"Grading fallback for chunk: {err}")
            # Fallback to vector similarity score if LLM grade fails
            if doc["similarity_score"] > 0.4:
                graded_docs.append(doc)
                relevance_scores.append(doc["similarity_score"])

    # Calculate overall confidence score
    if graded_docs:
        avg_similarity = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        relevance_ratio = len(graded_docs) / len(retrieved_docs)
        confidence_score = round((avg_similarity * 0.6) + (relevance_ratio * 0.4), 2)
        # Cap confidence score between 0.0 and 1.0
        confidence_score = max(0.0, min(1.0, confidence_score))
    else:
        confidence_score = 0.0
        
    logger.info(f"[LangGraph Node: Grade] Graded docs: {len(graded_docs)}/{len(retrieved_docs)}, Confidence Score: {confidence_score}")
    return {
        "graded_docs": graded_docs,
        "confidence_score": confidence_score
    }

# --- Node 3: Generate Grounded Answer ---
def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    graded_docs = state.get("graded_docs", [])
    confidence_score = state.get("confidence_score", 0.0)
    
    logger.info(f"[LangGraph Node: Generate] Generating answer for query (confidence={confidence_score})...")
    
    # Strict refusal if confidence is zero or no relevant contexts
    if not graded_docs or confidence_score < 0.15:
        refusal_answer = (
            "I'm sorry, but the provided Agentic AI eBook knowledge base does not contain "
            "sufficient relevant information to answer your question accurately."
        )
        return {"final_answer": refusal_answer}

    # Format context string
    context_str = "\n\n---\n\n".join([
        f"[Page {doc['page']}]: {doc['content']}" for doc in graded_docs
    ])

    system_prompt = (
        "You are an expert AI Assistant answering user questions STRICTLY based on the provided Agentic AI eBook context.\n"
        "Rules:\n"
        "1. Rely ONLY on the clear facts contained directly in the context below.\n"
        "2. Do NOT extrapolate or assume information not supported by the context.\n"
        "3. If the context does not contain enough information to answer fully, state what is known and what is missing.\n"
        "4. Include concise page references (e.g. [Page X]) when referencing specific concepts.\n\n"
        f"Context:\n{context_str}"
    )

    user_prompt = f"Question: {question}"

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0.2,
        openai_api_key=config.OPENAI_API_KEY
    )

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        final_answer = response.content.strip()
    except Exception as e:
        logger.error(f"Generation error: {e}")
        final_answer = "An error occurred while generating the answer. Please try again."

    return {"final_answer": final_answer}

# --- Build LangGraph Pipeline ---
def build_rag_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_contexts", grade_contexts_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # Add Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_contexts")
    workflow.add_edge("grade_contexts", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

# Instantiated Graph App
rag_pipeline = build_rag_graph()

def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """Execute the compiled LangGraph pipeline for a user question."""
    initial_state = {
        "question": question,
        "retrieved_docs": [],
        "graded_docs": [],
        "confidence_score": 0.0,
        "final_answer": ""
    }
    
    result = rag_pipeline.invoke(initial_state)
    
    # Return structured output with answer, context chunks, and confidence score
    chunks_output = [
        {
            "content": doc["content"],
            "page": doc["page"],
            "source": doc["source"],
            "similarity_score": doc.get("similarity_score", 0.0)
        }
        for doc in result.get("graded_docs", result.get("retrieved_docs", []))
    ]
    
    return {
        "question": question,
        "answer": result.get("final_answer", ""),
        "retrieved_chunks": chunks_output,
        "confidence_score": result.get("confidence_score", 0.0)
    }

if __name__ == "__main__":
    # Quick standalone test execution if called directly
    test_q = "What is Agentic AI?"
    print(f"Testing RAG Graph with query: '{test_q}'...")
    res = run_rag_pipeline(test_q)
    print("\n--- Output ---")
    print("Answer:", res["answer"])
    print("Confidence:", res["confidence_score"])
    print("Chunks count:", len(res["retrieved_chunks"]))
