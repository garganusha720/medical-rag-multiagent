"""
LangGraph Multi-Agent RAG Orchestration
=======================================
Wires the 5 specialized medical agents into an executable StateGraph:
  1. Supervisor Node
  2. Retrieval Node
  3. Critic Node
  4. Synthesizer Node
  5. Citation Node

Features:
  - Conditional routing with automatic retry loop for low-relevance retrievals
  - Insufficient context handling
  - LangSmith tracing support (optional via env)
"""

import logging
import os
from typing import Any, Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from graph.state import RAGState
from agents.supervisor import supervisor_agent
from agents.retrieval_agent import retrieval_agent
from agents.critic_agent import critic_agent
from agents.synthesizer_agent import synthesizer_agent
from agents.citation_agent import citation_agent

load_dotenv()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─── Conditional Routing Logic ──────────────────────────────────
def route_after_critic(state: RAGState) -> Literal["retriever", "synthesizer"]:
    """
    Evaluates state after Critic Agent to determine next step:
    - If insufficient context or passing chunks >= 2: Proceed to Synthesizer
    - If passing chunks < 2 and retry_count < 2: Loop back to Retriever
    - If retry_count >= 2: Proceed to Synthesizer
    """
    insufficient = state.get("insufficient_context", False)
    filtered = state.get("filtered_chunks", [])
    retry_count = state.get("retry_count", 0)

    if insufficient:
        logger.info("[Router] Insufficient context flagged -> Routing to Synthesizer")
        return "synthesizer"

    if len(filtered) < 2 and retry_count < 2:
        logger.info(f"[Router] Critic rejected chunks (passing={len(filtered)}) -> Routing back to Retriever (retry #{retry_count})")
        return "retriever"

    logger.info(f"[Router] {len(filtered)} chunks passed verification -> Routing to Synthesizer")
    return "synthesizer"


# ─── Graph Construction ─────────────────────────────────────────
def build_rag_graph() -> StateGraph:
    """Construct and compile the multi-agent RAG workflow."""
    builder = StateGraph(RAGState)

    # 1. Register Nodes
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("retriever", retrieval_agent)
    builder.add_node("critic", critic_agent)
    builder.add_node("synthesizer", synthesizer_agent)
    builder.add_node("citation", citation_agent)

    # 2. Add Fixed Edges
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "retriever")
    builder.add_edge("retriever", "critic")

    # 3. Add Conditional Edge after Critic
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retriever": "retriever",
            "synthesizer": "synthesizer",
        },
    )

    # 4. Final Pipeline Edges
    builder.add_edge("synthesizer", "citation")
    builder.add_edge("citation", END)

    return builder.compile()


# Global compiled workflow instance
compiled_rag_graph = build_rag_graph()


# ─── High-Level Pipeline Runner ─────────────────────────────────
def run_medical_rag_pipeline(
    query: str,
    session_id: str = "default_session",
    chat_history: list[dict] = [],
) -> dict[str, Any]:
    """
    Execute end-to-end multi-agent medical RAG query.

    Args:
        query: User medical question
        session_id: Tracking session ID
        chat_history: Prior conversation turns

    Returns:
        Final state dictionary with 'answer', 'citations', 'citation_coverage', etc.
    """
    initial_state: RAGState = {
        "query": query,
        "session_id": session_id,
        "chat_history": chat_history,
        "query_type": "general",
        "top_k": 6,
        "retrieval_method": "hybrid",
        "source_filter": None,
        "alpha": 0.6,
        "retrieved_chunks": [],
        "filtered_chunks": [],
        "critic_scores": [],
        "retry_count": 0,
        "insufficient_context": False,
        "answer": "",
        "citations": [],
        "citation_coverage": 0.0,
        "status_message": "Initializing...",
    }

    final_state = compiled_rag_graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_query = "What are the common clinical symptoms and post-viral complications of Long COVID?"
    print(f"\nRunning test pipeline for query: '{test_query}'\n")
    result = run_medical_rag_pipeline(test_query)
    print("\n" + "=" * 60)
    print("FINAL CLINICAL ANSWER:")
    print("=" * 60)
    print(result.get("answer"))
    print("\nVERIFIED CITATIONS:")
    for c in result.get("citations", []):
        print(f"  {c['marker']} [{c['source']}] {c['title']} - {c['url']}")
    print(f"\nCitation Coverage: {result.get('citation_coverage', 0):.0%}")
