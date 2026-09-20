"""
LangGraph Multi-Agent Pipeline State
====================================
Defines the shared state schema passed across all agent nodes in the
Medical RAG Multi-Agent graph.
"""

from typing import Annotated, Any, TypedDict
from pydantic import BaseModel, Field


class RAGState(TypedDict):
    """
    State object flowing through the LangGraph multi-agent pipeline.
    """
    # User Input & Session
    query: str
    session_id: str
    chat_history: list[dict[str, str]]

    # Supervisor Classification & Strategy
    query_type: str                  # "treatment", "symptom", "drug", "covid", "general"
    top_k: int                       # number of candidate chunks to retrieve
    retrieval_method: str            # "hybrid", "dense", "bm25", "mmr"
    source_filter: str | None        # "pubmed", "medquad", "medlineplus", or None
    alpha: float                     # hybrid dense weight (0.0 - 1.0)

    # Retrieval & Critic Evaluation
    retrieved_chunks: list[dict[str, Any]]
    filtered_chunks: list[dict[str, Any]]
    critic_scores: list[dict[str, Any]]
    retry_count: int                 # retry iteration counter (max 2)
    insufficient_context: bool       # True if evidence is inadequate to answer

    # Synthesis & Citations
    answer: str
    citations: list[dict[str, Any]]
    citation_coverage: float         # proportion of verified claims/markers (0.0 - 1.0)
    status_message: str              # agent progress update for UI streaming
