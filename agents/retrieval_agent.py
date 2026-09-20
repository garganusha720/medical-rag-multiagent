"""
Retrieval Agent
===============
Node 2 in LangGraph pipeline.
Executes hybrid dense + sparse retrieval across the verified medical corpus.
Adapts candidate pool size during retries if signaled by Critic Agent.
"""

import logging
from typing import Any

from graph.state import RAGState
from retrieval.hybrid_search import HybridRetriever

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Global singleton retriever instance for high performance
_RETRIEVER_INSTANCE: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Lazy load and cache HybridRetriever instance."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        logger.info("[Retrieval Agent] Initializing global HybridRetriever...")
        _RETRIEVER_INSTANCE = HybridRetriever(model_type="minilm")
    return _RETRIEVER_INSTANCE


def retrieval_agent(state: RAGState) -> dict[str, Any]:
    """
    Retrieval node: performs hybrid search with dynamic retry adaptation.
    """
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    base_top_k = state.get("top_k", 6)
    method = state.get("retrieval_method", "hybrid")
    alpha = state.get("alpha", 0.6)
    source_filter = state.get("source_filter")

    # If this is a retry attempt, broaden retrieval pool
    if retry_count > 0:
        effective_top_k = int(base_top_k * 1.5) + (retry_count * 3)
        effective_score_threshold = 0.0  # Loosen score threshold on retry
        logger.info(f"[Retrieval Agent] Retry #{retry_count}: Expanding top_k from {base_top_k} to {effective_top_k}")
    else:
        effective_top_k = base_top_k
        effective_score_threshold = 0.0

    retriever = get_retriever()
    results = retriever.search(
        query=query,
        top_k=effective_top_k,
        method=method,
        alpha=alpha,
        score_threshold=effective_score_threshold,
        source_filter=source_filter,
    )

    logger.info(f"[Retrieval Agent] Retrieved {len(results)} chunks for query: '{query[:60]}...'")

    return {
        "retrieved_chunks": results,
        "status_message": f"Retrieved {len(results)} medical evidence chunks from PubMed & NIH databases.",
    }
