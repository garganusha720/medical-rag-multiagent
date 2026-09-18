"""
Citation Agent
==============
Node 5 in LangGraph pipeline.
Parses inline citation markers [c0], [c1], etc. from the synthesized answer,
resolves them to exact source documents/chunks, and computes factual
citation coverage metrics.
"""

import logging
import re
from typing import Any

from graph.state import RAGState

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def extract_citation_indices(text: str) -> list[int]:
    """
    Extract unique citation indices from answer text.
    Matches formats like [c0], [c1], [C0], [0], [c0, c1].
    """
    matches = re.findall(r"\[c?(\d+)\]", text, re.IGNORECASE)
    indices = []
    for m in matches:
        try:
            indices.append(int(m))
        except ValueError:
            pass
    return sorted(list(set(indices)))


def count_claims_or_sentences(text: str) -> int:
    """Estimate number of factual statements/sentences in text."""
    # Split by sentence enders (. ! ?)
    sentences = re.split(r"[.!?]+", text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    return max(len(valid_sentences), 1)


def citation_agent(state: RAGState) -> dict[str, Any]:
    """
    Citation node: verifies citation markers, builds citation cards, and calculates coverage.
    """
    answer = state.get("answer", "")
    filtered_chunks = state.get("filtered_chunks", [])
    insufficient_context = state.get("insufficient_context", False)

    if insufficient_context or not answer:
        return {
            "citations": [],
            "citation_coverage": 0.0,
            "status_message": "Pipeline completed.",
        }

    # Extract citation indices from text
    cited_indices = extract_citation_indices(answer)
    logger.info(f"[Citation Agent] Found {len(cited_indices)} cited indices in answer: {cited_indices}")

    citations = []
    valid_citations_count = 0

    for idx in cited_indices:
        if 0 <= idx < len(filtered_chunks):
            chunk = filtered_chunks[idx]
            valid_citations_count += 1
            
            snippet = chunk.get("text", "").strip()
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."

            citations.append({
                "marker": f"c{idx}",
                "chunk_id": chunk.get("chunk_id", ""),
                "source": chunk.get("source", "NIH"),
                "text": snippet,
                "url": chunk.get("url", ""),
                "score": chunk.get("critic_score", 4),
            })
        else:
            logger.warning(f"[Citation Agent] Invalid citation index [c{idx}] out of bounds ({len(filtered_chunks)} chunks available)")

    # Calculate citation coverage
    total_claims = count_claims_or_sentences(answer)
    coverage = min(1.0, round(valid_citations_count / total_claims, 2)) if total_claims > 0 else 0.0

    logger.info(f"[Citation Agent] Packaged {len(citations)} verified citation cards (Coverage: {coverage:.0%})")

    return {
        "citations": citations,
        "citation_coverage": coverage,
        "status_message": f"Completed. {len(citations)} citations verified with {coverage:.0%} coverage.",
    }
