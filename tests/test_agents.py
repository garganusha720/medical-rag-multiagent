"""
Unit Tests for Multi-Agent System & LangGraph Workflow
======================================================
Tests Supervisor classification, Critic filtering & retry triggers,
Synthesizer prompt construction, and Citation resolution.
"""

import pytest
from agents.citation_agent import extract_citation_indices, count_claims_or_sentences, citation_agent
from agents.synthesizer_agent import build_context_prompt
from graph.rag_graph import route_after_critic
from graph.state import RAGState


def test_extract_citation_indices():
    """Verify regex accurately parses all variants of inline citation markers."""
    text1 = "Patients with Long COVID often experience severe fatigue [c0] and brain fog [c1]."
    assert extract_citation_indices(text1) == [0, 1]

    text2 = "Treatment includes pulmonary rehabilitation [c2] and pacing strategies [c0]."
    assert extract_citation_indices(text2) == [0, 2]

    text3 = "No citations here."
    assert extract_citation_indices(text3) == []


def test_count_claims_or_sentences():
    """Verify sentence/claim count estimation."""
    text = "First clinical finding. Second statement regarding dosage! Third sentence on outcomes?"
    assert count_claims_or_sentences(text) == 3


def test_build_context_prompt():
    """Verify context formatting for synthesizer."""
    chunks = [
        {"source": "MedlinePlus", "title": "Asthma", "text": "Inhaled corticosteroids reduce airway inflammation."},
        {"source": "PubMed", "title": "Asthma Trial", "text": "Biologic agents target severe eosinophilic asthma."},
    ]
    prompt = build_context_prompt(chunks)
    assert "[Chunk c0]" in prompt
    assert "[Chunk c1]" in prompt
    assert "Inhaled corticosteroids" in prompt
    assert "Biologic agents" in prompt


def test_citation_agent_resolution():
    """Verify citation agent resolves markers to chunk metadata and computes coverage."""
    mock_state: RAGState = {
        "query": "What are asthma treatments?",
        "session_id": "s1",
        "chat_history": [],
        "query_type": "treatment",
        "top_k": 5,
        "retrieval_method": "hybrid",
        "source_filter": None,
        "alpha": 0.6,
        "retrieved_chunks": [],
        "filtered_chunks": [
            {
                "chunk_id": "mplus_01_000",
                "doc_id": "mplus_01",
                "source": "MedlinePlus",
                "title": "Asthma Guide",
                "url": "https://medlineplus.gov/asthma.html",
                "text": "Inhaled corticosteroids are first-line controller medications for persistent asthma.",
                "critic_score": 5,
            }
        ],
        "critic_scores": [],
        "retry_count": 0,
        "insufficient_context": False,
        "answer": "Inhaled corticosteroids serve as primary controller medications for persistent asthma [c0].",
        "citations": [],
        "citation_coverage": 0.0,
        "status_message": "",
    }

    result = citation_agent(mock_state)
    citations = result["citations"]
    assert len(citations) == 1
    assert citations[0]["marker"] == "c0"
    assert citations[0]["source"] == "MedlinePlus"
    assert citations[0]["url"] == "https://medlineplus.gov/asthma.html"
    assert result["citation_coverage"] > 0.5


def test_route_after_critic():
    """Verify conditional router directs retry and continuation correctly."""
    # Case 1: Insufficient context -> Synthesizer
    state_insufficient = {"insufficient_context": True, "filtered_chunks": [], "retry_count": 2}
    assert route_after_critic(state_insufficient) == "synthesizer"

    # Case 2: Rejected chunks, retry < 2 -> Retriever
    state_retry = {"insufficient_context": False, "filtered_chunks": [], "retry_count": 0}
    assert route_after_critic(state_retry) == "retriever"

    # Case 3: Chunks pass -> Synthesizer
    state_pass = {"insufficient_context": False, "filtered_chunks": [{"id": "c1"}, {"id": "c2"}], "retry_count": 0}
    assert route_after_critic(state_pass) == "synthesizer"
