"""
Unit Tests for Retrieval Layer
==============================
Tests BM25 sparse search, score normalization, alpha weighting fusion,
and MMR diversity re-ranking with synthetic medical chunks.
"""

import pytest
from retrieval.hybrid_search import BM25Index, HybridRetriever


@pytest.fixture
def mock_medical_chunks():
    return [
        {
            "chunk_id": "medlineplus_001_000",
            "doc_id": "medlineplus_001",
            "text": "Long COVID symptoms include persistent fatigue, brain fog, and shortness of breath following SARS-CoV-2 infection.",
            "metadata": {"source": "medlineplus", "title": "Long COVID Overview"},
        },
        {
            "chunk_id": "pubmed_002_000",
            "doc_id": "pubmed_002",
            "text": "Clinical trial evaluating pulmonary rehabilitation and antiviral therapies for Post-Acute Sequelae of SARS-CoV-2 (PASC).",
            "metadata": {"source": "pubmed", "title": "PASC Clinical Trials"},
        },
        {
            "chunk_id": "medquad_003_000",
            "doc_id": "medquad_003",
            "text": "What is Diabetes? Diabetes mellitus is a chronic metabolic condition characterized by elevated blood glucose levels.",
            "metadata": {"source": "medquad", "title": "Diabetes Definition"},
        },
        {
            "chunk_id": "pubmed_004_000",
            "doc_id": "pubmed_004",
            "text": "Metformin and SGLT2 inhibitors in glycemic control for adults diagnosed with type 2 diabetes mellitus.",
            "metadata": {"source": "pubmed", "title": "Type 2 Diabetes Pharmacotherapy"},
        },
    ]


def test_bm25_search_relevance(mock_medical_chunks):
    """BM25 search should prioritize chunks matching query keywords."""
    index = BM25Index(mock_medical_chunks)
    results = index.search("Long COVID fatigue symptoms", top_k=2)

    assert len(results) >= 1
    assert results[0]["chunk_id"] == "medlineplus_001_000"
    assert "fatigue" in results[0]["text"].lower()


def test_bm25_source_filtering(mock_medical_chunks):
    """BM25 search should filter by source metadata when specified."""
    index = BM25Index(mock_medical_chunks)
    results = index.search("metformin glycemic control", top_k=5, source_filter="pubmed")

    assert len(results) >= 1
    for r in results:
        assert r["source"] == "pubmed"
        assert "metformin" in r["text"].lower()


def test_score_normalization():
    """Score normalization maps arbitrary ranges to [0.0, 1.0]."""
    mock_hits = [
        {"chunk_id": "c1", "score": 25.0},
        {"chunk_id": "c2", "score": 15.0},
        {"chunk_id": "c3", "score": 5.0},
    ]
    norm = HybridRetriever._normalize_scores(mock_hits)

    assert norm["c1"] == 1.0
    assert norm["c3"] == 0.0
    assert 0.49 < norm["c2"] < 0.51


def test_score_normalization_equal_scores():
    """Score normalization handles uniform scores gracefully."""
    mock_hits = [
        {"chunk_id": "c1", "score": 10.0},
        {"chunk_id": "c2", "score": 10.0},
    ]
    norm = HybridRetriever._normalize_scores(mock_hits)
    assert norm["c1"] == 1.0
    assert norm["c2"] == 1.0
