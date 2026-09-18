"""
Unit Tests for Evaluation Metrics
=================================
Tests Recall@K, MRR@10, and ROUGE-L scoring against known ground truth.
"""

import pytest
from eval.evaluate import compute_retrieval_metrics, compute_rouge_score


def test_retrieval_metrics_perfect_first_rank():
    """Target doc at rank 1 gives Recall@1=1.0 and MRR@10=1.0."""
    target_id = "doc_100"
    retrieved = [
        {"doc_id": "doc_100", "score": 0.95},
        {"doc_id": "doc_101", "score": 0.80},
        {"doc_id": "doc_102", "score": 0.70},
    ]
    metrics = compute_retrieval_metrics(target_id, retrieved, k_list=[1, 5, 10])

    assert metrics["recall@1"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1.0


def test_retrieval_metrics_third_rank():
    """Target doc at rank 3 gives Recall@1=0.0, Recall@5=1.0, and MRR@10=1/3."""
    target_id = "doc_target"
    retrieved = [
        {"doc_id": "doc_wrong_1", "score": 0.95},
        {"doc_id": "doc_wrong_2", "score": 0.80},
        {"doc_id": "doc_target", "score": 0.75},
        {"doc_id": "doc_wrong_3", "score": 0.60},
    ]
    metrics = compute_retrieval_metrics(target_id, retrieved, k_list=[1, 5, 10])

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert abs(metrics["mrr@10"] - (1.0 / 3.0)) < 1e-5


def test_retrieval_metrics_not_found():
    """Target doc absent from results gives 0.0 for all recall and MRR metrics."""
    target_id = "doc_missing"
    retrieved = [
        {"doc_id": "doc_1", "score": 0.90},
        {"doc_id": "doc_2", "score": 0.80},
    ]
    metrics = compute_retrieval_metrics(target_id, retrieved, k_list=[1, 5, 10])

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 0.0
    assert metrics["recall@10"] == 0.0
    assert metrics["mrr@10"] == 0.0


def test_rouge_score_identical_texts():
    """Identical reference and generated strings should yield ROUGE-L F1 = 1.0."""
    text = "Metformin is a first-line medication for the treatment of type 2 diabetes."
    score = compute_rouge_score(text, text)
    assert abs(score - 1.0) < 1e-3


def test_rouge_score_partial_overlap():
    """Partially overlapping sentences should give score between 0.0 and 1.0."""
    ref = "Type 2 diabetes is treated with lifestyle changes and metformin."
    gen = "Management of type 2 diabetes involves metformin and lifestyle modifications."
    score = compute_rouge_score(ref, gen)
    assert 0.4 < score < 0.9
