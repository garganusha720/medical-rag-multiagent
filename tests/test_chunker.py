"""
Unit Tests for Document Chunker
================================
Tests chunking logic, overlap correctness, short-document handling,
and stable chunk_id generation.
"""

import pytest
from retrieval.chunker import (
    chunk_text,
    chunk_document,
    tokenize,
    detokenize,
    SHORT_DOC_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def test_tokenize_detokenize_roundtrip():
    """Verify tokenizer encodes and decodes text faithfully."""
    sample = "Adult Acute Lymphoblastic Leukemia is a type of cancer affecting bone marrow."
    tokens = tokenize(sample)
    assert len(tokens) > 0
    decoded = detokenize(tokens)
    assert decoded == sample


def test_short_document_not_split():
    """Documents under SHORT_DOC_THRESHOLD (100 tokens) should remain a single chunk."""
    short_text = "What are the common symptoms of mild asthma? Coughing and shortness of breath."
    tokens = tokenize(short_text)
    assert len(tokens) < SHORT_DOC_THRESHOLD

    chunks = chunk_text(short_text, chunk_size=256, chunk_overlap=32)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_long_document_splits_with_overlap():
    """Long documents should split into multiple chunks with correct token overlap."""
    # Generate long synthetic text (~500 tokens)
    sentence = "Hypertension and cardiovascular disease require continuous clinical management and monitoring. "
    long_text = sentence * 30
    tokens = tokenize(long_text)
    assert len(tokens) > 300

    chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1

    # Verify each chunk is within target size (allowing ±2 tokens for BPE boundary re-encoding)
    for chunk in chunks:
        c_tokens = tokenize(chunk)
        assert len(c_tokens) <= 100 + 2

    # Verify overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        curr_tail_tokens = set(tokenize(chunks[i])[-20:])
        next_head_tokens = set(tokenize(chunks[i + 1])[:20])
        # Intersection should have tokens
        assert len(curr_tail_tokens & next_head_tokens) > 0


def test_chunk_document_id_format():
    """Verify chunk_id format {doc_id}_{idx:03d} and metadata preservation."""
    sample_doc = {
        "id": "medlineplus_00042",
        "source": "medlineplus",
        "text": "This is a detailed health article about diabetes care. " * 30,
        "title": "Diabetes Care Guidelines",
        "url": "https://medlineplus.gov/diabetes.html",
        "metadata": {
            "categories": ["Endocrine", "Diabetes"],
            "primary_institute": "NIDDK",
        },
    }

    chunks = chunk_document(sample_doc)
    assert len(chunks) >= 1

    for idx, c in enumerate(chunks):
        assert c["chunk_id"] == f"medlineplus_00042_{idx:03d}"
        assert c["doc_id"] == "medlineplus_00042"
        assert c["chunk_index"] == idx
        assert c["total_chunks"] == len(chunks)
        assert c["metadata"]["source"] == "medlineplus"
        assert c["metadata"]["title"] == "Diabetes Care Guidelines"
        assert c["metadata"]["categories"] == ["Endocrine", "Diabetes"]
        assert c["token_count"] > 0
        assert c["word_count"] > 0
