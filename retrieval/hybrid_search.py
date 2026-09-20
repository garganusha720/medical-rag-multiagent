"""
Hybrid Retrieval Engine
=======================
Combines BM25 sparse keyword search and Qdrant dense vector search
with score normalization, weighted fusion, and MMR re-ranking.

Retrieval Methods:
  1. 'dense': Pure vector similarity search via Qdrant
  2. 'bm25': Pure keyword BM25Okapi search
  3. 'hybrid': Weighted combination:
       score = alpha * norm(dense_score) + (1 - alpha) * norm(bm25_score)
  4. 'mmr': Maximal Marginal Relevance re-ranking for result diversity:
       MMR = lambda * relevance - (1 - lambda) * max_similarity(chunk, selected)

Configurable parameters:
  - top_k: Number of retrieved chunks (default 5 or 8)
  - alpha: Weight for dense vector search (default 0.6)
  - score_threshold: Minimum final relevance score (default 0.0)
  - source_filter: Restrict retrieval to 'pubmed', 'medquad', or 'medlineplus'
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from retrieval.embedder import get_embedder, BaseEmbedder
from retrieval.qdrant_client import MedicalVectorStore, COLLECTION_MINILM, COLLECTION_COHERE

load_dotenv()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))


# ─── BM25 Sparse Search Engine ──────────────────────────────────
class BM25Index:
    """In-memory BM25 index over all medical chunk texts."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        
        logger.info(f"Building BM25 index on {len(chunks):,} chunk texts...")
        self.corpus_tokens = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info("BM25 index built successfully.")

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple clean word tokenizer for medical text."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 1]

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Search BM25 index for query."""
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Rank top indices
        top_indices = np.argsort(doc_scores)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
                continue
            
            chunk = self.chunks[idx]
            if source_filter and chunk["metadata"].get("source", "").lower() != source_filter.lower():
                continue

            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "score": score,
                "source": chunk["metadata"].get("source", ""),
                "title": chunk["metadata"].get("title", ""),
                "url": chunk["metadata"].get("url", ""),
                "metadata": chunk.get("metadata", {}),
            })
            if len(results) >= top_k:
                break

        return results


# ─── Dense Vector Retriever ─────────────────────────────────────
class DenseRetriever:
    """Dense vector retriever using Embedder + Qdrant."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        vector_store: MedicalVectorStore | None = None,
        model_type: str = "minilm",
    ):
        self.model_type = model_type
        self.embedder = embedder or get_embedder(model_type)
        self.vector_store = vector_store or MedicalVectorStore()
        self.collection_name = COLLECTION_COHERE if model_type == "cohere" else COLLECTION_MINILM

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Embed query and search vector store."""
        query_vec = self.embedder.embed_query(query)
        return self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            top_k=top_k,
            score_threshold=score_threshold,
            source_filter=source_filter,
        )


# ─── Hybrid Retriever ───────────────────────────────────────────
class HybridRetriever:
    """
    Unified hybrid retrieval orchestrator supporting:
    - Dense Search (Qdrant)
    - BM25 Sparse Search
    - Hybrid Weighted Fusion (Dense + BM25)
    - Maximal Marginal Relevance (MMR)
    """

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        model_type: str = "minilm",
        alpha: float = DEFAULT_ALPHA,
    ):
        self.chunks_path = chunks_path
        self.model_type = model_type
        self.alpha = alpha

        # Load chunks
        self.chunks = self._load_chunks(chunks_path)
        self.chunk_lookup = {c["chunk_id"]: c for c in self.chunks}

        # Initialize sub-retrievers
        self.bm25_index = BM25Index(self.chunks)
        self.dense_retriever = DenseRetriever(model_type=model_type)

    def _load_chunks(self, path: Path) -> list[dict]:
        """Load all chunks from JSONL."""
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found at {path}")
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    @staticmethod
    def _normalize_scores(results: list[dict]) -> dict[str, float]:
        """Min-max normalize scores to [0.0, 1.0]."""
        if not results:
            return {}
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if math.isclose(max_score, min_score):
            return {r["chunk_id"]: 1.0 for r in results}

        return {
            r["chunk_id"]: (r["score"] - min_score) / (max_score - min_score)
            for r in results
        }

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        method: Literal["dense", "bm25", "hybrid", "mmr"] = "hybrid",
        alpha: float | None = None,
        score_threshold: float = 0.0,
        source_filter: str | None = None,
        mmr_lambda: float = 0.5,
    ) -> list[dict]:
        """
        Execute search across chosen retrieval strategy.
        """
        effective_alpha = alpha if alpha is not None else self.alpha

        if method == "dense":
            results = self.dense_retriever.search(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                source_filter=source_filter,
            )
            return results

        elif method == "bm25":
            results = self.bm25_index.search(
                query=query,
                top_k=top_k,
                source_filter=source_filter,
            )
            return results

        elif method in ["hybrid", "mmr"]:
            # Retrieve candidate pool from both (fetch 3x top_k for high recall)
            candidate_k = max(top_k * 3, 20)
            
            dense_hits = self.dense_retriever.search(
                query=query,
                top_k=candidate_k,
                source_filter=source_filter,
            )
            bm25_hits = self.bm25_index.search(
                query=query,
                top_k=candidate_k,
                source_filter=source_filter,
            )

            dense_norm = self._normalize_scores(dense_hits)
            bm25_norm = self._normalize_scores(bm25_hits)

            # Combine scores
            all_chunk_ids = set(dense_norm.keys()) | set(bm25_norm.keys())
            combined_scores = {}
            candidate_records = {}

            # Map chunk records
            for hit in dense_hits + bm25_hits:
                candidate_records[hit["chunk_id"]] = hit

            for cid in all_chunk_ids:
                d_score = dense_norm.get(cid, 0.0)
                b_score = bm25_norm.get(cid, 0.0)
                final_score = (effective_alpha * d_score) + ((1.0 - effective_alpha) * b_score)
                combined_scores[cid] = final_score

            # Rank candidates
            ranked_cids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

            if method == "hybrid":
                final_results = []
                for cid in ranked_cids:
                    score = combined_scores[cid]
                    if score < score_threshold:
                        continue
                    rec = candidate_records[cid]
                    rec["score"] = float(score)
                    rec["dense_score"] = float(dense_norm.get(cid, 0.0))
                    rec["bm25_score"] = float(bm25_norm.get(cid, 0.0))
                    final_results.append(rec)
                    if len(final_results) >= top_k:
                        break
                return final_results

            elif method == "mmr":
                return self._mmr_rerank(
                    query=query,
                    ranked_cids=ranked_cids,
                    candidate_records=candidate_records,
                    combined_scores=combined_scores,
                    top_k=top_k,
                    mmr_lambda=mmr_lambda,
                )

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    def _mmr_rerank(
        self,
        query: str,
        ranked_cids: list[str],
        candidate_records: dict[str, dict],
        combined_scores: dict[str, float],
        top_k: int = 5,
        mmr_lambda: float = 0.5,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance (MMR) re-ranking.
        Balances query relevance with novelty among selected chunks.
        """
        if not ranked_cids:
            return []

        # Jaccard/word overlap based similarity for diversity penalty
        def chunk_sim(text1: str, text2: str) -> float:
            words1 = set(BM25Index.tokenize(text1))
            words2 = set(BM25Index.tokenize(text2))
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)

        selected_cids: list[str] = []
        remaining_cids = list(ranked_cids[: min(len(ranked_cids), top_k * 4)])

        while len(selected_cids) < top_k and remaining_cids:
            best_cid = None
            best_mmr_score = -float("inf")

            for cid in remaining_cids:
                rel_score = combined_scores[cid]
                chunk_text = candidate_records[cid]["text"]

                # Max similarity to already selected chunks
                if selected_cids:
                    max_sim = max(
                        chunk_sim(chunk_text, candidate_records[sel]["text"])
                        for sel in selected_cids
                    )
                else:
                    max_sim = 0.0

                mmr_score = (mmr_lambda * rel_score) - ((1.0 - mmr_lambda) * max_sim)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_cid = cid

            if best_cid is not None:
                selected_cids.append(best_cid)
                remaining_cids.remove(best_cid)
            else:
                break

        results = []
        for cid in selected_cids:
            rec = candidate_records[cid]
            rec["score"] = float(combined_scores[cid])
            results.append(rec)

        return results
