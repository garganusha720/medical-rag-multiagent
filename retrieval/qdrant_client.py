"""
Qdrant Vector Database Wrapper
==============================
Manages vector storage and similarity search for medical chunks.

Features:
  - Dual Mode:
      * 'local': Connects to local Qdrant server (http://localhost:6333) or local on-disk storage
      * 'cloud': Connects to Qdrant Cloud cluster via QDRANT_URL and QDRANT_API_KEY
  - Multi-Collection support (e.g. 'medical_minilm', 'medical_cohere')
  - Automatic collection creation with Cosine distance metric
  - Fast batch upserting with rich metadata payload
  - Keep-alive ping mechanism to prevent Qdrant Cloud free tier inactivity suspension
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

load_dotenv()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Config
QDRANT_MODE = os.getenv("QDRANT_MODE", "local").lower()
QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

COLLECTION_MINILM = "medical_minilm"
COLLECTION_COHERE = "medical_cohere"
PROCESSED_CHUNKS_FILE = Path("data/processed/chunks.jsonl")


_SHARED_CLIENT: QdrantClient | None = None


class MedicalVectorStore:
    """Wrapper for Qdrant vector database operations."""

    def __init__(self, mode: str | None = None, url: str | None = None, api_key: str | None = None):
        global _SHARED_CLIENT
        self.mode = (mode or QDRANT_MODE).lower()
        self.url = url or (QDRANT_URL if self.mode == "cloud" else QDRANT_LOCAL_URL)
        self.api_key = api_key or (QDRANT_API_KEY if self.mode == "cloud" else None)

        if _SHARED_CLIENT is not None:
            self.client = _SHARED_CLIENT
            return

        logger.info(f"Initializing Qdrant client in mode '{self.mode}' (URL: {self.url or 'local/memory'})")
        
        try:
            if self.mode == "cloud":
                if not self.url or not self.api_key:
                    logger.warning("QDRANT_URL or QDRANT_API_KEY not configured. Falling back to local in-memory/disk store.")
                    self.client = QdrantClient(path="data/qdrant_storage")
                else:
                    self.client = QdrantClient(url=self.url, api_key=self.api_key)
            elif self.mode == "local":
                # Try connecting to local Docker instance, fallback to local embedded storage
                try:
                    self.client = QdrantClient(url=self.url, timeout=3.0)
                    self.client.get_collections()
                except Exception:
                    logger.info("Local Qdrant server not responding at 6333. Using embedded on-disk storage at 'data/qdrant_storage'")
                    self.client = QdrantClient(path="data/qdrant_storage")
            else:
                self.client = QdrantClient(path="data/qdrant_storage")
        except Exception as e:
            logger.warning(f"Error initializing Qdrant client: {e}. Defaulting to in-memory store.")
            self.client = QdrantClient(":memory:")

        _SHARED_CLIENT = self.client

    def ping(self) -> bool:
        """Ping Qdrant to keep connection active and prevent cloud tier suspension."""
        try:
            collections = self.client.get_collections()
            logger.info(f"Qdrant keep-alive ping successful ({len(collections.collections)} collections found)")
            return True
        except Exception as e:
            logger.error(f"Qdrant ping failed: {e}")
            return False

    def create_collection_if_not_exists(self, collection_name: str, vector_dim: int) -> None:
        """Create a collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            logger.info(f"Creating Qdrant collection '{collection_name}' (dim={vector_dim}, metric=Cosine)")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # Create payload index on 'source' for filtered search
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="source",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    def upload_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: np.ndarray,
        batch_size: int = 500,
    ) -> None:
        """
        Upload chunk texts, embeddings, and metadata payloads to Qdrant.

        Args:
            collection_name: Target collection
            chunks: List of chunk dicts from chunks.jsonl
            embeddings: Numpy array of shape (len(chunks), vector_dim)
            batch_size: Number of points to upsert per request
        """
        assert len(chunks) == len(embeddings), "Mismatch between chunk count and embedding vectors"
        dim = embeddings.shape[1]
        self.create_collection_if_not_exists(collection_name, dim)

        total = len(chunks)
        logger.info(f"Uploading {total:,} chunks to collection '{collection_name}' in batches of {batch_size}...")

        for i in tqdm(range(0, total, batch_size), desc="Uploading to Qdrant"):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            points = []
            for offset, chunk in enumerate(batch_chunks):
                idx = i + offset
                vector = batch_embeddings[offset].tolist()
                payload = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "source": chunk["metadata"].get("source", ""),
                    "title": chunk["metadata"].get("title", ""),
                    "url": chunk["metadata"].get("url", ""),
                    "token_count": chunk.get("token_count", 0),
                    "word_count": chunk.get("word_count", 0),
                    "metadata": chunk.get("metadata", {}),
                }
                points.append(
                    qmodels.PointStruct(
                        id=idx,
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=False if i + batch_size < total else True,
            )

        logger.info(f"Successfully uploaded {total:,} points to '{collection_name}'")

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Perform vector similarity search.

        Args:
            collection_name: Target collection
            query_vector: Dense vector for search query
            top_k: Number of results to return
            score_threshold: Minimum cosine similarity score
            source_filter: Optional source filter ('pubmed', 'medquad', 'medlineplus')

        Returns:
            List of matching records with 'chunk_id', 'text', 'score', and 'metadata'
        """
        query_filter = None
        if source_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source",
                        match=qmodels.MatchValue(value=source_filter.lower()),
                    )
                ]
            )

        # Execute search
        try:
            query_response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
            points = query_response.points
        except Exception as e:
            logger.debug(f"Qdrant query error (collection may not be indexed yet): {e}")
            return []

        formatted_results = []
        for hit in points:
            payload = hit.payload or {}
            formatted_results.append({
                "chunk_id": payload.get("chunk_id", ""),
                "doc_id": payload.get("doc_id", ""),
                "text": payload.get("text", ""),
                "score": float(hit.score),
                "source": payload.get("source", ""),
                "title": payload.get("title", ""),
                "url": payload.get("url", ""),
                "metadata": payload.get("metadata", {}),
            })

        return formatted_results


# ─── Script Entrypoint for Populating Qdrant ────────────────────
def index_embeddings_to_qdrant(
    model_type: Literal["minilm", "cohere"] = "minilm",
    chunks_file: Path = PROCESSED_CHUNKS_FILE,
    embeddings_file: Path | None = None,
    limit: int | None = None,
) -> None:
    """Load chunks and .npy embeddings, then index into Qdrant."""
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / chunks_file
    
    if embeddings_file is None:
        embeddings_path = project_root / f"data/embeddings/{model_type}_embeddings.npy"
    else:
        embeddings_path = Path(embeddings_file)

    if not embeddings_path.exists():
        logger.error(f"Embeddings file not found: {embeddings_path}. Run embedder.py first.")
        return

    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    logger.info(f"Loading chunks from {chunks_path}")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if limit and len(chunks) >= limit:
                    break

    if len(embeddings) < len(chunks):
        chunks = chunks[: len(embeddings)]
    elif len(chunks) < len(embeddings):
        embeddings = embeddings[: len(chunks)]

    collection_name = COLLECTION_COHERE if model_type == "cohere" else COLLECTION_MINILM
    store = MedicalVectorStore()
    store.upload_chunks(collection_name=collection_name, chunks=chunks, embeddings=embeddings)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant")
    parser.add_argument("--model", type=str, default="minilm", choices=["minilm", "cohere"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    index_embeddings_to_qdrant(model_type=args.model, limit=args.limit)
