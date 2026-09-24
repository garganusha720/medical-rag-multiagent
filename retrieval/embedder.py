"""
Embedding Generation Module
============================
Generates dense vector representations for all chunks in data/processed/chunks.jsonl
and for runtime user search queries.

Supported Embedding Models:
  1. Cohere (Primary, API):
     - Model: embed-english-v3.0
     - Dimensions: 1024
     - Requires: COHERE_API_KEY in .env
     - Input types: "search_document" for chunk indexing, "search_query" for runtime queries

  2. MiniLM (Comparison, Local):
     - Model: sentence-transformers/all-MiniLM-L6-v2
     - Dimensions: 384
     - Runs 100% locally on CPU/GPU, zero API cost, no rate limits

Outputs:
  - data/embeddings/cohere_embeddings.npy
  - data/embeddings/minilm_embeddings.npy
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import cohere
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────
PROCESSED_CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMBEDDINGS_DIR = Path("data/embeddings")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "minilm").lower()

COHERE_MODEL_NAME = "embed-english-v3.0"
COHERE_DIMENSION = 1024
COHERE_BATCH_SIZE = 96  # Cohere free trial limit is 96 texts per request

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIMENSION = 384
MINILM_BATCH_SIZE = 128


# ─── Embedder Base & Implementations ────────────────────────────
class BaseEmbedder:
    """Base interface for embedding models."""

    def __init__(self, dimension: int, model_name: str):
        self.dimension = dimension
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class CohereEmbedder(BaseEmbedder):
    """Cohere API Embedder (1024 dimensions)."""

    def __init__(self, api_key: str | None = None):
        super().__init__(dimension=COHERE_DIMENSION, model_name=COHERE_MODEL_NAME)
        self.api_key = api_key or COHERE_API_KEY
        if not self.api_key:
            raise ValueError(
                "COHERE_API_KEY is not set in environment or .env file. "
                "Get a free trial key from https://dashboard.cohere.com/api-keys"
            )
        self.client = cohere.ClientV2(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of document chunks using input_type='search_document'.
        Includes retry logic for rate limits.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.embed(
                    texts=texts,
                    model=self.model_name,
                    input_type="search_document",
                    embedding_types=["float"],
                )
                embeddings = response.embeddings.float_
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Cohere API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** (attempt + 1)
                    time.sleep(sleep_time)
                else:
                    raise

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query using input_type='search_query'."""
        response = self.client.embed(
            texts=[query],
            model=self.model_name,
            input_type="search_query",
            embedding_types=["float"],
        )
        return np.array(response.embeddings.float_[0], dtype=np.float32)


class LocalMiniLMEmbedder(BaseEmbedder):
    """Local SentenceTransformers Embedder (384 dimensions)."""

    def __init__(self):
        super().__init__(dimension=MINILM_DIMENSION, model_name=MINILM_MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        try:
            self.model = SentenceTransformer(MINILM_MODEL_NAME, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(MINILM_MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed document chunks locally."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=MINILM_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query locally."""
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].astype(np.float32)


def get_embedder(model_type: Literal["cohere", "minilm"] = "minilm") -> BaseEmbedder:
    """Factory function to instantiate the selected embedder."""
    model_type = model_type.lower()
    if model_type == "cohere":
        return CohereEmbedder()
    elif model_type in ["minilm", "local"]:
        return LocalMiniLMEmbedder()
    else:
        raise ValueError(f"Unknown embedding model: {model_type}. Choose 'cohere' or 'minilm'.")


# ─── Batch Embedding Pipeline ───────────────────────────────────
def generate_embeddings_for_corpus(
    chunks_path: Path = PROCESSED_CHUNKS_FILE,
    model_type: Literal["cohere", "minilm"] = "minilm",
    output_dir: Path = EMBEDDINGS_DIR,
    max_chunks: int | None = None,
) -> Path:
    """
    Generate and save dense vector embeddings for all chunks in chunks.jsonl.

    Args:
        chunks_path: Path to chunks.jsonl
        model_type: 'cohere' or 'minilm'
        output_dir: Target directory for .npy file
        max_chunks: Optional subset cap (for fast testing/eval)

    Returns:
        Path to saved .npy file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{model_type}_embeddings.npy"
    output_path = output_dir / out_filename

    # Load chunk texts
    logger.info(f"Loading chunks from {chunks_path}")
    texts = []
    chunk_ids = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item["text"])
                chunk_ids.append(item["chunk_id"])
                if max_chunks and len(texts) >= max_chunks:
                    break

    total_chunks = len(texts)
    logger.info(f"Loaded {total_chunks:,} chunks to embed using '{model_type}'")

    embedder = get_embedder(model_type)
    batch_size = COHERE_BATCH_SIZE if model_type == "cohere" else MINILM_BATCH_SIZE

    all_embeddings = []
    start_time = time.time()

    logger.info(f"Generating {embedder.dimension}-dim embeddings in batches of {batch_size}...")

    for i in tqdm(range(0, total_chunks, batch_size), desc=f"Embedding with {model_type}"):
        batch_texts = texts[i : i + batch_size]
        batch_vecs = embedder.embed_documents(batch_texts)
        all_embeddings.append(batch_vecs)

        # Rate-limiting pause for Cohere free trial (approx 10-20 calls/min)
        if model_type == "cohere":
            time.sleep(0.5)

    final_matrix = np.vstack(all_embeddings).astype(np.float32)
    elapsed = time.time() - start_time

    # Save to .npy
    np.save(output_path, final_matrix)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("=" * 60)
    logger.info(f"Embeddings Generated: {model_type.upper()}")
    logger.info(f"Shape:         {final_matrix.shape} (chunks x dimension)")
    logger.info(f"Dtype:         {final_matrix.dtype}")
    logger.info(f"Output File:   {output_path} ({file_size_mb:.1f} MB)")
    logger.info(f"Time Taken:    {elapsed:.1f}s ({total_chunks / max(elapsed, 0.001):.1f} chunks/sec)")
    logger.info("=" * 60)

    return output_path


# ─── CLI Entrypoint ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for medical corpus chunks")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=["cohere", "minilm"],
        help="Embedding model to use (default from .env or minilm)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks (optional, for quick testing)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    chunks_file = project_root / PROCESSED_CHUNKS_FILE
    out_dir = project_root / EMBEDDINGS_DIR

    generate_embeddings_for_corpus(
        chunks_path=chunks_file,
        model_type=args.model,
        output_dir=out_dir,
        max_chunks=args.limit,
    )
