"""
Token-Based Document Chunker
=============================
Splits the merged corpus into overlapping token-based chunks for
embedding and retrieval. Uses tiktoken's cl100k_base tokenizer
(same tokenizer used by OpenAI models) for accurate token counting.

Chunking strategy:
  - Short documents (<100 tokens): kept as a single chunk, not split
    → Most MedQuAD Q&A pairs fall here — splitting would lose context
  - Long documents (≥100 tokens): split with overlap
    → MedlinePlus articles and PubMed abstracts benefit from chunking

Configurable via .env:
  CHUNK_SIZE=256     (tokens per chunk)
  CHUNK_OVERLAP=32   (overlap tokens between consecutive chunks)

Input:  data/processed/corpus.jsonl
Output: data/processed/chunks.jsonl

Each chunk has:
  chunk_id:  {source}_{doc_num}_{chunk_index}  e.g. medlineplus_00042_003
  doc_id:    original document ID
  text:      chunk text
  metadata:  source, url, title, word_count, + source-specific fields
"""

import json
import logging
import os
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────
INPUT_FILE = Path("data/processed/corpus.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load config from .env
load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "32"))
SHORT_DOC_THRESHOLD = 100  # Docs under this token count = single chunk

# Initialize tokenizer
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ─── Core Chunking Logic ────────────────────────────────────────
def tokenize(text: str) -> list[int]:
    """Encode text to token IDs."""
    return TOKENIZER.encode(text)


def detokenize(token_ids: list[int]) -> str:
    """Decode token IDs back to text."""
    return TOKENIZER.decode(token_ids)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping token-based chunks.

    Args:
        text: The input text to chunk
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks

    Returns:
        List of chunk text strings
    """
    tokens = tokenize(text)
    num_tokens = len(tokens)

    # Short document → keep as single chunk
    if num_tokens <= SHORT_DOC_THRESHOLD:
        return [text.strip()]

    # If document fits in one chunk, don't split
    if num_tokens <= chunk_size:
        return [text.strip()]

    # Split into overlapping chunks
    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, num_tokens, step):
        end = min(start + chunk_size, num_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = detokenize(chunk_tokens).strip()

        if chunk_text:  # Skip empty chunks
            chunks.append(chunk_text)

        # Stop if we've reached the end
        if end >= num_tokens:
            break

    return chunks


# ─── Chunk Document ─────────────────────────────────────────────
def chunk_document(doc: dict) -> list[dict]:
    """
    Chunk a single corpus document into one or more chunks.

    Args:
        doc: Document dict from corpus.jsonl

    Returns:
        List of chunk dicts with chunk_id, doc_id, text, metadata
    """
    text = doc.get("text", "")
    if not text or not text.strip():
        return []

    doc_id = doc["id"]
    source = doc["source"]

    # Chunk the text
    text_chunks = chunk_text(text)

    # Build chunk records
    chunk_records = []
    for idx, chunk in enumerate(text_chunks):
        chunk_id = f"{doc_id}_{idx:03d}"

        chunk_record = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_index": idx,
            "total_chunks": len(text_chunks),
            "text": chunk,
            "token_count": len(tokenize(chunk)),
            "word_count": len(chunk.split()),
            "metadata": {
                "source": source,
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                **doc.get("metadata", {}),
            },
        }
        chunk_records.append(chunk_record)

    return chunk_records


# ─── Pipeline ───────────────────────────────────────────────────
def chunk_corpus(input_path: Path) -> list[dict]:
    """
    Load corpus and chunk all documents.

    Args:
        input_path: Path to corpus.jsonl

    Returns:
        List of all chunk dicts
    """
    # Load corpus
    logger.info(f"Loading corpus from {input_path}")
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    logger.info(f"Loaded {len(docs):,} documents")

    # Chunk all documents
    logger.info(f"Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
                f"short_threshold={SHORT_DOC_THRESHOLD}")

    all_chunks = []
    single_chunk_docs = 0
    multi_chunk_docs = 0

    for doc in tqdm(docs, desc="Chunking documents"):
        chunks = chunk_document(doc)
        if len(chunks) == 1:
            single_chunk_docs += 1
        elif len(chunks) > 1:
            multi_chunk_docs += 1
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks):,}")
    logger.info(f"Single-chunk docs: {single_chunk_docs:,} "
                f"(kept whole, <{SHORT_DOC_THRESHOLD} tokens or <{CHUNK_SIZE} tokens)")
    logger.info(f"Multi-chunk docs:  {multi_chunk_docs:,} "
                f"(split into 2+ chunks)")

    return all_chunks


# ─── Save ───────────────────────────────────────────────────────
def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(records):,} chunks to {output_path} ({file_size_mb:.1f} MB)")


# ─── Statistics ─────────────────────────────────────────────────
def log_statistics(chunks: list[dict]) -> None:
    """Log detailed chunk statistics."""
    if not chunks:
        logger.warning("No chunks!")
        return

    token_counts = [c["token_count"] for c in chunks]
    word_counts = [c["word_count"] for c in chunks]

    # Source breakdown
    from collections import Counter
    source_counts = Counter(c["metadata"]["source"] for c in chunks)

    # Chunks per doc distribution
    doc_chunk_counts = Counter()
    for c in chunks:
        doc_chunk_counts[c["doc_id"]] = max(doc_chunk_counts[c["doc_id"]], c["total_chunks"])
    chunk_dist = Counter(doc_chunk_counts.values())

    logger.info("=" * 60)
    logger.info("Chunking Statistics")
    logger.info("=" * 60)
    logger.info(f"Total chunks: {len(chunks):,}")
    logger.info(f"From {len(doc_chunk_counts):,} documents")
    logger.info("")
    logger.info("Chunks per source:")
    for source, count in source_counts.most_common():
        logger.info(f"  {source:15s} {count:>7,} chunks")
    logger.info("")
    logger.info("Token count per chunk:")
    logger.info(f"  Average: {sum(token_counts)/len(token_counts):.0f}")
    logger.info(f"  Median:  {sorted(token_counts)[len(token_counts)//2]}")
    logger.info(f"  Min:     {min(token_counts)}")
    logger.info(f"  Max:     {max(token_counts)}")
    logger.info("")
    logger.info("Word count per chunk:")
    logger.info(f"  Average: {sum(word_counts)/len(word_counts):.0f}")
    logger.info(f"  Median:  {sorted(word_counts)[len(word_counts)//2]}")
    logger.info("")
    logger.info("Chunks-per-document distribution:")
    for num_chunks, num_docs in sorted(chunk_dist.items()):
        logger.info(f"  {num_chunks} chunk(s): {num_docs:>6,} docs")
    logger.info("")

    # Verify overlap for multi-chunk docs
    overlap_ok = 0
    overlap_fail = 0
    for doc_id in list(doc_chunk_counts.keys())[:100]:  # Check first 100
        doc_chunks = sorted([c for c in chunks if c["doc_id"] == doc_id],
                            key=lambda x: x["chunk_index"])
        if len(doc_chunks) >= 2:
            # Check that consecutive chunks share some text (overlap)
            for i in range(len(doc_chunks) - 1):
                tokens_curr = set(tokenize(doc_chunks[i]["text"])[-CHUNK_OVERLAP:])
                tokens_next = set(tokenize(doc_chunks[i+1]["text"])[:CHUNK_OVERLAP])
                if tokens_curr & tokens_next:
                    overlap_ok += 1
                else:
                    overlap_fail += 1

    if overlap_ok + overlap_fail > 0:
        logger.info(f"Overlap verification (sampled 100 docs):")
        logger.info(f"  Overlaps found: {overlap_ok}, missing: {overlap_fail}")

    logger.info("=" * 60)


# ─── Main ───────────────────────────────────────────────────────
def main():
    """Run the full chunking pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / INPUT_FILE
    output_file = project_root / OUTPUT_FILE

    if not input_file.exists():
        logger.error(f"Corpus file not found: {input_file}")
        logger.error("Run merge_corpus.py first")
        return

    logger.info("Starting document chunking")
    logger.info(f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    logger.info("-" * 50)

    chunks = chunk_corpus(input_file)
    save_jsonl(chunks, output_file)
    log_statistics(chunks)

    logger.info("Chunking complete!")


if __name__ == "__main__":
    main()
