"""
Corpus Merger & Deduplicator
============================
Merges all 3 processed datasets (MedQuAD, MedlinePlus, PubMed) into
a single unified corpus file for the chunking pipeline.

Input files:
  data/processed/medquad.jsonl      → Q&A pairs (id: medquad_XXXXX)
  data/processed/medlineplus.jsonl  → Health topic articles (id: medlineplus_XXXXX)
  data/processed/pubmed.jsonl       → Research abstracts (id: pubmed_XXXXX)

Output:
  data/processed/corpus.jsonl       → Merged corpus with unified schema

Unified schema per record:
  id:          stable global ID (preserved from source)
  source:      "medquad" | "medlineplus" | "pubmed"
  text:        the main text content (answer / full_text / abstract)
  title:       topic/question title
  url:         source URL
  metadata:    source-specific metadata (topic, year, mesh_terms, etc.)
  word_count:  word count of the text field
"""

import json
import hashlib
import logging
from pathlib import Path
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "corpus.jsonl"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─── Loaders ────────────────────────────────────────────────────
def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records):,} records from {filepath.name}")
    return records


# ─── Transformers (normalize each source to unified schema) ─────
def transform_medquad(records: list[dict]) -> list[dict]:
    """
    Transform MedQuAD records to unified corpus schema.
    Uses the answer as text, question as title.
    """
    corpus = []
    for r in records:
        corpus.append({
            "id": r["id"],
            "source": "medquad",
            "text": r["answer"],
            "title": r["question"],
            "url": r.get("source_url", ""),
            "metadata": {
                "focus": r.get("focus", ""),
                "question_type": r.get("question_type", ""),
                "original_source": r.get("source", ""),
                "doc_id": r.get("doc_id", ""),
            },
            "word_count": r.get("word_count", len(r["answer"].split())),
        })
    return corpus


def transform_medlineplus(records: list[dict]) -> list[dict]:
    """
    Transform MedlinePlus records to unified corpus schema.
    Uses full_text as text, title as title.
    """
    corpus = []
    for r in records:
        corpus.append({
            "id": r["id"],
            "source": "medlineplus",
            "text": r["full_text"],
            "title": r["title"],
            "url": r.get("url", ""),
            "metadata": {
                "categories": r.get("categories", []),
                "also_called": r.get("also_called", []),
                "mesh_terms": r.get("mesh_terms", []),
                "primary_institute": r.get("primary_institute", ""),
                "meta_description": r.get("meta_description", ""),
            },
            "word_count": r.get("word_count", len(r["full_text"].split())),
        })
    return corpus


def transform_pubmed(records: list[dict]) -> list[dict]:
    """
    Transform PubMed records to unified corpus schema.
    Uses abstract as text, title as title.
    """
    corpus = []
    for r in records:
        corpus.append({
            "id": r["id"],
            "source": "pubmed",
            "text": r["abstract"],
            "title": r["title"],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/" if r.get("pmid") else "",
            "metadata": {
                "pmid": r.get("pmid", ""),
                "authors": r.get("authors", []),
                "year": r.get("year", ""),
                "journal": r.get("journal", ""),
                "mesh_terms": r.get("mesh_terms", []),
                "publication_type": r.get("publication_type", []),
                "doi": r.get("doi", ""),
            },
            "word_count": r.get("word_count", len(r["abstract"].split())),
        })
    return corpus


# ─── Deduplication ──────────────────────────────────────────────
def content_hash(text: str) -> str:
    """Compute hash of normalized text for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def deduplicate_corpus(records: list[dict]) -> list[dict]:
    """
    Remove near-duplicate records across sources based on text content hash.
    Keeps the first occurrence (priority: medquad > medlineplus > pubmed).
    """
    seen_hashes = set()
    unique = []

    for record in records:
        h = content_hash(record["text"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(record)

    removed = len(records) - len(unique)
    logger.info(f"Cross-source deduplication: {removed} duplicates removed, "
                f"{len(unique):,} unique records remain")
    return unique


# ─── Save ───────────────────────────────────────────────────────
def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(records):,} records to {output_path} ({file_size_mb:.1f} MB)")


# ─── Statistics ─────────────────────────────────────────────────
def log_statistics(records: list[dict]) -> None:
    """Log final corpus statistics."""
    if not records:
        logger.warning("No records!")
        return

    source_counts = Counter(r["source"] for r in records)
    word_counts = [r["word_count"] for r in records]
    total_words = sum(word_counts)

    logger.info("=" * 60)
    logger.info("Merged Corpus Statistics")
    logger.info("=" * 60)
    logger.info(f"Total documents: {len(records):,}")
    logger.info(f"Total words:     {total_words:,}")
    logger.info("")
    logger.info("Documents per source:")
    for source, count in source_counts.most_common():
        avg_wc = sum(r["word_count"] for r in records if r["source"] == source) / count
        logger.info(f"  {source:15s} {count:>7,} docs  (avg {avg_wc:.0f} words)")
    logger.info("")
    logger.info("Overall word count:")
    logger.info(f"  Average: {sum(word_counts)/len(word_counts):.0f}")
    logger.info(f"  Median:  {sorted(word_counts)[len(word_counts)//2]}")
    logger.info(f"  Min:     {min(word_counts)}")
    logger.info(f"  Max:     {max(word_counts)}")
    logger.info("=" * 60)


# ─── Main ───────────────────────────────────────────────────────
def main():
    """Run the full merge pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    processed = project_root / PROCESSED_DIR
    output = project_root / OUTPUT_FILE

    logger.info("Starting corpus merge")
    logger.info("-" * 50)

    # Load all sources
    medquad = load_jsonl(processed / "medquad.jsonl")
    medlineplus = load_jsonl(processed / "medlineplus.jsonl")
    pubmed = load_jsonl(processed / "pubmed.jsonl")

    # Transform to unified schema
    corpus = []
    corpus.extend(transform_medquad(medquad))
    corpus.extend(transform_medlineplus(medlineplus))
    corpus.extend(transform_pubmed(pubmed))
    logger.info(f"Total before dedup: {len(corpus):,}")

    # Deduplicate
    corpus = deduplicate_corpus(corpus)

    # Save
    save_jsonl(corpus, output)
    log_statistics(corpus)

    logger.info("Corpus merge complete!")


if __name__ == "__main__":
    main()
