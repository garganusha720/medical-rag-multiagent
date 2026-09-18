"""
MedQuAD Dataset Parser
======================
Parses all XML files from the MedQuAD repository (https://github.com/abachaa/MedQuAD)
into a clean JSONL file for the RAG pipeline.

MedQuAD contains ~47,000 Q&A pairs across 12 NIH sources:
  1_CancerGov_QA, 2_GARD_QA, 3_GHR_QA, 4_MPlus_Health_Topics_QA,
  5_NIDDK_QA, 6_NINDS_QA, 7_SeniorHealth_QA, 8_NHLBI_QA_XML,
  9_CDC_QA, 10_MPlus_ADAM_QA, 11_MPlusDrugs_QA, 12_MPlusHerbsSupplements_QA

XML Structure (consistent across all sources):
  <Document id="..." source="..." url="...">
    <Focus>Disease/Topic Name</Focus>
    <QAPairs>
      <QAPair pid="1">
        <Question qid="..." qtype="...">Question text</Question>
        <Answer>Answer text</Answer>
      </QAPair>
      ...
    </QAPairs>
  </Document>

Output: data/processed/medquad.jsonl
  Each line = one Q&A pair as JSON with fields:
    id, question, answer, source, source_url, focus, question_type, doc_id, word_count
"""

import json
import hashlib
import re
import unicodedata
import logging
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────
MEDQUAD_DIR = Path("MedQuAD")              # Relative to project root
OUTPUT_FILE = Path("data/processed/medquad.jsonl")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─── Text Normalization ─────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Clean and normalize text extracted from XML.
    - Strips leading/trailing whitespace
    - Normalizes Unicode to NFC form
    - Replaces XML entities (&apos; &quot; &amp;)
    - Collapses multiple whitespace/newlines into single spaces
    - Removes bullet markers (-, •) at start of lines
    """
    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Replace common XML entities that ElementTree may leave
    text = text.replace("&apos;", "'")
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")

    # Collapse whitespace: tabs, newlines, multiple spaces → single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ─── XML Parsing ────────────────────────────────────────────────
def parse_single_xml(filepath: Path) -> list[dict]:
    """
    Parse a single MedQuAD XML file and extract all Q&A pairs.

    Args:
        filepath: Path to the XML file

    Returns:
        List of dicts, one per Q&A pair, with fields:
            question, answer, source, source_url, focus, question_type,
            doc_id, qid, pid
    """
    records = []

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"XML parse error in {filepath}: {e}")
        return records
    except Exception as e:
        logger.warning(f"Error reading {filepath}: {e}")
        return records

    # Extract document-level metadata
    doc_id = root.get("id", "")
    source = root.get("source", "")
    source_url = root.get("url", "")

    # Extract focus (disease/topic name)
    focus_elem = root.find("Focus")
    focus = normalize_text(focus_elem.text) if focus_elem is not None and focus_elem.text else ""

    # Extract all Q&A pairs
    qa_pairs = root.find("QAPairs")
    if qa_pairs is None:
        logger.debug(f"No QAPairs found in {filepath}")
        return records

    for qa_pair in qa_pairs.findall("QAPair"):
        pid = qa_pair.get("pid", "")

        # Extract question
        question_elem = qa_pair.find("Question")
        if question_elem is None or not question_elem.text:
            continue
        question = normalize_text(question_elem.text)
        qid = question_elem.get("qid", "")
        qtype = question_elem.get("qtype", "")

        # Extract answer
        answer_elem = qa_pair.find("Answer")
        if answer_elem is None or not answer_elem.text:
            continue
        answer = normalize_text(answer_elem.text)

        # Skip if question or answer is too short (likely garbage)
        if len(question) < 10 or len(answer) < 20:
            continue

        records.append({
            "question": question,
            "answer": answer,
            "source": source,
            "source_url": source_url,
            "focus": focus,
            "question_type": qtype,
            "doc_id": doc_id,
            "qid": qid,
            "pid": pid,
        })

    return records


def parse_all_medquad(medquad_dir: Path) -> list[dict]:
    """
    Recursively parse all XML files in the MedQuAD directory.

    Args:
        medquad_dir: Path to the cloned MedQuAD repository

    Returns:
        List of all Q&A pair dicts
    """
    all_records = []
    xml_files = sorted(medquad_dir.rglob("*.xml"))

    logger.info(f"Found {len(xml_files)} XML files in {medquad_dir}")

    parse_errors = 0
    for filepath in xml_files:
        records = parse_single_xml(filepath)
        if not records:
            parse_errors += 1
        all_records.extend(records)

    logger.info(f"Parsed {len(all_records)} Q&A pairs ({parse_errors} files had no valid Q&A)")
    return all_records


# ─── Deduplication ──────────────────────────────────────────────
def deduplicate(records: list[dict]) -> list[dict]:
    """
    Remove duplicate Q&A pairs based on question text hash.
    Keeps the first occurrence of each unique question.
    """
    seen_hashes = set()
    unique_records = []

    for record in records:
        q_hash = compute_hash(record["question"].lower())
        if q_hash not in seen_hashes:
            seen_hashes.add(q_hash)
            unique_records.append(record)

    removed = len(records) - len(unique_records)
    logger.info(f"Deduplication: {removed} duplicates removed, {len(unique_records)} unique records remain")
    return unique_records


# ─── Assign Stable IDs ─────────────────────────────────────────
def assign_ids(records: list[dict]) -> list[dict]:
    """
    Assign stable global IDs in format: medquad_XXXXX
    Also compute word_count for each answer.
    """
    for idx, record in enumerate(records):
        record["id"] = f"medquad_{idx + 1:05d}"
        record["word_count"] = len(record["answer"].split())
    return records


# ─── Save to JSONL ──────────────────────────────────────────────
def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save records as JSONL (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(records)} records to {output_path} ({file_size_mb:.1f} MB)")


# ─── Statistics & Logging ───────────────────────────────────────
def log_statistics(records: list[dict]) -> None:
    """Log detailed corpus statistics."""
    if not records:
        logger.warning("No records to analyze!")
        return

    # Source breakdown
    source_counts = Counter(r["source"] for r in records)
    logger.info("=" * 60)
    logger.info("MedQuAD Ingestion Statistics")
    logger.info("=" * 60)
    logger.info(f"Total Q&A pairs: {len(records):,}")
    logger.info("")
    logger.info("Records per source:")
    for source, count in source_counts.most_common():
        logger.info(f"  {source:35s} {count:>6,}")

    # Question type breakdown
    qtype_counts = Counter(r["question_type"] for r in records)
    logger.info("")
    logger.info("Question types:")
    for qtype, count in qtype_counts.most_common():
        label = qtype if qtype else "(no type)"
        logger.info(f"  {label:35s} {count:>6,}")

    # Answer length stats
    word_counts = [r["word_count"] for r in records]
    avg_wc = sum(word_counts) / len(word_counts)
    min_wc = min(word_counts)
    max_wc = max(word_counts)
    median_wc = sorted(word_counts)[len(word_counts) // 2]

    logger.info("")
    logger.info("Answer length (words):")
    logger.info(f"  Average: {avg_wc:.0f}")
    logger.info(f"  Median:  {median_wc}")
    logger.info(f"  Min:     {min_wc}")
    logger.info(f"  Max:     {max_wc}")

    # Short answers (will be kept as single chunks later)
    short = sum(1 for wc in word_counts if wc < 100)
    logger.info(f"  Short answers (<100 words): {short:,} ({100*short/len(records):.1f}%)")
    logger.info("=" * 60)


# ─── Main ───────────────────────────────────────────────────────
def main():
    """Run the full MedQuAD ingestion pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    medquad_dir = project_root / MEDQUAD_DIR
    output_file = project_root / OUTPUT_FILE

    # Validate input
    if not medquad_dir.exists():
        logger.error(f"MedQuAD directory not found: {medquad_dir}")
        logger.error("Please clone: git clone https://github.com/abachaa/MedQuAD.git")
        return

    # Parse → Deduplicate → Assign IDs → Save
    logger.info(f"Starting MedQuAD ingestion from {medquad_dir}")
    records = parse_all_medquad(medquad_dir)
    records = deduplicate(records)
    records = assign_ids(records)
    save_jsonl(records, output_file)
    log_statistics(records)

    logger.info("MedQuAD ingestion complete!")


if __name__ == "__main__":
    main()
