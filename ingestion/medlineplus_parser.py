"""
MedlinePlus Dataset Parser
==========================
Parses the MedlinePlus health topics XML file into a clean JSONL file
for the RAG pipeline.

MedlinePlus contains ~2,000 health topic pages from the National Library
of Medicine (NLM). Each topic has a full-summary article (500-2000 words),
which provides rich retrieval text that complements MedQuAD's shorter Q&A pairs.

Source: https://medlineplus.gov/xml.html
XML downloaded to: data/raw/medlineplus.xml

XML Structure:
  <health-topics total="2033">
    <health-topic title="..." url="..." id="..." language="English" ...>
      <full-summary>HTML-encoded article text</full-summary>
      <also-called>Alias name</also-called>
      <group>Category Name</group>
      <mesh-heading><descriptor>MeSH Term</descriptor></mesh-heading>
      <primary-institute>Source Institute</primary-institute>
      ...
    </health-topic>
  </health-topics>

Output: data/processed/medlineplus.jsonl
  Each line = one health topic as JSON with fields:
    id, title, full_text, url, meta_description, also_called,
    categories, mesh_terms, primary_institute, date_created, word_count
"""

import json
import re
import html
import unicodedata
import logging
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────
INPUT_FILE = Path("data/raw/medlineplus.xml")
OUTPUT_FILE = Path("data/processed/medlineplus.jsonl")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─── HTML / Text Cleaning ───────────────────────────────────────
def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text while preserving readable content.
    - Converts <li> items to "- " bullet points
    - Converts <p> and <br> to newlines
    - Strips all remaining HTML tags
    - Decodes HTML entities (&amp; → &, etc.)
    """
    if not text:
        return ""

    # Decode HTML entities first (handles &lt;p&gt; → <p> etc.)
    text = html.unescape(text)

    # Convert list items to bullet points
    text = re.sub(r"<li[^>]*>", "- ", text)

    # Convert block elements to newlines
    text = re.sub(r"</?(?:p|br|div|h[1-6]|ul|ol|tr|table)[^>]*>", "\n", text)

    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    return text


def normalize_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    - Unicode NFC normalization
    - Collapse multiple blank lines to single
    - Collapse multiple spaces to single
    - Strip leading/trailing whitespace per line
    - Strip overall leading/trailing whitespace
    """
    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Strip each line and remove empty lines
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]  # Remove empty lines

    # Join with single space (makes it one clean paragraph/block)
    text = " ".join(lines)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ─── XML Parsing ────────────────────────────────────────────────
def parse_health_topic(topic_elem: ET.Element) -> dict | None:
    """
    Parse a single <health-topic> element.

    Returns:
        Dict with topic fields, or None if topic should be skipped
        (e.g., non-English, no summary text)
    """
    # Only process English topics
    language = topic_elem.get("language", "")
    if language != "English":
        return None

    # Extract full-summary (the main article text)
    summary_elem = topic_elem.find("full-summary")
    if summary_elem is None or not summary_elem.text:
        return None

    raw_summary = summary_elem.text
    clean_text = normalize_text(strip_html_tags(raw_summary))

    # Skip topics with very short summaries (likely stubs)
    if len(clean_text.split()) < 30:
        return None

    # Extract metadata
    title = topic_elem.get("title", "").strip()
    url = topic_elem.get("url", "")
    topic_id = topic_elem.get("id", "")
    meta_desc = topic_elem.get("meta-desc", "").strip()
    date_created = topic_elem.get("date-created", "")

    # Extract also-called (aliases)
    also_called = [
        ac.text.strip()
        for ac in topic_elem.findall("also-called")
        if ac.text and ac.text.strip()
    ]

    # Extract topic categories (groups)
    categories = [
        g.text.strip()
        for g in topic_elem.findall("group")
        if g.text and g.text.strip()
    ]

    # Extract MeSH terms
    mesh_terms = []
    for mh in topic_elem.findall("mesh-heading"):
        descriptor = mh.find("descriptor")
        if descriptor is not None and descriptor.text:
            mesh_terms.append(descriptor.text.strip())

    # Extract primary institute
    pi_elem = topic_elem.find("primary-institute")
    primary_institute = ""
    if pi_elem is not None and pi_elem.text:
        primary_institute = pi_elem.text.strip()

    return {
        "title": title,
        "full_text": clean_text,
        "url": url,
        "topic_id": topic_id,
        "meta_description": meta_desc,
        "also_called": also_called,
        "categories": categories,
        "mesh_terms": mesh_terms,
        "primary_institute": primary_institute,
        "date_created": date_created,
        "word_count": len(clean_text.split()),
    }


def parse_medlineplus(input_path: Path) -> list[dict]:
    """
    Parse the MedlinePlus XML file and extract all English health topics.

    Args:
        input_path: Path to medlineplus.xml

    Returns:
        List of health topic dicts
    """
    logger.info(f"Parsing MedlinePlus XML: {input_path}")

    tree = ET.parse(input_path)
    root = tree.getroot()

    total_in_xml = root.get("total", "?")
    logger.info(f"XML declares {total_in_xml} total topics (includes all languages)")

    topics = root.findall("health-topic")
    logger.info(f"Found {len(topics)} <health-topic> elements")

    records = []
    skipped_language = 0
    skipped_no_summary = 0
    skipped_too_short = 0

    for topic in topics:
        language = topic.get("language", "")
        if language != "English":
            skipped_language += 1
            continue

        summary_elem = topic.find("full-summary")
        if summary_elem is None or not summary_elem.text:
            skipped_no_summary += 1
            continue

        result = parse_health_topic(topic)
        if result is None:
            skipped_too_short += 1
            continue

        records.append(result)

    logger.info(f"Parsed {len(records)} English topics with valid summaries")
    logger.info(f"Skipped: {skipped_language} non-English, "
                f"{skipped_no_summary} no summary, "
                f"{skipped_too_short} too short")

    return records


# ─── Assign Stable IDs ─────────────────────────────────────────
def assign_ids(records: list[dict]) -> list[dict]:
    """Assign stable global IDs in format: medlineplus_XXXXX"""
    for idx, record in enumerate(records):
        record["id"] = f"medlineplus_{idx + 1:05d}"
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


# ─── Statistics ─────────────────────────────────────────────────
def log_statistics(records: list[dict]) -> None:
    """Log detailed corpus statistics."""
    if not records:
        logger.warning("No records to analyze!")
        return

    word_counts = [r["word_count"] for r in records]
    avg_wc = sum(word_counts) / len(word_counts)
    median_wc = sorted(word_counts)[len(word_counts) // 2]

    # Category breakdown
    cat_counter = Counter()
    for r in records:
        for cat in r["categories"]:
            cat_counter[cat] += 1

    # Institute breakdown
    inst_counter = Counter(r["primary_institute"] for r in records if r["primary_institute"])

    logger.info("=" * 60)
    logger.info("MedlinePlus Ingestion Statistics")
    logger.info("=" * 60)
    logger.info(f"Total health topics: {len(records):,}")
    logger.info("")
    logger.info("Article length (words):")
    logger.info(f"  Average: {avg_wc:.0f}")
    logger.info(f"  Median:  {median_wc}")
    logger.info(f"  Min:     {min(word_counts)}")
    logger.info(f"  Max:     {max(word_counts)}")
    long = sum(1 for wc in word_counts if wc >= 200)
    logger.info(f"  Long articles (>=200 words): {long:,} ({100*long/len(records):.1f}%)")

    logger.info("")
    logger.info(f"Top 15 categories (of {len(cat_counter)} total):")
    for cat, count in cat_counter.most_common(15):
        logger.info(f"  {cat:40s} {count:>5,}")

    logger.info("")
    logger.info(f"Top 10 source institutes:")
    for inst, count in inst_counter.most_common(10):
        logger.info(f"  {inst:50s} {count:>5,}")

    logger.info("=" * 60)


# ─── Main ───────────────────────────────────────────────────────
def main():
    """Run the full MedlinePlus ingestion pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / INPUT_FILE
    output_file = project_root / OUTPUT_FILE

    if not input_file.exists():
        logger.error(f"MedlinePlus XML not found: {input_file}")
        logger.error("Download from: https://medlineplus.gov/xml.html")
        return

    # Parse → Assign IDs → Save
    records = parse_medlineplus(input_file)
    records = assign_ids(records)
    save_jsonl(records, output_file)
    log_statistics(records)

    logger.info("MedlinePlus ingestion complete!")


if __name__ == "__main__":
    main()
