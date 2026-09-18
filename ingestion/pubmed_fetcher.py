"""
PubMed Abstract Fetcher
=======================
Fetches medical research paper abstracts from PubMed via the free
NCBI Entrez API. Focuses on recent medical topics (2020-2024+)
to supplement MedQuAD and MedlinePlus with current research.

Uses Bio.Entrez (biopython) — completely free, no credentials needed
for basic access. With an API key (NCBI_API_KEY), rate limit is
10 requests/sec instead of 3.

Search strategy:
  - Multiple focused queries to get diverse medical coverage
  - COVID/Long COVID, recent treatments, clinical trials, common diseases
  - Date range: 2020 onwards
  - Only abstracts in English with actual text

Output: data/processed/pubmed.jsonl
  Each line = one abstract as JSON with fields:
    pmid, title, abstract, authors, year, journal, mesh_terms,
    publication_type, doi, word_count
"""

import json
import time
import logging
from pathlib import Path
from collections import Counter

from Bio import Entrez
from dotenv import load_dotenv
import os
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────
OUTPUT_FILE = Path("data/processed/pubmed.jsonl")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "medicalrag@example.com")
FETCH_LIMIT = int(os.getenv("PUBMED_FETCH_LIMIT", "20000"))

# Configure Entrez
Entrez.email = NCBI_EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY
    RATE_LIMIT_DELAY = 0.11  # 10 req/sec with API key
    logger.info("NCBI API key found — using 10 req/sec rate limit")
else:
    RATE_LIMIT_DELAY = 0.35  # ~3 req/sec without API key
    logger.info("No NCBI API key — using 3 req/sec rate limit")

BATCH_SIZE = 500  # Max records per efetch call

# ─── Search Queries ─────────────────────────────────────────────
# Diverse queries to cover major medical topics with recent focus
SEARCH_QUERIES = [
    # COVID & Long COVID (high priority)
    ("COVID-19 treatment clinical trial", 3000),
    ("Long COVID symptoms management", 2000),
    ("SARS-CoV-2 vaccine efficacy", 2000),

    # Major disease areas
    ("diabetes type 2 treatment 2020:2026[dp]", 1500),
    ("hypertension management guidelines", 1500),
    ("cancer immunotherapy recent advances", 1500),
    ("heart failure treatment outcomes", 1200),
    ("chronic kidney disease management", 1000),
    ("mental health depression anxiety treatment", 1200),
    ("autoimmune disease therapy", 1000),

    # Common conditions
    ("asthma COPD treatment", 1000),
    ("stroke rehabilitation outcomes", 800),
    ("Alzheimer disease treatment clinical", 800),
    ("antibiotic resistance infection", 800),
    ("obesity metabolic syndrome intervention", 800),
    ("rare diseases genetic therapy", 600),
    ("pediatric diseases treatment children", 600),
]


# ─── PubMed API Functions ───────────────────────────────────────
def search_pubmed(query: str, max_results: int) -> list[str]:
    """
    Search PubMed and return list of PMIDs.

    Args:
        query: PubMed search query string
        max_results: Maximum number of results to return

    Returns:
        List of PMID strings
    """
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            datetype="pdat",
            mindate="2020",
            maxdate="2026",
        )
        results = Entrez.read(handle)
        handle.close()
        time.sleep(RATE_LIMIT_DELAY)

        pmids = results.get("IdList", [])
        total_available = int(results.get("Count", 0))
        logger.info(f"  Query: '{query[:50]}...' → {len(pmids)} PMIDs "
                    f"(of {total_available:,} available)")
        return pmids

    except Exception as e:
        logger.error(f"  Search failed for '{query[:50]}': {e}")
        return []


def fetch_abstracts_batch(pmids: list[str]) -> list[dict]:
    """
    Fetch full records for a batch of PMIDs.

    Args:
        pmids: List of PMID strings (max 500)

    Returns:
        List of parsed article dicts
    """
    records = []

    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="xml",
            retmode="xml",
        )
        result = Entrez.read(handle)
        handle.close()
        time.sleep(RATE_LIMIT_DELAY)

        articles = result.get("PubmedArticle", [])

        for article in articles:
            parsed = parse_article(article)
            if parsed:
                records.append(parsed)

    except Exception as e:
        logger.error(f"  Fetch failed for batch of {len(pmids)}: {e}")

    return records


def parse_article(article: dict) -> dict | None:
    """
    Parse a single PubMed article record into a clean dict.

    Returns None if the article has no abstract.
    """
    try:
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})

        # PMID
        pmid_obj = medline.get("PMID", "")
        pmid = str(pmid_obj)

        # Title
        title = str(article_data.get("ArticleTitle", "")).strip()
        if not title:
            return None

        # Abstract
        abstract_data = article_data.get("Abstract", {})
        abstract_texts = abstract_data.get("AbstractText", [])

        if not abstract_texts:
            return None

        # AbstractText can be a list of StringElement objects with labels
        abstract_parts = []
        for part in abstract_texts:
            part_str = str(part).strip()
            if hasattr(part, "attributes") and "Label" in part.attributes:
                label = part.attributes["Label"]
                abstract_parts.append(f"{label}: {part_str}")
            else:
                abstract_parts.append(part_str)

        abstract = " ".join(abstract_parts).strip()

        if len(abstract) < 50:  # Too short to be useful
            return None

        # Authors
        author_list = article_data.get("AuthorList", [])
        authors = []
        for author in author_list[:5]:  # First 5 authors
            last = str(author.get("LastName", ""))
            first = str(author.get("ForeName", ""))
            if last:
                authors.append(f"{last} {first}".strip())

        # Year
        pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = str(pub_date.get("Year", ""))
        if not year:
            medline_date = pub_date.get("MedlineDate", "")
            if medline_date:
                # Extract year from "2020 Jan-Mar" format
                year = str(medline_date)[:4]

        # Journal
        journal = str(article_data.get("Journal", {}).get("Title", ""))

        # MeSH terms
        mesh_list = medline.get("MeshHeadingList", [])
        mesh_terms = []
        for mesh in mesh_list:
            descriptor = mesh.get("DescriptorName", "")
            if descriptor:
                mesh_terms.append(str(descriptor))

        # Publication types
        pub_types = article_data.get("PublicationTypeList", [])
        pub_type_list = [str(pt) for pt in pub_types]

        # DOI
        doi = ""
        id_list = article_data.get("ELocationID", [])
        for eid in id_list:
            if hasattr(eid, "attributes") and eid.attributes.get("EIdType") == "doi":
                doi = str(eid)

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "journal": journal,
            "mesh_terms": mesh_terms,
            "publication_type": pub_type_list,
            "doi": doi,
            "word_count": len(abstract.split()),
        }

    except Exception as e:
        logger.debug(f"  Failed to parse article: {e}")
        return None


# ─── Main Pipeline ──────────────────────────────────────────────
def fetch_all_abstracts() -> list[dict]:
    """
    Execute all search queries and fetch abstracts, deduplicating by PMID.

    Returns:
        List of unique article dicts
    """
    all_pmids = set()
    query_pmids = {}

    # Phase 1: Collect PMIDs from all queries
    logger.info(f"Phase 1: Searching PubMed ({len(SEARCH_QUERIES)} queries)")
    logger.info(f"Target: {FETCH_LIMIT:,} total abstracts")
    logger.info("-" * 50)

    for query, max_results in SEARCH_QUERIES:
        pmids = search_pubmed(query, max_results)
        new_pmids = [p for p in pmids if p not in all_pmids]
        all_pmids.update(new_pmids)
        query_pmids[query[:40]] = len(new_pmids)

        # Stop if we have enough
        if len(all_pmids) >= FETCH_LIMIT:
            logger.info(f"  Reached target of {FETCH_LIMIT:,} PMIDs, stopping search")
            break

    # Trim to limit
    pmid_list = list(all_pmids)[:FETCH_LIMIT]
    logger.info(f"\nTotal unique PMIDs collected: {len(pmid_list):,}")

    # Phase 2: Fetch abstracts in batches
    logger.info(f"\nPhase 2: Fetching abstracts ({len(pmid_list):,} PMIDs in "
                f"batches of {BATCH_SIZE})")
    logger.info("-" * 50)

    all_records = []
    num_batches = (len(pmid_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(pmid_list), BATCH_SIZE),
                  total=num_batches, desc="Fetching batches"):
        batch = pmid_list[i:i + BATCH_SIZE]
        records = fetch_abstracts_batch(batch)
        all_records.extend(records)

        # Progress log every 10 batches
        if (i // BATCH_SIZE + 1) % 10 == 0:
            logger.info(f"  Progress: {len(all_records):,} abstracts fetched so far")

    logger.info(f"Total abstracts with valid text: {len(all_records):,}")
    return all_records


def deduplicate_by_pmid(records: list[dict]) -> list[dict]:
    """Remove any duplicate PMIDs (shouldn't happen but safety check)."""
    seen = set()
    unique = []
    for r in records:
        if r["pmid"] not in seen:
            seen.add(r["pmid"])
            unique.append(r)
    removed = len(records) - len(unique)
    if removed > 0:
        logger.info(f"Deduplication: removed {removed} duplicate PMIDs")
    return unique


def assign_ids(records: list[dict]) -> list[dict]:
    """Assign stable global IDs in format: pubmed_XXXXX"""
    for idx, record in enumerate(records):
        record["id"] = f"pubmed_{idx + 1:05d}"
    return records


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(records)} records to {output_path} ({file_size_mb:.1f} MB)")


def log_statistics(records: list[dict]) -> None:
    """Log detailed corpus statistics."""
    if not records:
        logger.warning("No records to analyze!")
        return

    word_counts = [r["word_count"] for r in records]
    avg_wc = sum(word_counts) / len(word_counts)
    median_wc = sorted(word_counts)[len(word_counts) // 2]

    year_counts = Counter(r["year"] for r in records if r["year"])
    mesh_counter = Counter()
    for r in records:
        for m in r["mesh_terms"]:
            mesh_counter[m] += 1

    logger.info("=" * 60)
    logger.info("PubMed Ingestion Statistics")
    logger.info("=" * 60)
    logger.info(f"Total abstracts: {len(records):,}")
    logger.info("")
    logger.info("Abstract length (words):")
    logger.info(f"  Average: {avg_wc:.0f}")
    logger.info(f"  Median:  {median_wc}")
    logger.info(f"  Min:     {min(word_counts)}")
    logger.info(f"  Max:     {max(word_counts)}")

    logger.info("")
    logger.info("By publication year:")
    for year, count in sorted(year_counts.items()):
        logger.info(f"  {year}: {count:>6,}")

    logger.info("")
    logger.info("Top 15 MeSH terms:")
    for term, count in mesh_counter.most_common(15):
        logger.info(f"  {term:45s} {count:>5,}")

    logger.info("=" * 60)


# ─── Main ───────────────────────────────────────────────────────
def main():
    """Run the full PubMed ingestion pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    output_file = project_root / OUTPUT_FILE

    logger.info("Starting PubMed abstract fetching")
    logger.info(f"Target: {FETCH_LIMIT:,} abstracts")
    logger.info(f"API key: {'configured' if NCBI_API_KEY else 'NOT SET (slower)'}")
    logger.info("")

    if not NCBI_API_KEY:
        logger.warning("No NCBI_API_KEY in .env — rate limited to 3 req/sec")
        logger.warning("Get one free at: https://www.ncbi.nlm.nih.gov/account/settings/")

    # Fetch → Deduplicate → Assign IDs → Save
    records = fetch_all_abstracts()
    records = deduplicate_by_pmid(records)
    records = assign_ids(records)
    save_jsonl(records, output_file)
    log_statistics(records)

    logger.info("PubMed ingestion complete!")


if __name__ == "__main__":
    main()
