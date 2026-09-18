# 🏥 Medical RAG Multi-Agent: Production-Grade Medical AI Assistant

A production-grade, citation-backed, hallucination-free Medical AI Assistant specifically designed for clinical and consumer health queries. Powered by a verified corpus of **NIH (MedQuAD, MedlinePlus)** and **PubMed** literature, orchestrated via **LangGraph multi-agent architecture**, and served via **FastAPI** with real-time streaming to **Next.js**.

---

## 🎯 The Core Problem & Solution

| Standard General LLMs (ChatGPT) | Medical RAG Multi-Agent System |
| :--- | :--- |
| ❌ Hallucinates medical facts & fake drug dosages | ✅ **Grounded strictly** in verified NIH & PubMed corpus |
| ❌ No verifiable citations or document provenance | ✅ **Every clinical claim** mapped to inline citations `[c0]`, `[c1]` |
| ❌ Guesses when medical data is absent | ✅ Explicitly responds **"I don't know / Insufficient context"** |
| ❌ Single unverified inference call | ✅ **5 specialized agents** with Critic verification & retry loops |

---

## 🏗️ System Architecture

```
User Query: "What are treatments for Long COVID?"
                         │
                         ▼
             ┌─────────────────────────┐
             │    Next.js Frontend     │  (Vercel AI SDK, shadcn/ui, Supabase Auth)
             └───────────┬─────────────┘
                         │ JWT Token + Streaming Request
                         ▼
             ┌─────────────────────────┐
             │     FastAPI Backend     │  (JWT Verification, Request Telemetry, CORS)
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ [1] Supervisor Agent    │  ← LangGraph Node 1
             │ Classifies: "covid"     │  (Groq qwen3.8-27b: Sets top_k=8, alpha=0.65)
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ [2] Retrieval Agent     │  ← LangGraph Node 2
             │ Hybrid Dense + Sparse   │  (BM25Okapi + Qdrant Cosine Vector Search)
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ [3] Critic Agent        │  ← LangGraph Node 3
             │ Scores Chunks (1 - 5)   │  (Filters score < 3; triggers dynamic retry if < 2 pass)
             └───────────┬─────────────┘
                         │ Passing Chunks
                         ▼
             ┌─────────────────────────┐
             │ [4] Synthesizer Agent   │  ← LangGraph Node 4
             │ Strictly Grounded LLM   │  (Generates clinical answer with [c0], [c1] markers)
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ [5] Citation Agent      │  ← LangGraph Node 5
             │ Maps Claims to Sources  │  (Resolves markers → URL, title, snippet & coverage %)
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │  FastAPI Token Stream   │  (x-vercel-ai-ui-message-stream: v1)
             └─────────────────────────┘
```

---

## 📚 Knowledge Base Corpus (3 Verified Sources)

1. **MedQuAD (NIH)**: 14,344 deduplicated Q&A pairs spanning 12 NIH institutes (GARD rare diseases, GHR genetics, CancerGov, NIDDK, NINDS, CDC).
2. **MedlinePlus (NLM)**: 1,014 full-length comprehensive health topic articles (avg 351 words) providing rich contextual retrieval.
3. **PubMed (NCBI Entrez)**: 19,089 recent peer-reviewed clinical research abstracts (2020–2026) focusing on COVID-19, Long COVID, immunotherapies, and clinical trials.
* **Total Merged Corpus**: **33,842 verified documents** (7.7+ million words) → **62,355 token-based chunks**.

---

## 🛠️ Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Multi-Agent Orchestration** | `LangGraph`, `LangChain` | StateGraph workflow, conditional loops, retry routing |
| **LLM Inference** | `Groq` (`qwen/qwen3.8-27b` via `GROQ_MODEL` env var) | Ultra-fast classification, critic validation, and synthesis |
| **Dense Embeddings** | `Cohere` (`embed-english-v3.0`, 1024d) & `MiniLM-L6-v2` (384d) | High-accuracy semantic retrieval (API & local zero-cost modes) |
| **Sparse Retrieval** | `rank-bm25` (`BM25Okapi`) | Exact medical keyword, drug name, and acronym matching |
| **Vector Store** | `Qdrant` (Local on-disk / Docker & Qdrant Cloud) | Cosine similarity vector search with metadata payload filtering |
| **Backend API** | `FastAPI`, `Uvicorn`, `Pydantic` | Async REST endpoints, Vercel AI SDK SSE token streaming |
| **Auth & Database** | `Supabase` (PostgreSQL, GoTrue JWT) | User sessions, message history, Google OAuth |
| **Evaluation** | `pytest`, `rouge-score`, `matplotlib`, `seaborn` | Automated Recall@K, MRR@10, and ROUGE-L benchmarks |

---

## ⚙️ Prerequisites & Environment Variables

Copy `.env.example` to `.env` and configure your keys:

```bash
# --- LLM & Inference ---
GROQ_API_KEY=your_groq_api_key                # Free at https://console.groq.com/keys

# --- Embeddings ---
COHERE_API_KEY=your_cohere_api_key            # Free trial at https://dashboard.cohere.com/api-keys
EMBEDDING_MODEL=minilm                        # 'minilm' (local zero-cost) or 'cohere' (1024-dim API)

# --- NCBI PubMed API ---
NCBI_API_KEY=your_ncbi_api_key                # Free at https://www.ncbi.nlm.nih.gov/account/settings/
NCBI_EMAIL=your_email@example.com

# --- Vector Database ---
QDRANT_MODE=local                             # 'local' for embedded/Docker or 'cloud' for Qdrant Cloud
QDRANT_LOCAL_URL=http://localhost:6333
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_PING_INTERVAL_DAYS=5                   # Keep-alive ping preventing free tier suspension

# --- Supabase ---
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key

# --- Retrieval Hyperparameters ---
CHUNK_SIZE=256
CHUNK_OVERLAP=32
RETRIEVAL_METHOD=hybrid                       # 'hybrid', 'dense', 'bm25', 'mmr'
HYBRID_ALPHA=0.6
TOP_K=5
PUBMED_FETCH_LIMIT=20000
```

---

## 🚀 Step-by-Step Setup & Execution

### 1. Setup Environment & Install Dependencies
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip & install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Ingest Data Corpus (Phase 1)
```bash
# 1. Parse MedQuAD dataset (14,344 Q&A pairs)
python -m ingestion.medquad_parser

# 2. Parse MedlinePlus XML (1,014 full articles)
python -m ingestion.medlineplus_parser

# 3. Fetch PubMed abstracts via NCBI API (19,089 abstracts)
python -m ingestion.pubmed_fetcher

# 4. Merge & deduplicate all 3 sources into unified corpus
python -m ingestion.merge_corpus
```

### 3. Chunking & Embeddings (Phase 2)
```bash
# Generate 62,355 token-based chunks with overlap
python -m retrieval.chunker

# Generate dense vector embeddings (MiniLM local or Cohere API)
python -m retrieval.embedder --model minilm
```

### 4. Run Unit & Integration Tests (22 Tests)
```bash
pytest tests/ -v
```

### 5. Launch FastAPI Backend Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
* **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 6. Docker Quick Start (Recommended for Collaboration)

If both Person A and Person B are working together, Docker ensures identical environments:

```bash
# Start all services (backend + Qdrant + frontend when ready)
docker-compose up --build

# Or start only backend + Qdrant (Person A workflow)
docker-compose up qdrant backend --build

# Or start only Qdrant for local development
docker-compose up qdrant
```

| Service | URL | Purpose |
| :--- | :--- | :--- |
| Backend API | http://localhost:8000 | FastAPI + LangGraph pipeline |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector store UI |
| Frontend | http://localhost:3000 | Next.js chat UI (when available) |

> **Note:** Data directory is mounted as a volume — `data/processed/`, `data/embeddings/` are shared between host and container. Run the ingestion pipeline on your host first, then `docker-compose up` will pick up the data automatically.

---

## 📊 Evaluation & Benchmarks (Phase 7)

Run the automated quantitative evaluation matrix across all 8 configurations:
```bash
# Run 8-configuration evaluation benchmark
python -m eval.evaluate --queries 100

# Generate publication-quality visualization figures
python -m eval.plots
```

Generated charts saved to `results/plots/`:
- `recall_k.png`: Recall@1, Recall@5, Recall@10 across retrieval configurations
- `rouge_l.png`: Answer generation quality (ROUGE-L F1) vs ground truth NIH references
- `mrr_chart.png`: Mean Reciprocal Rank (MRR@10) across Dense vs Hybrid vs MMR
- `citation_coverage.png`: Fact citation coverage percentage across configurations

---

## 💡 Key Architectural Questions & Solutions

### Q1: Why use a Multi-Agent architecture instead of a single LLM prompt?
A single LLM call is prone to hallucinations, cannot verify whether retrieved passages actually answer the clinical query, and has no recovery mechanism if the first search is noisy. Our LangGraph pipeline separates concerns:
- **Supervisor** sets specialized search filters based on query intent (e.g., boosting PubMed for COVID trials vs MedQuAD for drug monographs).
- **Critic** evaluates retrieved chunks *before* generation, discarding off-topic noise and triggering an automatic retry with an expanded search window if evidence is weak.
- **Synthesizer** focuses exclusively on strictly cited generation from verified text.
- **Citation Agent** guarantees end-to-end auditability and provenance.

### Q2: Why is Hybrid Retrieval (Dense + BM25) essential for medical queries?
Dense vector search excels at conceptual matching (e.g., mapping *"trouble catching breath after illness"* to *"post-viral dyspnea"*), but often struggles with exact pharmaceutical brand names, rare disease acronyms (e.g., *PASC, ALL, SGLT2*), and specific gene variants. BM25 guarantees exact keyword recall for clinical terminology, while dense embeddings capture semantics. Combining both with score normalization ($\alpha=0.6$) achieves superior Recall@K and MRR@10.

### Q3: How does the Critic Agent prevent clinical hallucinations?
The Critic inspects retrieved chunks against the query using a structured JSON evaluation prompt, scoring each candidate from 1 to 5. Chunks scoring below 3 (tangential or irrelevant) are discarded. If fewer than 2 relevant chunks survive, the Critic forces a retry loop. If context remains insufficient after 2 retries, the system explicitly declares insufficient context rather than hallucinating plausible-sounding medical advice.

### Q4: Why token-based chunking with overlap rather than character or sentence chunking?
Medical articles frequently contain complex multi-clause sentences, clinical trial parameter tables, and dosage schedules. Character-based chunking can slice critical numbers or medical terms in half. Using `tiktoken` (`cl100k_base`) ensures clean token boundaries matching LLM context windows. Overlapping chunks (32 tokens) ensure clinical facts spanning chunk borders are not lost during semantic indexing.

### Q5: How is 100% claim traceability achieved?
The Synthesizer prompt strictly enforces inline markers `[c0]`, `[c1]`, etc. immediately after any medical statement. The Citation Agent parses these markers, validates that the referenced index exists in the Critic-approved chunks, extracts the exact source URL and title, and computes the `citation_coverage` metric. Every card rendered in the UI directly links to the official NIH / PubMed article.

### Q6: How does Qdrant dual-mode optimize development and deployment?
Qdrant Cloud free tier provides 1GB storage, which can be exhausted during rapid experimentation with multiple embedding models. Our `MedicalVectorStore` supports `QDRANT_MODE=local` (embedded on-disk or local Docker) for zero-cost, offline development and testing, reserving the Qdrant Cloud cluster for production staging. Additionally, an automated background keep-alive ping prevents free tier cluster suspension after 1 week of inactivity.
