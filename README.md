
# ==========================================
# Medical RAG Multi-Agent — Environment Keys
# ==========================================

# --- Groq (LLM inference) ---
GROQ_API_KEY=
GROQ_MODEL=qwen/qwen3.8-27b

# --- Cohere (Embeddings — primary) ---
COHERE_API_KEY=

# --- NCBI / PubMed (Data ingestion) ---
NCBI_API_KEY=
NCBI_EMAIL=your_email@example.com

# --- Qdrant (Vector store) ---
QDRANT_MODE=local
QDRANT_LOCAL_URL=http://localhost:6333
QDRANT_URL=
QDRANT_API_KEY=

# --- Supabase (Auth + Database) ---
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_KEY=

# --- LangSmith (Tracing — optional) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=medical-rag

# --- App Config ---
EMBEDDING_MODEL=cohere
CHUNK_SIZE=256
CHUNK_OVERLAP=32
RETRIEVAL_METHOD=hybrid
HYBRID_ALPHA=0.6
TOP_K=5
PUBMED_FETCH_LIMIT=20000

# --- Frontend (frontend/.env.local — NOT this file) ---
# NEXT_PUBLIC_API_URL=
# NEXT_PUBLIC_SUPABASE_URL=
# NEXT_PUBLIC_SUPABASE_ANON_KEY=