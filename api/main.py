"""
Medical RAG Multi-Agent — FastAPI Application Entrypoint
=========================================================
Production REST API serving the multi-agent clinical RAG system.
Includes CORS, structured logging, JWT authentication, and Qdrant keep-alive pings.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.middleware.cors import configure_cors
from api.middleware.logging import LoggingMiddleware
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from retrieval.qdrant_client import MedicalVectorStore

load_dotenv()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("api.main")

QDRANT_PING_INTERVAL_DAYS = int(os.getenv("QDRANT_PING_INTERVAL_DAYS", "5"))


# ─── Background Keep-Alive Ping Task ────────────────────────────
async def qdrant_keepalive_worker():
    """Periodically pings Qdrant to prevent free tier inactivity suspension."""
    interval_seconds = max(QDRANT_PING_INTERVAL_DAYS, 1) * 86400  # Days to seconds
    store = MedicalVectorStore()
    while True:
        try:
            logger.info("[KeepAlive] Performing periodic Qdrant health ping...")
            store.ping()
        except Exception as e:
            logger.warning(f"[KeepAlive] Qdrant ping exception: {e}")
        await asyncio.sleep(interval_seconds)


# ─── Lifespan Context Manager ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("Starting Medical RAG Multi-Agent Backend API")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info("=" * 60)

    # Launch background keep-alive ping task only for cloud deployment
    ping_task = None
    if os.getenv("QDRANT_MODE", "local").lower() == "cloud" and os.getenv("QDRANT_URL"):
        ping_task = asyncio.create_task(qdrant_keepalive_worker())

    yield

    # Shutdown
    if ping_task:
        ping_task.cancel()
    logger.info("Medical RAG Backend API shut down gracefully.")


# ─── App Initialization ─────────────────────────────────────────
app = FastAPI(
    title="Medical RAG Multi-Agent API",
    description=(
        "Production-grade Medical AI Assistant backed by verified NIH, MedlinePlus, "
        "and PubMed data with LangGraph multi-agent orchestration and strict citation tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register Middleware
configure_cors(app)
app.add_middleware(LoggingMiddleware)

# Register Routes
app.include_router(health_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    """Root landing endpoint."""
    return {
        "service": "Medical RAG Multi-Agent API",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "chat_endpoint": "/chat",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
