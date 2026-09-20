"""
Health Check Route
==================
Exposes system health status and corpus size per API contract.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])

CORPUS_SIZE = 33842


@router.get("/health")
async def health_check():
    """
    Returns API health status and corpus size (instant response).
    """
    return {
        "status": "ok",
        "corpus_size": CORPUS_SIZE,
    }
