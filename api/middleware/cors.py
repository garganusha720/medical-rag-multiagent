"""
CORS Middleware Configuration
=============================
Allows frontend requests from local Next.js dev server and deployed production domains.
"""

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


def configure_cors(app: FastAPI) -> None:
    """Register CORS middleware with permissive development origins."""
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "*",  # Allow all during development, restrict in production
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
