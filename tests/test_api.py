"""
Integration Tests for FastAPI Endpoints
=======================================
Tests /, /health, /chat, and /chat/sessions using FastAPI TestClient.
Aligned with the API contract (contract.md).
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root landing route returns online status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_health_endpoint():
    """Verify health endpoint matches contract: {status: 'ok', corpus_size: N}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "corpus_size" in data
    assert isinstance(data["corpus_size"], int)


def test_list_sessions_endpoint():
    """Verify sessions listing returns a plain JSON array (not wrapped)."""
    response = client.get("/chat/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)  # plain array, not {"sessions": [...]}


def test_chat_endpoint_validation():
    """Verify empty chat request triggers validation error."""
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 400


def test_delete_nonexistent_session():
    """Verify deleting a nonexistent session returns 204 (idempotent)."""
    response = client.delete("/chat/sessions/nonexistent-id")
    assert response.status_code == 204
    assert response.content == b""  # empty body per contract


def test_get_nonexistent_session():
    """Verify getting a nonexistent session returns 404."""
    response = client.get("/chat/sessions/nonexistent-id")
    assert response.status_code == 404
