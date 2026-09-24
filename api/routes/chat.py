"""
Chat API Routes
===============
Endpoints for running the multi-agent medical RAG pipeline and streaming
real-time token responses to the Next.js frontend.

API Contract:
  POST /chat              — stream answer + final JSON with citations
  GET  /chat/sessions     — list sessions (plain array)
  GET  /chat/sessions/{id} — session message history
  DELETE /chat/sessions/{id} — delete session → 204
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.middleware.auth import get_current_user
from graph.rag_graph import run_medical_rag_pipeline
from graph.state import RAGState
from agents.supervisor import supervisor_agent
from agents.retrieval_agent import retrieval_agent
from agents.critic_agent import critic_agent
from agents.synthesizer_agent import stream_synthesizer_response, synthesizer_agent
from agents.citation_agent import citation_agent

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

# In-memory session store (persists during process lifetime; Supabase can mirror this)
_IN_MEMORY_SESSIONS: dict[str, dict] = {}
_IN_MEMORY_MESSAGES: dict[str, list[dict]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Request Model ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's medical question")
    session_id: Optional[str] = Field(None, description="Existing session UUID, or null to start new")


# ─── POST /chat ─────────────────────────────────────────────────
@router.post("")
@router.post("/")
async def chat_endpoint(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Primary multi-agent chat endpoint.
    Streams tokens as plain text, then sends a final JSON object with
    answer, citations, citation_coverage, and session_id.
    """
    user_query = req.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    user_id = current_user.get("user_id", "guest_user")
    now = _now_iso()

    # Build chat history from existing session messages
    chat_history = []
    if session_id in _IN_MEMORY_MESSAGES:
        for msg in _IN_MEMORY_MESSAGES[session_id]:
            chat_history.append({"role": msg["role"], "content": msg["content"]})

    logger.info(f"[/chat] query from '{user_id}': '{user_query[:70]}...' (session={session_id})")

    # Record session
    if session_id not in _IN_MEMORY_SESSIONS:
        _IN_MEMORY_SESSIONS[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "title": user_query[:40] + ("..." if len(user_query) > 40 else ""),
            "created_at": now,
        }
        _IN_MEMORY_MESSAGES[session_id] = []

    # Store user message
    _IN_MEMORY_MESSAGES[session_id].append({
        "role": "user",
        "content": user_query,
        "created_at": now,
    })

    # ── Streaming Pipeline ──────────────────────────────────────
    async def event_stream():
        # Flush HTTP headers immediately to prevent timeout on client
        yield ": connected\n\n"

        # Step 1: Supervisor classification
        initial_state: RAGState = {
            "query": user_query,
            "session_id": session_id,
            "chat_history": chat_history,
            "query_type": "general",
            "top_k": 6,
            "retrieval_method": "hybrid",
            "source_filter": None,
            "alpha": 0.6,
            "retrieved_chunks": [],
            "filtered_chunks": [],
            "critic_scores": [],
            "retry_count": 0,
            "insufficient_context": False,
            "answer": "",
            "citations": [],
            "citation_coverage": 0.0,
            "status_message": "Analyzing query...",
        }

        # Run Supervisor → Retriever → Critic
        sup_out = supervisor_agent(initial_state)
        state = {**initial_state, **sup_out}

        ret_out = retrieval_agent(state)
        state = {**state, **ret_out}

        crit_out = critic_agent(state)
        state = {**state, **crit_out}

        # Retry once if critic needs more chunks
        if crit_out.get("retry_count", 0) > initial_state.get("retry_count", 0):
            ret_out = retrieval_agent(state)
            state = {**state, **ret_out}
            crit_out = critic_agent(state)
            state = {**state, **crit_out}

        filtered_chunks = state.get("filtered_chunks", [])

        # Stream Synthesizer tokens as plain text
        full_answer = []
        for token in stream_synthesizer_response(user_query, filtered_chunks, chat_history):
            full_answer.append(token)
            # Vercel AI SDK Text Part format: 0:"<text>"\n
            escaped = json.dumps(token)
            yield f"0:{escaped}\n"

        complete_text = "".join(full_answer)
        state["answer"] = complete_text

        # Compute citations
        cit_out = citation_agent(state)
        citations = cit_out.get("citations", [])
        coverage = cit_out.get("citation_coverage", 0.0)

        ans_time = _now_iso()

        # Save assistant message
        _IN_MEMORY_MESSAGES[session_id].append({
            "role": "assistant",
            "content": complete_text,
            "citations": citations,
            "created_at": ans_time,
        })

        # ── Final JSON payload (contract shape) ─────────────────
        final_payload = json.dumps({
            "answer": complete_text,
            "citations": citations,
            "citation_coverage": coverage,
            "session_id": session_id,
        })
        # Vercel AI SDK Data Part: 2:[{...}]\n
        yield f"2:[{final_payload}]\n"

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "x-vercel-ai-ui-message-stream": "v1",
    }
    return StreamingResponse(event_stream(), headers=headers)


# ─── GET /chat/sessions ────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    """List all chat sessions for the current user. Returns a plain JSON array."""
    user_id = current_user.get("user_id", "guest_user")
    user_sessions = [
        {"id": s["id"], "title": s["title"], "created_at": s["created_at"]}
        for s in reversed(_IN_MEMORY_SESSIONS.values())
        if s.get("user_id") == user_id or user_id in ("guest_user", "dev_user_001")
    ]
    return user_sessions  # plain array, not wrapped in {"sessions": ...}


# ─── GET /chat/sessions/{id} ───────────────────────────────────
@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str, current_user: dict = Depends(get_current_user)):
    """Retrieve complete message history for a session."""
    if session_id not in _IN_MEMORY_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    raw_messages = _IN_MEMORY_MESSAGES.get(session_id, [])
    messages = []
    for msg in raw_messages:
        entry: dict[str, Any] = {
            "role": msg["role"],
            "content": msg["content"],
            "created_at": msg.get("created_at", ""),
        }
        if msg["role"] == "assistant" and "citations" in msg:
            entry["citations"] = msg["citations"]
        messages.append(entry)

    return {
        "id": session_id,
        "messages": messages,
    }


# ─── DELETE /chat/sessions/{id} ────────────────────────────────
@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a session and its messages. Returns 204 No Content."""
    _IN_MEMORY_SESSIONS.pop(session_id, None)
    _IN_MEMORY_MESSAGES.pop(session_id, None)
    return Response(status_code=204)
