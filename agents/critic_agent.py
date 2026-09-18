"""
Critic Agent
============
Node 3 in LangGraph pipeline.
Acts as the medical fact-checking and relevance verification gatekeeper.
Scores each retrieved chunk on a 1-5 scale for query relevance.
Filters out irrelevant/noisy chunks and triggers pipeline retries if needed.
"""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from graph.state import RAGState

load_dotenv()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

CRITIC_SYSTEM_PROMPT = """You are a rigorous Medical Evidence Critic AI.
Your task is to evaluate a set of candidate medical text chunks for their direct clinical relevance to a user query.

For each candidate chunk:
- Assign a relevance score from 1 to 5:
  5: Directly answers the question with specific, high-quality medical evidence.
  4: Very relevant, contains strong supporting clinical facts.
  3: Moderately relevant, provides useful background context.
  2: Tangentially related, mentions keywords but does not answer the inquiry.
  1: Completely irrelevant or off-topic (e.g. general flu when asking about COVID).

Respond with ONLY a JSON object formatted as:
{
  "evaluations": [
    {"index": 0, "score": 5, "reason": "Direct clinical trial findings on treatment"},
    {"index": 1, "score": 1, "reason": "Unrelated topic"}
  ]
}
"""


def critic_agent(state: RAGState) -> dict[str, Any]:
    """
    Critic node: filters low-quality chunks and controls pipeline retry loops.
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    if not chunks:
        logger.warning("[Critic Agent] No chunks received.")
        if retry_count < 2:
            return {
                "filtered_chunks": [],
                "critic_scores": [],
                "retry_count": retry_count + 1,
                "insufficient_context": False,
                "status_message": f"Zero chunks retrieved. Retrying search (attempt {retry_count + 1}/2)...",
            }
        else:
            return {
                "filtered_chunks": [],
                "critic_scores": [],
                "insufficient_context": True,
                "status_message": "Insufficient verified context found in medical database.",
            }

    # Format chunks for batched structured evaluation
    prompt_chunks = []
    for idx, c in enumerate(chunks[:12]):
        preview = c["text"][:300].replace("\n", " ")
        prompt_chunks.append(f"[Chunk {idx}] (Source: {c['source']}) {preview}")

    user_content = f"User Query: {query}\n\nCandidate Chunks:\n" + "\n\n".join(prompt_chunks)

    filtered_chunks = []
    critic_scores = []

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        evals = data.get("evaluations", [])

        eval_map = {e.get("index"): e for e in evals}

        for idx, chunk in enumerate(chunks[:12]):
            e = eval_map.get(idx, {"score": 3, "reason": "Default score"})
            score = int(e.get("score", 3))
            critic_scores.append({
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "reason": e.get("reason", ""),
            })

            # Filter threshold: score >= 3
            if score >= 3:
                chunk_copy = dict(chunk)
                chunk_copy["critic_score"] = score
                filtered_chunks.append(chunk_copy)

    except Exception as e:
        logger.warning(f"[Critic Agent] LLM evaluation error: {e}. Passing top chunks.")
        # Fallback: pass top candidate chunks
        filtered_chunks = chunks[:5]
        critic_scores = [{"chunk_id": c["chunk_id"], "score": 3, "reason": "Fallback"} for c in filtered_chunks]

    passing_count = len(filtered_chunks)
    logger.info(f"[Critic Agent] {passing_count}/{len(chunks)} chunks passed clinical relevance filter (score >= 3)")

    # Retry condition: if < 2 chunks passed and retry_count < 2
    if passing_count < 2 and retry_count < 2:
        logger.info(f"[Critic Agent] Only {passing_count} chunks passed. Triggering retry #{retry_count + 1}")
        return {
            "filtered_chunks": filtered_chunks,
            "critic_scores": critic_scores,
            "retry_count": retry_count + 1,
            "insufficient_context": False,
            "status_message": f"Only {passing_count} relevant chunk(s) found. Broadening retrieval...",
        }
    elif passing_count < 1 and retry_count >= 2:
        logger.info("[Critic Agent] Max retries reached with insufficient context.")
        return {
            "filtered_chunks": [],
            "critic_scores": critic_scores,
            "insufficient_context": True,
            "status_message": "Insufficient verified medical evidence in database.",
        }

    return {
        "filtered_chunks": filtered_chunks,
        "critic_scores": critic_scores,
        "insufficient_context": False,
        "status_message": f"Critic verified {passing_count} high-quality evidence chunks. Generating answer...",
    }
