"""
Synthesizer Agent
=================
Node 4 in LangGraph pipeline.
Generates hallucination-free, clinically sound answers strictly grounded
in the verified medical chunks provided by the Critic Agent.
Enforces inline citation markers [c0], [c1], etc. for claim traceability.
"""

import logging
import os
import re
from typing import Any, AsyncGenerator, Generator

from dotenv import load_dotenv
from groq import Groq
from graph.state import RAGState

load_dotenv()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

SYNTHESIZER_SYSTEM_PROMPT = """You are a production-grade, cited Medical AI Assistant backed exclusively by verified NIH, MedlinePlus, and PubMed clinical data.

RESPONSE FORMAT:
- Provide a comprehensive, well-structured clinical answer.
- Use **bold headers** to organize sections (e.g., **Overview**, **Common Symptoms**, **Treatment Options**).
- Use bullet points for lists of symptoms, medications, or criteria.
- Every factual claim MUST include an inline citation marker like [c0], [c1] referencing the provided context chunks.
- If context is insufficient, say so honestly.
- Maintain a professional, empathetic, and objective clinical tone.
- Do NOT include any <think> tags or internal reasoning. Respond directly with the answer.
"""


def build_context_prompt(chunks: list[dict]) -> str:
    """Format filtered chunks into numbered context blocks with source labels."""
    context_blocks = []
    for idx, c in enumerate(chunks):
        source = c.get("source", "NIH")
        title = c.get("title", "")
        text = c.get("text", "").strip()
        context_blocks.append(
            f"--- [Chunk c{idx}] (Source: {source} | Title: {title}) ---\n{text}"
        )
    return "\n\n".join(context_blocks)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def synthesizer_agent(state: RAGState) -> dict[str, Any]:
    """
    Synthesizer node: generates referenced response from verified evidence.
    """
    query = state.get("query", "")
    filtered_chunks = state.get("filtered_chunks", [])
    insufficient_context = state.get("insufficient_context", False)
    chat_history = state.get("chat_history", [])

    if insufficient_context or not filtered_chunks:
        answer = (
            "Based on the verified NIH and PubMed medical database, there is insufficient "
            "clinical evidence available to answer your specific question. Please consult a "
            "licensed healthcare provider for personalized medical advice."
        )
        return {
            "answer": answer,
            "status_message": "Generated response with insufficient context notice.",
        }

    context_str = build_context_prompt(filtered_chunks)

    # Assemble messages with conversation context
    messages = [{"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT}]

    # Include recent chat history (up to last 3 turns)
    for msg in chat_history[-6:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    user_prompt = f"User Question: {query}\n\nProvided Medical Context:\n{context_str}\n\nPlease generate a comprehensive medical answer with inline citations [c0], [c1], etc.:"
    messages.append({"role": "user", "content": user_prompt})

    logger.info(f"[Synthesizer Agent] Generating cited answer for: '{query[:60]}...'")

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,  # Low temperature for strict factual adherence
            max_tokens=1024,
        )
        answer = _strip_think_tags(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"[Synthesizer Agent] Groq generation error: {e}")
        answer = "An error occurred while generating the medical summary. Please try again."

    return {
        "answer": answer,
        "status_message": "Answer synthesized. Verifying citation coverage...",
    }


def stream_synthesizer_response(
    query: str,
    filtered_chunks: list[dict],
    chat_history: list[dict] = [],
) -> Generator[str, None, None]:
    """
    Streaming generator for real-time token delivery to FastAPI & Vercel AI SDK.
    Filters out <think>...</think> blocks from models that use chain-of-thought.
    """
    if not filtered_chunks:
        yield "Based on the verified NIH and PubMed database, there is insufficient clinical evidence to answer this question."
        return

    context_str = build_context_prompt(filtered_chunks)
    messages = [{"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    user_prompt = f"User Question: {query}\n\nProvided Medical Context:\n{context_str}\n\nPlease generate a comprehensive medical answer with inline citations [c0], [c1], etc.:"
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = Groq(api_key=GROQ_API_KEY)
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        # Buffer to detect and skip <think>...</think> blocks
        inside_think = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue

            # Handle <think> tag filtering
            if "<think>" in delta:
                inside_think = True
                # Yield any text before the tag
                before = delta.split("<think>")[0]
                if before:
                    yield before
                continue

            if inside_think:
                if "</think>" in delta:
                    inside_think = False
                    # Yield any text after the closing tag
                    after = delta.split("</think>")[-1]
                    if after:
                        yield after
                continue

            yield delta

    except Exception as e:
        logger.error(f"[Synthesizer Agent] Streaming error: {e}")
        yield f"An error occurred while generating the response: {str(e)}"
