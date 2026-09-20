"""
Supervisor Agent
================
Node 1 in LangGraph pipeline.
Analyzes the user question and conversation history to:
1. Classify query type ('covid', 'drug', 'treatment', 'symptom', 'general')
2. Set optimal retrieval strategy (top_k, alpha, retrieval_method, source_filter)
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


SUPERVISOR_SYSTEM_PROMPT = """You are an expert Clinical Supervisor AI.
Your job is to analyze the user's medical query and classify it into one of the following exact query types:
- 'covid': COVID-19, SARS-CoV-2, Long COVID, vaccines, post-viral sequelae
- 'drug': Medications, pharmacotherapy, drug dosages, side effects, interactions
- 'symptom': Signs, symptoms, diagnostic indicators, differential diagnosis
- 'treatment': Therapies, clinical guidelines, surgical/medical procedures, management plans
- 'general': Anatomy, medical definitions, genetics, general health inquiries

Respond with ONLY a JSON object in this exact format:
{
  "query_type": "covid" | "drug" | "symptom" | "treatment" | "general",
  "reasoning": "brief 1-sentence rationale"
}
"""


def supervisor_agent(state: RAGState) -> dict[str, Any]:
    """
    Supervisor node: classifies intent and configures retrieval parameters.
    """
    query = state.get("query", "")
    logger.info(f"[Supervisor Agent] Analyzing query: '{query[:80]}...'")

    query_type = "general"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {query}"},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        query_type = parsed.get("query_type", "general").lower()
    except Exception as e:
        logger.warning(f"[Supervisor Agent] Classification error: {e}. Defaulting to 'general'.")
        # Rule-based fallback
        q_lower = query.lower()
        if any(k in q_lower for k in ["covid", "sars", "long covid", "vaccine"]):
            query_type = "covid"
        elif any(k in q_lower for k in ["drug", "dose", "medication", "pill", "side effect"]):
            query_type = "drug"
        elif any(k in q_lower for k in ["symptom", "sign", "pain", "fever", "rash"]):
            query_type = "symptom"
        elif any(k in q_lower for k in ["treat", "cure", "therapy", "surgery", "rehab"]):
            query_type = "treatment"

    # Strategy Assignment based on classification
    if query_type == "covid":
        # PubMed has the most recent 2020-2026 COVID clinical data
        top_k = 8
        alpha = 0.65
        retrieval_method = "hybrid"
        source_filter = None
    elif query_type == "drug":
        # MedQuAD (MPlusDrugs) and MedlinePlus are rich in verified drug guides
        top_k = 8
        alpha = 0.55
        retrieval_method = "hybrid"
        source_filter = None
    elif query_type == "treatment":
        top_k = 8
        alpha = 0.60
        retrieval_method = "hybrid"
        source_filter = None
    elif query_type == "symptom":
        top_k = 6
        alpha = 0.50
        retrieval_method = "hybrid"
        source_filter = None
    else:
        top_k = 6
        alpha = 0.60
        retrieval_method = "hybrid"
        source_filter = None

    logger.info(f"[Supervisor Agent] Classified as '{query_type}' (top_k={top_k}, alpha={alpha})")

    return {
        "query_type": query_type,
        "top_k": top_k,
        "alpha": alpha,
        "retrieval_method": retrieval_method,
        "source_filter": source_filter,
        "retry_count": state.get("retry_count", 0),
        "insufficient_context": False,
        "status_message": f"Classified query as '{query_type}'. Initiating hybrid medical search...",
    }
