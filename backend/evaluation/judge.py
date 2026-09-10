from __future__ import annotations

import json
from typing import Any

import requests

from app.config import Settings


JUDGE_RUBRIC = """Score each dimension from 1 to 5 using only the customer message, evidence, and draft reply.
1 means unacceptable and 5 means excellent.
- reply_quality: relevance, clarity, empathy, and professional tone.
- grounding_evidence_correctness: whether the reply is supported by the supplied historical evidence and avoids unsupported promises.
Return JSON only: {\"reply_quality\": integer, \"grounding_evidence_correctness\": integer, \"reason\": string}."""


def judge_reply(record: dict[str, Any], settings: Settings) -> dict[str, Any]:
    prompt = f"""You are an evaluator for a customer-support agent.\n{JUDGE_RUBRIC}\n\nCustomer message:\n{record['message']}\n\nEvidence:\n{json.dumps(record.get('evidence', []))}\n\nDraft reply:\n{record.get('reply', '')}"""
    response = requests.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=settings.ollama_timeout_seconds,
    )
    response.raise_for_status()
    result = json.loads(response.json()["response"])
    return {"judge_score": result["reply_quality"], "judge_grounding_score": result["grounding_evidence_correctness"], "judge_reason": result.get("reason", "")}