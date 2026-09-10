from __future__ import annotations

import logging
import re
from typing import Any

import requests

from app.config import Settings

logger = logging.getLogger(__name__)


class ResponseGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, message: str, examples: list[dict[str, Any]]) -> str | None:
        if not examples:
            return None
        brand = self.settings.selected_brand or "the selected brand"
        evidence = "\n\n".join(
            f"Customer: {example['customer_message']}\nHistorical response: {example.get('brand_response', '')}"
            for example in examples
        )
        prompt = f"""You are {brand}'s customer support agent.
    Answer the customer message using only the historical examples below as grounding. Do not invent policies, refunds, timelines, guarantees, usernames, handles, booking references, ticket numbers, or account identifiers. Never output placeholders such as @YourTicketNumber, <ticket_number>, [name], or similar template text. If a required identifier is missing, ask the customer to provide it or recommend human review. Keep the response concise and professional.

    Customer message:
    {message}

    Historical examples:
    {evidence}"""
        try:
            response = requests.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                json={"model": self.settings.ollama_model, "prompt": prompt, "stream": False},
                timeout=self.settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if not text or _contains_unresolved_placeholder(text):
                logger.warning("Rejected an ungrounded placeholder in the generated response")
                return None
            return text
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("Ollama response generation failed: %s", exc)
            return None


def _contains_unresolved_placeholder(text: str) -> bool:
    patterns = (
        r"@[A-Za-z]*Your[A-Za-z0-9_]*",
        r"<[^>]+>",
        r"\[[^\]]+\]",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
