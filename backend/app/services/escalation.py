from __future__ import annotations

from typing import Any

from app.config import Settings
from app.services.intent_classifier import IntentPrediction


class EscalationService:
    sensitive_terms = ("fraud", "scam", "lawsuit", "legal", "threat", "medical", "injury", "personal data")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def decide(self, message: str, prediction: IntentPrediction, evidence: list[dict[str, Any]], reply: str | None) -> tuple[str, str]:
        if prediction.confidence < self.settings.intent_confidence_threshold:
            return "ESCALATE", "Intent confidence is below the configured threshold."
        if not evidence or max(float(item.get("similarity", 0)) for item in evidence) < self.settings.retrieval_similarity_threshold:
            return "ESCALATE", "No sufficiently similar historical resolution was retrieved."
        if any(term in message.casefold() for term in self.sensitive_terms):
            return "ESCALATE", "The message contains a sensitive issue requiring human verification."
        if not reply:
            return "ESCALATE", "The LLM could not produce a grounded response."
        return "AUTO_HANDLE", "Intent, historical evidence, and a grounded response met the configured thresholds."
