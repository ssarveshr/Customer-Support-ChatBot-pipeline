from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float


class IntentClassifier:
    """Replaceable baseline intent classifier using transparent keyword rules."""

    rules: dict[str, tuple[str, ...]] = {
        "refund_or_cancellation": ("refund", "cancel", "cancelled", "canceled", "money back", "chargeback"),
        "baggage": ("bag", "baggage", "luggage", "suitcase"),
        "delay_or_schedule": ("delay", "delayed", "late", "schedule", "flight time"),
        "booking_or_reservation": ("book", "booking", "reservation", "ticket", "confirmation"),
        "payment_or_charge": ("charged", "payment", "card", "price", "fee", "invoice"),
        "account_or_login": ("login", "log in", "password", "account"),
    }

    def predict(self, message: str) -> IntentPrediction:
        text = message.casefold()
        scores = {intent: sum(bool(re.search(rf"\b{re.escape(term)}\b", text)) for term in terms) for intent, terms in self.rules.items()}
        intent, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            return IntentPrediction("other", 0.2)
        return IntentPrediction(intent, min(0.55 + (score - 1) * 0.15, 0.95))
