from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

from app.services.intent_classifier import IntentClassifier, IntentPrediction


class Classifier(Protocol):
    def predict(self, message: str) -> IntentPrediction:
        ...


@dataclass
class MajorityIntentClassifier:
    """Trivial baseline that always predicts the most frequent training intent."""

    majority_intent: str = "other"

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> "MajorityIntentClassifier":
        counts = Counter(label for label in labels if label)
        return cls(counts.most_common(1)[0][0] if counts else "other")

    def predict(self, message: str) -> IntentPrediction:
        return IntentPrediction(self.majority_intent, 1.0)


class RuleIntentClassifier(IntentClassifier):
    """Named adapter for the simple keyword/rule baseline."""
