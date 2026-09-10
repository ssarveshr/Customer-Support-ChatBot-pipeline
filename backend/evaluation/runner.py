from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.baselines import MajorityIntentClassifier, RuleIntentClassifier
from evaluation.golden_set import load_records
from evaluation.metrics import evaluate_escalation, evaluate_intent, evaluate_judge_agreement, evaluate_reply_quality


def _validate_labels(records: list[dict[str, Any]]) -> None:
    missing = [record.get("id") for record in records if not record.get("intent") or record["intent"] == "LABEL_ME" or record.get("should_escalate") is None]
    if missing:
        raise ValueError(f"Golden set has unlabeled records: {missing[:10]}")


def evaluate_baselines(path: Path, settings: Settings | None = None) -> dict[str, Any]:
    records = load_records(path)
    _validate_labels(records)
    labels = [str(record["intent"]) for record in records]
    majority = MajorityIntentClassifier.from_labels(labels)
    classifiers = {"majority": majority, "rules": RuleIntentClassifier()}
    result: dict[str, Any] = {"records": len(records), "classifiers": {}}
    for name, classifier in classifiers.items():
        predictions = [classifier.predict(str(record["message"])) for record in records]
        result["classifiers"][name] = evaluate_intent(labels, [prediction.intent for prediction in predictions])

    threshold = (settings or Settings()).intent_confidence_threshold
    rule_predictions = [RuleIntentClassifier().predict(str(record["message"])) for record in records]
    escalation_predictions = [prediction.confidence < threshold or prediction.intent == "other" for prediction in rule_predictions]
    result["rule_escalation"] = evaluate_escalation(
        [bool(record["should_escalate"]) for record in records], escalation_predictions
    )
    result["reply_quality"] = evaluate_reply_quality(records)
    result["judge_agreement"] = evaluate_judge_agreement(records)
    return result


def write_report(path: Path, output: Path, settings: Settings | None = None) -> dict[str, Any]:
    report = evaluate_baselines(path, settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report