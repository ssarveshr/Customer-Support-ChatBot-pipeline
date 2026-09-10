from __future__ import annotations

from typing import Any, Sequence


def evaluate_intent(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Placeholder contract for accuracy and macro-F1 on the golden set."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return {"intent_accuracy": 0.0, "intent_macro_f1": 0.0}
    accuracy = sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)
    labels = set(y_true) | set(y_pred)
    f1_values = []
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(y_true, y_pred))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(y_true, y_pred))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"intent_accuracy": accuracy, "intent_macro_f1": sum(f1_values) / len(f1_values)}


def evaluate_escalation(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    """Placeholder contract for escalation precision and recall."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    tp = sum(actual and predicted for actual, predicted in zip(y_true, y_pred))
    fp = sum(not actual and predicted for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual and not predicted for actual, predicted in zip(y_true, y_pred))
    return {"escalation_precision": tp / (tp + fp) if tp + fp else 0.0, "escalation_recall": tp / (tp + fn) if tp + fn else 0.0}


def evaluate_reply_quality(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate human or judge scores already stored on evaluation records."""
    score_fields = {
        "reply_quality": ("reply_quality", "judge_score"),
        "grounding_evidence_correctness": ("grounding_evidence_correctness", "judge_grounding_score"),
    }
    result: dict[str, Any] = {"records": len(records)}
    for output_field, input_fields in score_fields.items():
        values = [
            float(record[next((field for field in input_fields if record.get(field) is not None), input_fields[0])])
            for record in records
            if any(record.get(field) is not None for field in input_fields)
        ]
        result[output_field] = sum(values) / len(values) if values else None
        result[f"{output_field}_scored_records"] = len(values)
    return result


def evaluate_judge_agreement(records: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    """Compare integer human and judge scores on the overlap of labeled records."""
    pairs = [
        (int(record["human_score"]), int(record["judge_score"]))
        for record in records
        if record.get("human_score") is not None and record.get("judge_score") is not None
    ]
    if not pairs:
        return {"agreement_records": 0, "exact_agreement": None, "within_one_agreement": None}
    return {
        "agreement_records": len(pairs),
        "exact_agreement": sum(actual == predicted for actual, predicted in pairs) / len(pairs),
        "within_one_agreement": sum(abs(actual - predicted) <= 1 for actual, predicted in pairs) / len(pairs),
    }
