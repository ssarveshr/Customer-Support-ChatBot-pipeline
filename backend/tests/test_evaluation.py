import json

import pytest

from evaluation.runner import evaluate_baselines
from evaluation.metrics import evaluate_reply_quality


def test_evaluation_rejects_unlabeled_golden_set(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps({"id": 1, "message": "hello", "intent": "LABEL_ME", "should_escalate": None}) + "\n")
    with pytest.raises(ValueError, match="unlabeled"):
        evaluate_baselines(path)


def test_evaluation_returns_baseline_metrics(tmp_path):
    path = tmp_path / "golden.jsonl"
    records = [
        {"id": 1, "message": "I need a refund", "intent": "refund_or_cancellation", "should_escalate": False},
        {"id": 2, "message": "My bag is missing", "intent": "baggage", "should_escalate": True},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    result = evaluate_baselines(path)
    assert result["records"] == 2
    assert "majority" in result["classifiers"]
    assert "rules" in result["classifiers"]


def test_reply_quality_aggregates_judge_fields():
    result = evaluate_reply_quality(
        [
            {"judge_score": 4, "judge_grounding_score": 5},
            {"judge_score": 2, "judge_grounding_score": 3},
        ]
    )
    assert result["reply_quality"] == 3.0
    assert result["grounding_evidence_correctness"] == 4.0
    assert result["reply_quality_scored_records"] == 2