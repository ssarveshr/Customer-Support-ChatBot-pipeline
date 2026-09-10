import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from evaluation.judge import judge_reply
from evaluation.metrics import evaluate_judge_agreement, evaluate_reply_quality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the Ollama reply-quality and grounding rubric.")
    parser.add_argument("--input", type=Path, default=Path("evaluation/agent_records_delta_final.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/judged_records_delta_final.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("evaluation/judge_results_delta_final.json"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    settings = get_settings()
    judged = []
    for record in records:
        record.update(judge_reply(record, settings))
        judged.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record) for record in judged) + "\n", encoding="utf-8")
    summary = {**evaluate_reply_quality(judged), **evaluate_judge_agreement(judged)}
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))