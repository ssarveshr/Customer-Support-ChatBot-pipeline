import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import evaluate_judge_agreement, evaluate_reply_quality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize existing judge and human calibration scores without calling Ollama.")
    parser.add_argument("--input", type=Path, default=Path("evaluation/judged_records_delta_final.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/judge_results_delta_final.json"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {**evaluate_reply_quality(records), **evaluate_judge_agreement(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))