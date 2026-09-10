import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from evaluation.runner import write_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate intent and escalation baselines on a labeled golden set.")
    parser.add_argument("--golden-set", type=Path, default=Path("evaluation/golden_set.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()
    report = write_report(args.golden_set, args.output, get_settings())
    print(f"Evaluated {report['records']} records. Results written to {args.output}.")