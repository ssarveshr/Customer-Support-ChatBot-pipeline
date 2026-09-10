import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from evaluation.golden_set import create_golden_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a reproducible unlabeled golden-set sample.")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("evaluation/golden_set_delta.jsonl"))
    args = parser.parse_args()
    settings = get_settings()
    destination = args.output
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing golden set: {destination}")
    count = create_golden_set(settings.normalized_data_path, destination, size=args.size)
    print(f"Created {count} records in {destination}. Label intent and should_escalate before evaluation.")