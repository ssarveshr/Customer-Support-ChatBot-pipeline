from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def create_golden_set(source: Path, destination: Path, size: int = 200, seed: int = 42) -> int:
    """Create a reproducible unlabeled sample for manual intent/escalation labeling."""
    frame = pd.read_parquet(source)
    if frame.empty:
        raise ValueError("The normalized dataset is empty")
    sample = frame.sample(min(size, len(frame)), random_state=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for index, row in sample.reset_index(drop=True).iterrows():
            output.write(
                json.dumps(
                    {
                        "id": index,
                        "message": row["customer_message"],
                        "historical_response": row["brand_response"],
                        "intent": "LABEL_ME",
                        "should_escalate": None,
                        "notes": "Label manually; do not use the historical response as a gold reply.",
                    }
                )
                + "\n"
            )
    return len(sample)


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]