import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.config import get_settings
from app.services.retrieval import RetrievalService


if __name__ == "__main__":
    settings = get_settings()
    frame = pd.read_parquet(settings.normalized_data_path)
    examples = frame.to_dict(orient="records")
    count = RetrievalService(settings).index_examples(examples)
    print(f"Indexed {count} examples")
