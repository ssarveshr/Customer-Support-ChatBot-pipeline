import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.services.preprocessing import build_normalized_dataset


if __name__ == "__main__":
    settings = get_settings()
    build_normalized_dataset(settings.dataset_path, settings.normalized_data_path, settings.selected_brand, settings.max_examples)
