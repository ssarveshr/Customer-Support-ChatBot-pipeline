from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Customer Support Chatbot"
    selected_brand: str = Field(default="", validation_alias="SELECTED_BRAND")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL"
    )
    ollama_model: str = Field(default="llama3.2:3b", validation_alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    chroma_path: Path = Field(default=PROJECT_ROOT / "backend/data/chroma", validation_alias="CHROMA_PATH")
    dataset_path: Path = Field(default=PROJECT_ROOT / "backend/data/raw", validation_alias="DATASET_PATH")
    normalized_data_path: Path = Field(
        default=PROJECT_ROOT / "backend/data/normalized_examples.parquet", validation_alias="NORMALIZED_DATA_PATH"
    )
    retrieval_top_k: int = Field(default=5, ge=1, le=50, validation_alias="RETRIEVAL_TOP_K")
    chroma_batch_size: int = Field(default=1000, ge=1, le=5000, validation_alias="CHROMA_BATCH_SIZE")
    intent_confidence_threshold: float = Field(default=0.45, ge=0, le=1, validation_alias="INTENT_CONFIDENCE_THRESHOLD")
    retrieval_similarity_threshold: float = Field(default=0.35, ge=0, le=1, validation_alias="RETRIEVAL_SIMILARITY_THRESHOLD")
    ollama_timeout_seconds: float = Field(default=60.0, gt=0, validation_alias="OLLAMA_TIMEOUT_SECONDS")
    max_examples: int = Field(default=10000, ge=1, validation_alias="MAX_EXAMPLES")

@lru_cache
def get_settings() -> Settings:
    return Settings()
