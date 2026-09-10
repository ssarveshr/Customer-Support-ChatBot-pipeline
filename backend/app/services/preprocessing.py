from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


CUSTOMER_COLUMNS = ("text", "tweet", "customer_message", "message", "inbound", "content")
RESPONSE_COLUMNS = ("response", "reply", "brand_response", "outbound", "agent_response")
BRAND_COLUMNS = ("brand", "company", "author", "airline")
CONVERSATION_COLUMNS = ("conversation_id", "conversation", "thread_id", "case_id")


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return str(normalized[candidate])
    for column in columns:
        lowered = str(column).lower()
        if any(candidate in lowered for candidate in candidates):
            return str(column)
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV, JSON/JSONL, or Parquet data without assuming exact column names."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    supported_suffixes = {".csv", ".json", ".jsonl", ".parquet", ".pq"}
    if path.is_file():
        source = path
    else:
        candidates = sorted(
            (file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in supported_suffixes),
            key=lambda file: (file.name.casefold() != "twcs.csv", str(file).casefold()),
        )
        source = candidates[0] if candidates else None
    if source is None:
        raise FileNotFoundError(
            f"No supported dataset files found in {path}. Expected CSV, JSON, JSONL, or Parquet."
        )
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(source, lines=suffix == ".jsonl")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported dataset format: {source.suffix}")


def normalize_dataset(frame: pd.DataFrame, selected_brand: str | None = None) -> pd.DataFrame:
    """Map common dataset schemas to a compact retrieval-ready format."""
    if frame.empty:
        return pd.DataFrame(columns=["customer_message", "brand_response", "brand", "conversation_id", "intent"])
    if {"tweet_id", "response_tweet_id", "inbound", "text"}.issubset(frame.columns):
        return _normalize_twitter_conversations(frame, selected_brand)
    customer_column = _find_column(frame.columns, CUSTOMER_COLUMNS)
    response_column = _find_column(frame.columns, RESPONSE_COLUMNS)
    brand_column = _find_column(frame.columns, BRAND_COLUMNS)
    conversation_column = _find_column(frame.columns, CONVERSATION_COLUMNS)
    if customer_column is None:
        raise ValueError(f"Could not identify a customer message column from: {list(frame.columns)}")
    if response_column is None:
        raise ValueError(f"Could not identify a brand response column from: {list(frame.columns)}")

    normalized = pd.DataFrame(
        {
            "customer_message": frame[customer_column].fillna("").astype(str),
            "brand_response": frame[response_column].fillna("").astype(str),
            "brand": frame[brand_column].fillna("").astype(str) if brand_column else "",
            "conversation_id": frame[conversation_column].fillna("").astype(str) if conversation_column else "",
        }
    )
    intent_column = _find_column(frame.columns, ("intent", "category", "label", "topic"))
    normalized["intent"] = frame[intent_column].fillna("").astype(str) if intent_column else ""
    if selected_brand and brand_column:
        normalized = normalized[normalized["brand"].str.casefold() == selected_brand.casefold()]
    normalized["customer_message"] = normalized["customer_message"].map(clean_text)
    normalized["brand_response"] = normalized["brand_response"].map(clean_text)
    normalized = normalized[
        (normalized["customer_message"].str.len() > 0) & (normalized["brand_response"].str.len() > 0)
    ].drop_duplicates(subset=["customer_message", "brand_response"])
    return normalized.reset_index(drop=True)


def _normalize_twitter_conversations(frame: pd.DataFrame, selected_brand: str | None) -> pd.DataFrame:
    """Pair inbound tweets with the outbound tweets listed in response_tweet_id."""
    tweets = frame.copy()
    tweets["tweet_id"] = tweets["tweet_id"].astype(str)
    tweets["text"] = tweets["text"].fillna("").astype(str).map(clean_text)
    tweets["inbound"] = tweets["inbound"].map(_as_bool)
    by_id = tweets.set_index("tweet_id")
    rows: list[dict[str, str]] = []
    for _, customer in tweets[tweets["inbound"].astype(bool)].iterrows():
        response_ids = str(customer.get("response_tweet_id", ""))
        for response_id in response_ids.split(","):
            response_key = response_id.strip()
            response = by_id.loc[response_key] if response_key in by_id.index else None
            if response is None or not response["text"]:
                continue
            rows.append(
                {
                    "customer_message": customer["text"],
                    "brand_response": response["text"],
                    "brand": str(response.get("author_id", "")),
                    "conversation_id": customer["tweet_id"],
                    "intent": "",
                }
            )
    normalized = pd.DataFrame(rows, columns=["customer_message", "brand_response", "brand", "conversation_id", "intent"])
    if selected_brand:
        normalized = normalized[normalized["brand"].str.casefold() == selected_brand.casefold()]
    return normalized.drop_duplicates(subset=["customer_message", "brand_response"]).reset_index(drop=True)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "t", "yes", "y"}


def clean_text(value: str) -> str:
    value = re.sub(r"https?://\S+|www\.\S+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def build_normalized_dataset(
    source: Path, destination: Path, selected_brand: str | None = None, max_examples: int = 10000
) -> pd.DataFrame:
    frame = normalize_dataset(load_dataset(source), selected_brand)
    if len(frame) > max_examples:
        frame = frame.sample(max_examples, random_state=42).reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    else:
        frame.to_parquet(destination, index=False)
    logger.info("Wrote %d normalized examples to %s", len(frame), destination)
    return frame
