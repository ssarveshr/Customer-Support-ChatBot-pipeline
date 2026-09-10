from __future__ import annotations

import logging
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Sentence-Transformers embeddings backed by a persistent Chroma collection."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._collection: Any = None
        self._embedder: Any = None

    def _ensure_ready(self) -> None:
        if self._collection is not None:
            return
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self._collection = client.get_or_create_collection("support_examples")
        self._embedder = SentenceTransformer(self.settings.embedding_model)

    def retrieve_similar_examples(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        self._ensure_ready()
        count = top_k or self.settings.retrieval_top_k
        if self._collection.count() == 0:
            return []
        embedding = self._embedder.encode(query).tolist()
        result = self._collection.query(query_embeddings=[embedding], n_results=count)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        examples = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            examples.append({"customer_message": document, "similarity": max(0.0, 1.0 - float(distance)), "metadata": metadata or {}, "brand_response": (metadata or {}).get("brand_response", ""), "intent": (metadata or {}).get("intent")})
        return examples

    def index_examples(self, examples: list[dict[str, Any]]) -> int:
        self._ensure_ready()
        if not examples:
            return 0
        documents = [example["customer_message"] for example in examples]
        embeddings = self._embedder.encode(documents).tolist()
        ids = [str(example.get("id", index)) for index, example in enumerate(examples)]
        metadatas = [{"brand_response": example.get("brand_response", ""), "intent": example.get("intent", ""), **example.get("metadata", {})} for example in examples]
        batch_size = self.settings.chroma_batch_size
        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        return len(examples)
