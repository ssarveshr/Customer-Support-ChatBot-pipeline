from app.config import Settings
from app.services.retrieval import RetrievalService


def test_retrieval_returns_empty_collection_without_examples(monkeypatch, tmp_path):
    class EmptyCollection:
        def count(self):
            return 0

    service = RetrievalService(Settings(chroma_path=tmp_path))
    service._collection = EmptyCollection()
    assert service.retrieve_similar_examples("hello") == []
