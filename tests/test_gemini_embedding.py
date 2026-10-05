import pytest

from memu.app.service import MemoryService
from memu.embedding.backends.gemini import GeminiEmbeddingBackend
from memu.embedding.backends.openai import OpenAIEmbeddingBackend
from memu.embedding.http_client import HTTPEmbeddingClient


def test_gemini_embedding_payloads_and_profile_routing() -> None:
    backend = GeminiEmbeddingBackend()
    text = backend.build_embedding_payload(inputs=["hello"], embed_model="gemini-embedding-2")
    media = backend.build_media_embedding_payload(
        data=b"image", mime_type="image/jpeg", embed_model="gemini-embedding-2"
    )
    assert text["requests"][0]["content"] == {"parts": [{"text": "hello"}]}
    assert text["requests"][0]["outputDimensionality"] == 3072
    assert media["content"]["parts"][0]["inline_data"] == {
        "mime_type": "image/jpeg",
        "data": "aW1hZ2U=",
    }

    service = MemoryService(
        llm_profiles={
            "default": {"provider": "openai"},
            "embedding": {
                "provider": "gemini",
                "base_url": "https://generativelanguage.googleapis.com/",
                "api_key": "test",
                "embed_model": "gemini-embedding-2",
            },
        },
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
    )
    assert isinstance(service._select_embedding_client(None)._client, HTTPEmbeddingClient)


def test_openai_embedding_response_restores_index_order() -> None:
    backend = OpenAIEmbeddingBackend()
    response = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }
    assert backend.parse_embedding_response(response) == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="indices"):
        backend.parse_embedding_response({"data": [{"index": 1, "embedding": [1.0]}]})
