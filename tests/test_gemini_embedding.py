import pytest

from memu.app.service import MemoryService
from memu.embedding.backends.gemini import GeminiEmbeddingBackend
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


@pytest.mark.asyncio
async def test_service_exposes_raw_media_embedding() -> None:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}}
    )

    class Client:
        async def embed_media(self, data: bytes, mime_type: str) -> list[float]:
            assert (data, mime_type) == (b"image", "image/jpeg")
            return [1.0, 0.0]

    service._llm_clients["embedding"] = Client()
    assert await service.embed_media(b"image", "image/jpeg") == [1.0, 0.0]
