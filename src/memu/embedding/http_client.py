from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable

import httpx

from memu.embedding.backends.base import EmbeddingBackend
from memu.embedding.backends.doubao import DoubaoEmbeddingBackend
from memu.embedding.backends.gemini import GeminiEmbeddingBackend
from memu.embedding.backends.openai import OpenAIEmbeddingBackend


def _load_proxy() -> str | None:
    return os.getenv("MEMU_HTTP_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY") or None


logger = logging.getLogger(__name__)

EMBEDDING_BACKENDS: dict[str, Callable[[], EmbeddingBackend]] = {
    OpenAIEmbeddingBackend.name: OpenAIEmbeddingBackend,
    DoubaoEmbeddingBackend.name: DoubaoEmbeddingBackend,
    GeminiEmbeddingBackend.name: GeminiEmbeddingBackend,
}


class HTTPEmbeddingClient:
    """HTTP client for embedding APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        embed_model: str,
        provider: str = "openai",
        endpoint_overrides: dict[str, str] | None = None,
        timeout: int = 60,
    ):
        # Ensure base_url ends with "/" so httpx doesn't discard the path
        # component when joining with endpoint paths.
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key or ""
        self.embed_model = embed_model
        self.provider = provider.lower()
        self.backend = self._load_backend(self.provider)
        overrides = endpoint_overrides or {}
        raw_embedding_ep = (
            overrides.get("embeddings")
            or overrides.get("embedding")
            or overrides.get("embed")
            or self.backend.embedding_endpoint
        )
        # Strip leading "/" so httpx resolves relative to base_url
        self.embedding_endpoint = raw_embedding_ep.format(embed_model=self.embed_model).lstrip("/")
        self.timeout = timeout
        self.proxy = _load_proxy()

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        """
        Create text embeddings.

        Args:
            inputs: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        payload = self.backend.build_embedding_payload(inputs=inputs, embed_model=self.embed_model)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, proxy=self.proxy) as client:
            resp = await client.post(self.embedding_endpoint, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        logger.debug("HTTP embedding response: %s", data)
        vectors = self.backend.parse_embedding_response(data)
        if len(vectors) != len(inputs):
            raise ValueError("Embedding response count does not match input")
        if any(not vector or not all(math.isfinite(value) for value in vector) for vector in vectors):
            raise ValueError("Embedding response contains an invalid vector")
        return vectors

    async def embed_media(self, data: bytes, mime_type: str) -> list[float]:
        build = getattr(self.backend, "build_media_embedding_payload", None)
        parse = getattr(self.backend, "parse_media_embedding_response", None)
        endpoint = getattr(self.backend, "media_embedding_endpoint", None)
        if not callable(build) or not callable(parse) or not isinstance(endpoint, str):
            raise ValueError(f"Embedding provider '{self.provider}' does not support raw media")
        payload = build(data=data, mime_type=mime_type, embed_model=self.embed_model)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, proxy=self.proxy) as client:
            resp = await client.post(
                endpoint.format(embed_model=self.embed_model).lstrip("/"),
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            response = resp.json()
        vector = parse(response)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Media embedding response contains an invalid vector")
        return vector

    def _headers(self) -> dict[str, str]:
        if self.provider == "gemini":
            return {"x-goog-api-key": self.api_key}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _load_backend(self, provider: str) -> EmbeddingBackend:
        factory = EMBEDDING_BACKENDS.get(provider)
        if not factory:
            msg = f"Unsupported embedding provider '{provider}'. Available: {', '.join(EMBEDDING_BACKENDS.keys())}"
            raise ValueError(msg)
        return factory()
