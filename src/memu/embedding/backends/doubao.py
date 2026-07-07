from __future__ import annotations

from typing import Any, cast

from memu.embedding.backends.base import EmbeddingBackend


class DoubaoEmbeddingBackend(EmbeddingBackend):
    """Backend for Doubao embedding API."""

    name = "doubao"
    embedding_endpoint = "/api/v3/embeddings"

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        """Build payload for standard text embeddings."""
        return {"model": embed_model, "input": inputs, "encoding_format": "float"}

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        """Parse embedding response."""
        return [cast(list[float], d["embedding"]) for d in data["data"]]
