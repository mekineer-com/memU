from __future__ import annotations

from typing import Any, cast

from memu.embedding.backends.base import EmbeddingBackend


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Backend for OpenAI-compatible embedding API."""

    name = "openai"
    embedding_endpoint = "/embeddings"

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        return {"model": embed_model, "input": inputs}

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        rows = data["data"]
        indices = [row["index"] for row in rows]
        if sorted(indices) != list(range(len(rows))) or len(set(indices)) != len(rows):
            raise ValueError("OpenAI embedding response indices are invalid")
        return [cast(list[float], row["embedding"]) for row in sorted(rows, key=lambda row: row["index"])]
