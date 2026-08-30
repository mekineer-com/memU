from __future__ import annotations

import base64
from typing import Any

from memu.embedding.backends.base import EmbeddingBackend


class GeminiEmbeddingBackend(EmbeddingBackend):
    name = "gemini"
    embedding_endpoint = "v1beta/models/{embed_model}:batchEmbedContents"
    media_embedding_endpoint = "v1beta/models/{embed_model}:embedContent"

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        return {
            "requests": [
                {
                    "model": f"models/{embed_model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": 3072,
                }
                for text in inputs
            ]
        }

    def build_media_embedding_payload(
        self, *, data: bytes, mime_type: str, embed_model: str
    ) -> dict[str, Any]:
        return {
            "model": f"models/{embed_model}",
            "content": {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(data).decode("ascii"),
                        }
                    }
                ]
            },
            "outputDimensionality": 3072,
        }

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        return [row["values"] for row in data["embeddings"]]

    def parse_media_embedding_response(self, data: dict[str, Any]) -> list[float]:
        return data["embedding"]["values"]
