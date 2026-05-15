from __future__ import annotations

import asyncio
import base64
import copy
import contextvars
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx

from memu.llm.backends.base import LLMBackend
from memu.llm.backends.openai import OpenAILLMBackend


def _load_proxy() -> str | None:
    return os.getenv("MEMU_HTTP_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY") or None


# Minimal embedding backend support (moved from embedding module)
class _EmbeddingBackend:
    name: str
    embedding_endpoint: str

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        raise NotImplementedError

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        raise NotImplementedError


class _OpenAIEmbeddingBackend(_EmbeddingBackend):
    name = "openai"
    embedding_endpoint = "/embeddings"

    def build_embedding_payload(self, *, inputs: list[str], embed_model: str) -> dict[str, Any]:
        return {"model": embed_model, "input": inputs}

    def parse_embedding_response(self, data: dict[str, Any]) -> list[list[float]]:
        return [cast(list[float], d["embedding"]) for d in data["data"]]


logger = logging.getLogger(__name__)

LLM_BACKENDS: dict[str, Callable[[], LLMBackend]] = {
    OpenAILLMBackend.name: OpenAILLMBackend,
}


class HTTPLLMClient:
    """HTTP client for LLM APIs (chat, vision, transcription) and embeddings."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        chat_model: str,
        provider: str = "openai",
        endpoint_overrides: dict[str, str] | None = None,
        timeout: int = 600,
        embed_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        # Ensure base_url ends with "/" so httpx doesn't discard the path
        # component when joining with endpoint paths.
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key or ""
        self.chat_model = chat_model
        self.provider = provider.lower()
        self.backend = self._load_backend(self.provider)
        self.embedding_backend = self._load_embedding_backend(self.provider)
        overrides = endpoint_overrides or {}
        raw_summary_ep = overrides.get("chat") or overrides.get("summary") or self.backend.summary_endpoint
        raw_embedding_ep = (
            overrides.get("embeddings")
            or overrides.get("embedding")
            or overrides.get("embed")
            or self.embedding_backend.embedding_endpoint
        )
        # Strip leading "/" from endpoints so httpx resolves them relative to
        # base_url instead of treating them as absolute paths.
        self.summary_endpoint = raw_summary_ep.lstrip("/")
        self.embedding_endpoint = raw_embedding_ep.lstrip("/")
        self.timeout = timeout
        self.embed_model = embed_model or chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.proxy = _load_proxy()
        self._min_call_gap: float = float(os.getenv("MEMU_LLM_CALL_GAP", "2.0"))
        self._max_retries: int = int(os.getenv("MEMU_LLM_RETRIES", "2"))
        self._last_call_time: float = 0.0
        self._last_payload_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
            "memu_http_llm_last_payload",
            default=None,
        )

    async def _throttle(self) -> None:
        """Enforce minimum gap between API calls to avoid burst limits."""
        if self._min_call_gap <= 0:
            return
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_call_gap:
            await asyncio.sleep(self._min_call_gap - elapsed)
        self._last_call_time = time.monotonic()

    async def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with throttle and retry on transient failures."""
        transient_statuses = {408, 429, 500, 502, 503, 504}
        last_exc: Exception | None = None
        for attempt in range(1 + self._max_retries):
            await self._throttle()
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, proxy=self.proxy) as client:
                    resp = await client.post(endpoint, json=payload, headers=self._headers())
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    if status not in transient_statuses:
                        raise
                last_exc = exc
                if attempt < self._max_retries:
                    wait = (attempt + 1) * 5
                    logger.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s", attempt + 1, 1 + self._max_retries, wait, exc)
                    await asyncio.sleep(wait)
        raise last_exc  # type: ignore[misc]

    async def chat(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generic chat completion."""
        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
        }
        if temp is not None:
            payload["temperature"] = temp
        if mtok is not None:
            payload["max_tokens"] = mtok
        if isinstance(response_format, dict) and response_format:
            payload["response_format"] = response_format

        self._last_payload_var.set(copy.deepcopy(payload))
        data = await self._post_with_retry(self.summary_endpoint, payload)
        logger.debug("HTTP LLM chat response: %s", data)
        return self.backend.parse_summary_response(data), data

    async def summarize(
        self, text: str, max_tokens: int | None = None, system_prompt: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        payload = self.backend.build_summary_payload(
            text=text, system_prompt=system_prompt, chat_model=self.chat_model, max_tokens=max_tokens
        )
        self._last_payload_var.set(copy.deepcopy(payload))
        data = await self._post_with_retry(self.summary_endpoint, payload)
        logger.debug("HTTP LLM summarize response: %s", data)
        return self.backend.parse_summary_response(data), data

    async def vision(
        self,
        prompt: str,
        image_path: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Call Vision API with an image.

        Args:
            prompt: Text prompt to send with the image
            image_path: Path to the image file
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt

        Returns:
            Tuple of (LLM response text, raw response dict)
        """
        # Read and encode image as base64
        image_data = Path(image_path).read_bytes()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # Detect image format
        suffix = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        payload = self.backend.build_vision_payload(
            prompt=prompt,
            base64_image=base64_image,
            mime_type=mime_type,
            system_prompt=system_prompt,
            chat_model=self.chat_model,
            max_tokens=max_tokens,
        )

        self._last_payload_var.set(copy.deepcopy(payload))
        data = await self._post_with_retry(self.summary_endpoint, payload)
        logger.debug("HTTP LLM vision response: %s", data)
        return self.backend.parse_summary_response(data), data

    async def embed(self, inputs: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        """Create text embeddings using the provider-specific embedding API."""
        payload = self.embedding_backend.build_embedding_payload(inputs=inputs, embed_model=self.embed_model)
        self._last_payload_var.set(copy.deepcopy(payload))
        data = await self._post_with_retry(self.embedding_endpoint, payload)
        logger.debug("HTTP embedding response: %s", data)
        return self.embedding_backend.parse_embedding_response(data), data

    def get_last_payload(self) -> dict[str, Any] | None:
        payload = self._last_payload_var.get()
        if not isinstance(payload, dict):
            return None
        return copy.deepcopy(payload)

    async def transcribe(
        self,
        audio_path: str,
        *,
        prompt: str | None = None,
        language: str | None = None,
        response_format: str = "text",
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Transcribe audio file using OpenAI Audio API.

        Args:
            audio_path: Path to the audio file
            prompt: Optional prompt to guide the transcription
            language: Optional language code (e.g., 'en', 'zh')
            response_format: Response format ('text', 'json', 'verbose_json')

        Returns:
            Tuple of (transcribed text, raw response dict or None for text format)
        """
        try:
            raw_response: dict[str, Any] | None = None
            # Prepare multipart form data
            with open(audio_path, "rb") as audio_file:
                files = {"file": (Path(audio_path).name, audio_file, "application/octet-stream")}
                data = {
                    "model": "gpt-4o-mini-transcribe",
                    "response_format": response_format,
                }
                if prompt:
                    data["prompt"] = prompt
                if language:
                    data["language"] = language

                async with httpx.AsyncClient(
                    base_url=self.base_url, timeout=self.timeout * 3, proxy=self.proxy
                ) as client:
                    resp = await client.post(
                        "/v1/audio/transcriptions",
                        files=files,
                        data=data,
                        headers=self._headers(),
                    )
                    resp.raise_for_status()

                    if response_format == "text":
                        result = resp.text
                    else:
                        raw_response = resp.json()
                        result = raw_response.get("text", "")

            logger.debug("HTTP audio transcribe response for %s: %s chars", audio_path, len(result))
        except Exception:
            logger.exception("Audio transcription failed for %s", audio_path)
            raise
        else:
            return result or "", raw_response

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _load_backend(self, provider: str) -> LLMBackend:
        factory = LLM_BACKENDS.get(provider)
        if not factory:
            msg = f"Unsupported LLM provider '{provider}'. Available: {', '.join(LLM_BACKENDS.keys())}"
            raise ValueError(msg)
        return factory()

    def _load_embedding_backend(self, provider: str) -> _EmbeddingBackend:
        backends: dict[str, type[_EmbeddingBackend]] = {
            _OpenAIEmbeddingBackend.name: _OpenAIEmbeddingBackend,
        }
        factory = backends.get(provider)
        if not factory:
            msg = f"Unsupported embedding provider '{provider}'. Available: {', '.join(backends.keys())}"
            raise ValueError(msg)
        return factory()
