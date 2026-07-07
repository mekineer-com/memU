from __future__ import annotations

import re
from typing import Any, cast

from memu.llm.backends.base import LLMBackend

_LEADING_THOUGHT_RE = re.compile(r"\A\s*<thought>.*?</thought>\s*", re.DOTALL)


class OpenAILLMBackend(LLMBackend):
    """Backend for OpenAI-compatible LLM API."""

    name = "openai"
    summary_endpoint = "/chat/completions"

    def parse_summary_response(self, data: dict[str, Any]) -> str:
        text = cast(str, data["choices"][0]["message"]["content"])
        return _LEADING_THOUGHT_RE.sub("", text)

    def build_vision_payload(
        self,
        *,
        prompt: str,
        base64_image: str,
        mime_type: str,
        system_prompt: str | None,
        chat_model: str,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Build payload for OpenAI Vision API."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                    },
                },
            ],
        })

        payload: dict[str, Any] = {
            "model": chat_model,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload
