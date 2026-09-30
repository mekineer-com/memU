from __future__ import annotations

import json

import httpx
import pytest

from memu.llm.http_client import HTTPLLMClient


def _client(monkeypatch: pytest.MonkeyPatch, handler) -> HTTPLLMClient:
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        return async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    client = HTTPLLMClient(
        base_url="https://example.test/v1",
        api_key="test",
        chat_model="test-model",
        max_tokens=128_000,
    )
    client._min_call_gap = 0
    return client


@pytest.mark.asyncio
async def test_chat_assembles_sse_and_preserves_usage(monkeypatch) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"reasoning":"ignored"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},'
                '"x_nanogpt_pricing":{"amount":0.1}}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, text=body)

    client = _client(monkeypatch, handler)
    text, raw = await client.chat(
        "prompt", response_format={"type": "json_object"}
    )

    assert text == "hello"
    assert raw["choices"][0]["finish_reason"] == "stop"
    assert raw["usage"]["total_tokens"] == 12
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["response_format"] == {"type": "json_object"}


class _BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.RemoteProtocolError("disconnected")


@pytest.mark.asyncio
async def test_chat_does_not_retry_partial_stream(monkeypatch) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=_BrokenStream())

    client = _client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match=r"frames=1, bytes="):
        await client.chat("prompt")
    assert calls == 1


@pytest.mark.asyncio
async def test_chat_does_not_retry_pre_header_disconnect(monkeypatch) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("disconnected")

    client = _client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match=r"frames=0, bytes=0"):
        await client.chat("prompt")
    assert calls == 1


@pytest.mark.asyncio
async def test_chat_rejects_error_and_incomplete_streams(monkeypatch) -> None:
    responses = iter(
        [
            'data: {"error":{"message":"upstream failed"}}\n\ndata: [DONE]\n\n',
            'data: {"choices":[{"delta":{"content":"partial"},'
            '"finish_reason":null}]}\n\n',
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=next(responses))

    client = _client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="LLM stream error"):
        await client.chat("prompt")
    with pytest.raises(RuntimeError, match="ended incomplete"):
        await client.chat("prompt")
