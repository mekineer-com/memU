from __future__ import annotations

import json

import httpx
import pytest

from memu.llm.http_client import HTTPLLMClient
from memu.llm.wrapper import LLMClientWrapper, LLMInterceptorRegistry


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
@pytest.mark.parametrize("content", ["hello", "I cannot help with that."])
async def test_chat_assembles_sse_and_preserves_usage(monkeypatch, content) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        body = "\n\n".join(
            [
                "event: message\nid: 1\nretry: 1000",
                'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"reasoning":"ignored"},"finish_reason":null}]}',
                f'data: {json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": None}]})}',
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

    assert text == content
    assert raw["choices"][0]["finish_reason"] == "stop"
    assert raw["usage"]["total_tokens"] == 12
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat", "vision"])
@pytest.mark.parametrize("finish_reason", ["length", "stop", "tool_calls", "content_filter", "refusal"])
async def test_provider_failures_are_reported_before_success_without_retry(
    monkeypatch, tmp_path, caplog, method, finish_reason
) -> None:
    calls = 0
    events = []
    content = '{"items": []}'

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        choice = {"finish_reason": "stop" if finish_reason == "refusal" else finish_reason}
        if method == "chat":
            choice["delta"] = {"content": content}
            frames = []
            if finish_reason == "refusal":
                frames.append({"choices": [{"delta": {"refusal": "No "}, "finish_reason": None}]})
                choice["delta"]["refusal"] = "thanks."
            frames.append({"choices": [choice]})
            return httpx.Response(
                200, text="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
            )
        choice["message"] = {"content": content}
        if finish_reason == "refusal":
            choice["message"] = {"content": None, "refusal": "No thanks."}
        return httpx.Response(200, json={"choices": [choice]})

    async def after(*_args):
        events.append("success")

    async def on_error(*_args):
        events.append("error")

    registry = LLMInterceptorRegistry()
    registry.register_after(after)
    registry.register_on_error(on_error)
    client = LLMClientWrapper(_client(monkeypatch, handler), registry=registry)
    args = ["prompt"]
    if method == "vision":
        image = tmp_path / "image.png"
        image.write_bytes(b"test image")
        args.append(str(image))
    caplog.set_level("DEBUG", logger="memu.llm.http_client")
    errors = {
        "length": "truncated at the output token limit",
        "content_filter": "blocked by provider content filter",
        "refusal": "refused by provider",
    }
    if finish_reason in errors:
        with pytest.raises(RuntimeError, match=errors[finish_reason]):
            await getattr(client, method)(*args)
        assert events == ["error"]
        if method == "chat" and finish_reason == "refusal":
            assert "'refusal': 'No thanks.'" in caplog.text
    else:
        assert await getattr(client, method)(*args) == content
        assert events == ["success"]
    assert calls == 1


class _BrokenStream(httpx.AsyncByteStream):
    def __init__(self, error_type) -> None:
        self.error_type = error_type

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise self.error_type("disconnected")


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.RemoteProtocolError, httpx.ReadError])
async def test_chat_does_not_retry_partial_stream(monkeypatch, error_type) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=_BrokenStream(error_type))

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


@pytest.mark.asyncio
async def test_chat_preserves_http_error_detail_without_retry(monkeypatch) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(402, json={"error": "insufficient balance"})

    client = _client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="402.*insufficient balance"):
        await client.chat("prompt")
    assert calls == 1
