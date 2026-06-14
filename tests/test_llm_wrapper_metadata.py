from __future__ import annotations

import pytest

from memu.llm.wrapper import LLMCallMetadata, LLMClientWrapper, LLMInterceptorRegistry


class _FakeClient:
    chat_model = "fake-chat"
    embed_model = "fake-embed"

    async def chat(self, prompt: str, **_kwargs: object) -> str:
        return f"echo:{prompt}"


@pytest.mark.asyncio
async def test_with_metadata_relabels_child_wrapper_without_replacing_client() -> None:
    seen: list[tuple[str | None, str | None]] = []
    registry = LLMInterceptorRegistry()

    async def capture(ctx, _request_view, _response_view, _usage) -> None:
        seen.append((ctx.operation, ctx.step_id))

    registry.register_after(capture)
    base = LLMClientWrapper(
        _FakeClient(),
        registry=registry,
        metadata=LLMCallMetadata(operation="memorize", step_id="extract_items_batch"),
    )

    child = base.with_metadata(step_id="router")

    assert await child.chat("hi") == "echo:hi"
    assert seen == [("memorize", "router")]

