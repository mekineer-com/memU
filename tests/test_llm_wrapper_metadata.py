from __future__ import annotations

import pytest

from memu.llm.wrapper import LLMCallMetadata, LLMClientWrapper, LLMInterceptorRegistry


class _FakeClient:
    chat_model = "fake-chat"
    embed_model = "fake-embed"

    def __init__(self):
        self.calls = []

    async def chat(self, prompt: str, **_kwargs: object) -> str:
        self.calls.append((prompt, _kwargs))
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


@pytest.mark.asyncio
async def test_native_budget_counts_system_and_survives_metadata_without_interceptor_swallowing():
    client = _FakeClient()
    wrapper = LLMClientWrapper(client, registry=LLMInterceptorRegistry(), input_budget=100)
    assert await wrapper.chat("small") == "echo:small"
    with pytest.raises(ValueError, match="input budget"):
        await wrapper.with_metadata(operation="memorize", step_id="background_summary").chat(
            "small", system_prompt="x" * 1000,
        )
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_native_budget_reserves_actual_output_override_and_default_remains_optional():
    client = _FakeClient()
    wrapper = LLMClientWrapper(client, registry=LLMInterceptorRegistry(), input_budget=100, context_window_tokens=500)
    assert await wrapper.chat("x" * 500, max_tokens=50)
    for override in (450, 500, -1, True):
        with pytest.raises(ValueError):
            await wrapper.chat("x" * 500, max_tokens=override)
    assert len(client.calls) == 1
    assert await LLMClientWrapper(client, registry=LLMInterceptorRegistry()).chat("x" * 1000)


@pytest.mark.asyncio
async def test_native_budget_has_no_artificial_100k_limit():
    client = _FakeClient()
    wrapper = LLMClientWrapper(client, registry=LLMInterceptorRegistry(), input_budget=250_000)
    assert await wrapper.chat("word " * 100_001)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_generic_service_policy_is_optional_and_embedding_is_not_chat_budgeted():
    from memu.app.service import MemoryService
    service = MemoryService(database_config={"metadata_store": {"dsn": "sqlite:///:memory:"}})
    client = _FakeClient()
    try:
        assert await service._wrap_llm_client(client, step_context={"operation": "memorize"}).chat("x" * 1000)
        service.memorize_config.enforce_input_budget = True
        service._llm_clients["embedding"] = client
        assert service._select_embedding_client({"operation": "memorize"})._input_budget is None
    finally:
        service.database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", ["rollup", "groups"])
async def test_background_summaries_reject_full_request_before_provider_call(summary):
    from memu.app.service import MemoryService
    service = MemoryService(database_config={"metadata_store": {"dsn": "sqlite:///:memory:"}},
                            memorize_config={"enforce_input_budget": True})
    service.llm_profiles.profiles["default"].context_window_tokens = 100
    client = _FakeClient()
    service._llm_clients["default"] = client
    messages = [{"role": "user", "content": "x" * 1000, "_message_index": 0}]
    try:
        with pytest.raises(ValueError, match="input budget"):
            if summary == "rollup":
                await service.summarize_background_chat_rollup(prior_summary=None, messages=messages)
            else:
                await service._summarize_background_groups_batched(grouped_messages={"test": messages}, group_order=["test"])
        assert client.calls == []
    finally:
        service.database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cli", [False, True])
@pytest.mark.parametrize("operation, step", [("memorize", "router"), ("memorize_segment", "extract"),
                                             ("memorize_segments_batch", "extract"), ("dossier", "dynamic_review")])
async def test_service_budget_uses_effective_model_and_leaves_retrieval_unrestricted(cli, operation, step):
    from memu.app.service import MemoryService
    service = MemoryService(database_config={"metadata_store": {"dsn": "sqlite:///:memory:"}},
                            memorize_config={"enforce_input_budget": True}, claude_code=cli,
                            claude_code_context_window_tokens=100)
    service.llm_profiles.profiles["default"].context_window_tokens = 100_000
    service.llm_profiles.profiles["small"] = service.llm_profiles.profiles["default"].model_copy(
        update={"context_window_tokens": 100, "chat_model": "fictional-small"},
    )
    client = _FakeClient()
    try:
        guarded = service._wrap_llm_client(client, profile="small", step_context={"operation": operation, "step_id": step})
        with pytest.raises(ValueError, match="input budget"):
            await guarded.chat("x" * 1000)
        assert client.calls == []
        unguarded = service._wrap_llm_client(client, profile="small", step_context={"operation": "retrieve"})
        assert await unguarded.chat("x" * 1000)
    finally:
        service.database.close()
