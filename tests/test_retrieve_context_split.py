from types import SimpleNamespace

import pytest

from memu.app.retrieve import RetrieveMixin
from memu.prompts.retrieve.pre_retrieval_decision import system_prompt_for_angle


def test_format_query_context_uses_markdown_sections_and_chat_first_order():
    mixin = RetrieveMixin()
    queries = [
        {
            "role": "identity_context",
            "content": {"text": "Today is Friday.\nI am Echo."},
        },
        {
            "role": "memory_cache",
            "content": {"text": "track bridge duplication issue"},
        },
        {
            "role": "intentions",
            "content": {"text": "- relax: Relax (reminder to breathe)"},
        },
        {
            "role": "cross_conversation",
            "content": {"text": "## My WhatsApp Conversations:\n\n[dm][Marcos]\n--- Wednesday ---\n[Marcos]: O hai!"},
        },
        {
            "role": "history",
            "content": {"text": "## My SillyTavern Conversations:\n\n[10] [Marcos] hello"},
        },
    ]

    rendered = mixin._format_query_context(queries)

    assert "Today is Friday.\nI am Echo." in rendered
    assert "## My SillyTavern Conversations:\n\n[10] [Marcos] hello" in rendered
    assert "## My WhatsApp Conversations:" in rendered
    assert "- [cross_conversation]:" not in rendered
    assert "My working thoughts:\ntrack bridge duplication issue" in rendered
    assert "My intentions:\n- relax: Relax (reminder to breathe)" in rendered
    assert rendered.index("## My SillyTavern Conversations:") < rendered.index("## My WhatsApp Conversations:")
    assert rendered.index("## My WhatsApp Conversations:") < rendered.index("My working thoughts:")
    assert rendered.index("My working thoughts:") < rendered.index("My intentions:")


def test_format_query_context_rejects_legacy_string_entries():
    mixin = RetrieveMixin()
    with pytest.raises(TypeError, match="INVALID_CONTEXT_QUERY"):
        mixin._format_query_context([{"role": "history", "content": {"text": "ok"}}, "legacy"])  # type: ignore[list-item]


def test_system_prompt_excludes_mental_health_block_when_disabled():
    prompt = system_prompt_for_angle(0, include_mental_health_query=False)
    assert "<mental_health_query>" not in prompt
    assert "also write a mental_health_query" not in prompt


def test_system_prompt_forbids_answering_user_in_route_step():
    prompt = system_prompt_for_angle(0, include_mental_health_query=True)
    assert "You are not speaking to your human in this step" in prompt
    assert "Do not answer the new message" in prompt
    assert "<active_query>" in prompt
    assert "<rewritten_query>" not in prompt


def test_extract_decision_raises_on_empty_llm_response():
    mixin = RetrieveMixin()
    with pytest.raises(ValueError, match="sufficiency check returned empty response"):
        mixin._extract_decision("")


@pytest.mark.asyncio
async def test_route_intention_disables_mental_health_query_extraction():
    mixin = RetrieveMixin()
    captured: dict[str, object] = {}

    mixin._get_step_llm_client = lambda _ctx: object()

    async def _fake_decide(  # type: ignore[no-untyped-def]
        new_message,
        context_queries,
        retrieved_content=None,
        system_prompt=None,
        include_mental_health_query=True,
        llm_client=None,
    ):
        captured["include_mental_health_query"] = include_mental_health_query
        return True, "rewritten", "<mental_health_query>sleep hygiene</mental_health_query>"

    mixin._decide_if_retrieval_needed = _fake_decide  # type: ignore[method-assign]
    state = {
        "new_message": "help",
        "context_queries": [],
        "rewrite_angle": 0,
        "mental_health_enabled": False,
    }

    out = await mixin._rag_route_intention(state, step_context=None)

    assert captured["include_mental_health_query"] is False
    assert out["mental_health_query"] is None


@pytest.mark.asyncio
async def test_route_intention_force_retrieve_skips_llm_and_uses_new_message_as_search_fallback():
    mixin = RetrieveMixin()

    async def _should_not_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("route_intention LLM path should be skipped when force_retrieve=true")

    mixin._decide_if_retrieval_needed = _should_not_run  # type: ignore[method-assign]
    state = {
        "new_message": "topic statement text",
        "context_queries": [],
        "rewrite_angle": 0,
        "mental_health_enabled": True,
        "force_retrieve": True,
    }

    out = await mixin._rag_route_intention(state, step_context=None)

    assert out["needs_retrieval"] is True
    assert out["active_query"] == "topic statement text"
    assert out["mental_health_query"] == "topic statement text"


@pytest.mark.asyncio
async def test_category_sufficiency_uses_second_step_mental_health_query():
    mixin = RetrieveMixin()
    mixin._get_step_llm_client = lambda _ctx: object()
    captured: dict[str, str] = {}
    state = {
        "needs_retrieval": True,
        "new_message": "original",
        "active_query": "original",
        "context_queries": [],
        "category_pool": {"c1": object()},
        "category_hits": [],
        "store": object(),
        "where": {},
        "mental_health_enabled": True,
        "mental_health_query": "first-step-query",
    }

    async def _fake_decide(  # type: ignore[no-untyped-def]
        new_message,
        context_queries,
        retrieved_content=None,
        system_prompt=None,
        include_mental_health_query=True,
        llm_client=None,
    ):
        captured["new_message"] = new_message
        return False, "second-step-rewrite", "<mental_health_query>second step query</mental_health_query>"

    mixin._decide_if_retrieval_needed = _fake_decide  # type: ignore[method-assign]

    out = await mixin._rag_category_sufficiency(state, step_context=None)

    assert captured["new_message"] == "original"
    assert out["mental_health_query"] == "second step query"
    assert out["active_query"] == "second-step-rewrite"
    assert out["proceed_to_items"] is False


@pytest.mark.asyncio
async def test_decide_if_retrieval_needed_requires_active_query_when_retrieving():
    mixin = RetrieveMixin()
    mixin._get_llm_client = lambda: object()
    mixin._escape_prompt_value = lambda text: text

    class Client:
        async def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "<decision>RETRIEVE</decision>"

    with pytest.raises(ValueError, match="missing active_query"):
        await mixin._decide_if_retrieval_needed(
            "new message",
            [],
            llm_client=Client(),
        )


def test_retrieve_context_excludes_message_to_self_audit_memories():
    mixin = RetrieveMixin()
    mixin._model_dump_without_embeddings = lambda obj: dict(obj)
    mixin._find_superseded_at = lambda *_args, **_kwargs: None
    store = SimpleNamespace(
        resource_repo=SimpleNamespace(list_resources=lambda _where: {}),
    )
    state = {
        "needs_retrieval": True,
        "new_message": "hello",
        "item_hits": [("keep", 0.9), ("audit", 0.8)],
        "item_pool": {
            "keep": {
                "id": "keep",
                "memory_type": "profile",
                "summary": "Marcos likes continuity.",
                "extra": {},
            },
            "audit": {
                "id": "audit",
                "memory_type": "subconscious",
                "summary": "Notice the quiet signal before answering.",
                "extra": {"apimw_message_to_self": True},
            },
        },
        "category_hits": [],
        "category_pool": {"unused": object()},
        "resource_hits": [],
        "resource_pool": {},
        "store": store,
        "where": {},
    }

    out = mixin._rag_build_context(state, None)

    assert [item["id"] for item in out["response"]["items"]] == ["keep"]
