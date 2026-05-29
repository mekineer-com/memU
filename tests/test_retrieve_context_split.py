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
        query,
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
        "original_query": "help",
        "context_queries": [],
        "rewrite_angle": 0,
        "mental_health_enabled": False,
    }

    out = await mixin._rag_route_intention(state, step_context=None)

    assert captured["include_mental_health_query"] is False
    assert out["mental_health_query"] is None


@pytest.mark.asyncio
async def test_category_sufficiency_uses_second_step_mental_health_query():
    mixin = RetrieveMixin()
    mixin._get_step_llm_client = lambda _ctx: object()
    captured: dict[str, str] = {}
    state = {
        "needs_retrieval": True,
        "original_query": "original",
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
        query,
        context_queries,
        retrieved_content=None,
        system_prompt=None,
        include_mental_health_query=True,
        llm_client=None,
    ):
        captured["query"] = query
        return False, "second-step-rewrite", "<mental_health_query>second step query</mental_health_query>"

    mixin._decide_if_retrieval_needed = _fake_decide  # type: ignore[method-assign]

    out = await mixin._rag_category_sufficiency(state, step_context=None)

    assert captured["query"] == "original"
    assert out["mental_health_query"] == "second step query"
    assert out["active_query"] == "second-step-rewrite"
    assert out["proceed_to_items"] is False
