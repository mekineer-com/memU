import pytest

from memu.app.retrieve import RetrieveMixin
from memu.prompts.retrieve.pre_retrieval_decision import system_prompt_for_angle


def test_split_context_queries_returns_identical_copies():
    queries = [
        {"role": "all_categories_summary", "content": {"text": "cats"}},
        {"role": "history", "content": {"text": "conversation history"}},
        {"role": "memory_cache", "content": {"text": "cache"}},
    ]
    route_ctx, downstream_ctx = RetrieveMixin._split_context_queries(queries)
    assert route_ctx == queries
    assert downstream_ctx == queries


def test_split_context_queries_no_history_unchanged():
    queries = [
        {"role": "all_categories_summary", "content": {"text": "cats"}},
        {"role": "intentions", "content": {"text": "intentions"}},
    ]
    route_ctx, downstream_ctx = RetrieveMixin._split_context_queries(queries)
    assert route_ctx == queries
    assert downstream_ctx == queries


def test_format_query_context_uses_separated_multiline_blocks():
    mixin = RetrieveMixin()
    queries = [
        {
            "role": "identity_context",
            "content": {"text": "Today is Friday.\nI am Echo."},
        },
        {
            "role": "cross_conversation",
            "content": {"text": "--- Wednesday ---\n[whatsapp:dm] [user]: O hai!"},
        },
        {
            "role": "history",
            "content": {"text": "[10] [Marcos] hello"},
        },
    ]

    rendered = mixin._format_query_context(queries)

    assert "Today is Friday.\nI am Echo." in rendered
    assert "- [cross_conversation]:\n--- Wednesday ---\n[whatsapp:dm] [user]: O hai!" in rendered
    assert "\n\n- [cross_conversation]:\n" in rendered
    assert "\n\n- [history]:\n[10] [Marcos] hello" in rendered


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
    state = {
        "needs_retrieval": True,
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
        return False, "second-step-rewrite", "<mental_health_query>second step query</mental_health_query>"

    mixin._decide_if_retrieval_needed = _fake_decide  # type: ignore[method-assign]

    out = await mixin._rag_category_sufficiency(state, step_context=None)

    assert out["mental_health_query"] == "second step query"
    assert out["active_query"] == "second-step-rewrite"
    assert out["proceed_to_items"] is False
