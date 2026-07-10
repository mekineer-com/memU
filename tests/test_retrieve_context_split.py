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
    assert "My Working Thoughts:\ntrack bridge duplication issue" in rendered
    assert "My Intentions:\n- relax: Relax (reminder to breathe)" in rendered
    assert rendered.index("## My SillyTavern Conversations:") < rendered.index("## My WhatsApp Conversations:")
    assert rendered.index("## My WhatsApp Conversations:") < rendered.index("My Working Thoughts:")
    assert rendered.index("My Working Thoughts:") < rendered.index("My Intentions:")


def test_format_query_context_rejects_legacy_string_entries():
    mixin = RetrieveMixin()
    with pytest.raises(TypeError, match="INVALID_CONTEXT_QUERY"):
        mixin._format_query_context([{"role": "history", "content": {"text": "ok"}}, "legacy"])  # type: ignore[list-item]


def test_format_category_content_uses_summary_text_without_wrapper():
    mixin = RetrieveMixin()
    store = SimpleNamespace(
        memory_category_repo=SimpleNamespace(categories={}),
    )
    out = mixin._format_category_content(
        [("relationships", 0.7)],
        {"relationships": "# Relationships\nClose bonds matter."},
        store,
        categories={"relationships": SimpleNamespace(name="Relationships", summary="")},
    )

    assert out == "# Relationships\nClose bonds matter."
    assert "Category:" not in out
    assert "Summary:" not in out
    assert "Score:" not in out


def test_system_prompt_excludes_mental_health_block_when_disabled():
    prompt = system_prompt_for_angle(0, include_mental_health_query=False)
    assert "<mental_health_query>" not in prompt
    assert "also write a mental_health_query" not in prompt


def test_system_prompt_forbids_answering_user_in_route_step():
    prompt = system_prompt_for_angle(0, include_mental_health_query=True)
    assert "This turn is for you to give a search query" in prompt
    assert "respond in the next turn (not this turn)" in prompt
    assert "Do not add any prose, dialogue, markdown, or extra sections" in prompt
    assert "<active_query>" in prompt
    assert "<rewritten_query>" not in prompt


@pytest.mark.asyncio
async def test_decide_if_retrieval_needed_omits_empty_retrieved_placeholder():
    mixin = RetrieveMixin()
    mixin._escape_prompt_value = lambda text: text
    captured: dict[str, str] = {}

    class Client:
        async def chat(self, prompt, **_kwargs):  # type: ignore[no-untyped-def]
            captured["prompt"] = prompt
            return "<decision>NO_RETRIEVE</decision>"

    await mixin._decide_if_retrieval_needed(
        "new message",
        [
            {
                "role": "identity_context",
                "content": {"text": "Today is Thursday, June 18, 2026 17:34 -05.\n\nYou are Siri,"},
            }
        ],
        retrieved_content=None,
        llm_client=Client(),
    )

    assert not captured["prompt"].lstrip().startswith("# Input")
    assert "My Soul:" not in captured["prompt"]
    assert captured["prompt"].lstrip().startswith("Today is Thursday, June 18, 2026 17:34 -05.\n\nYou are Siri,")
    assert "Soul context:" not in captured["prompt"]
    assert "Retrieved so far:" not in captured["prompt"]
    assert "No content retrieved yet." not in captured["prompt"]
    assert "<mental_health_query>" not in captured["prompt"]


@pytest.mark.asyncio
async def test_decide_if_retrieval_needed_preserves_actual_retrieved_content():
    mixin = RetrieveMixin()
    mixin._escape_prompt_value = lambda text: text
    captured: dict[str, str] = {}

    class Client:
        async def chat(self, prompt, **_kwargs):  # type: ignore[no-untyped-def]
            captured["prompt"] = prompt
            return "<decision>NO_RETRIEVE</decision>"

    await mixin._decide_if_retrieval_needed(
        "new message",
        [],
        retrieved_content="[profile] Marcos likes careful prompts.",
        llm_client=Client(),
    )

    assert "Retrieved so far:\n\n[profile] Marcos likes careful prompts." in captured["prompt"]


def test_extract_decision_raises_on_empty_llm_response():
    mixin = RetrieveMixin()
    with pytest.raises(ValueError, match="sufficiency check returned empty response"):
        mixin._extract_decision("")


@pytest.mark.asyncio
async def test_route_intention_disables_mental_health_query_extraction():
    mixin = RetrieveMixin()
    captured: dict[str, object] = {}

    mixin._select_chat_client = lambda _ctx: object()

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
async def test_route_intention_force_retrieve_skips_first_llm_without_mental_health_query():
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
    assert out["active_query"] == ""
    assert out["mental_health_query"] is None


@pytest.mark.asyncio
async def test_force_retrieve_skips_category_summary_search():
    mixin = RetrieveMixin()

    class EmbedClient:
        async def embed(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("force retrieve should not embed message for category search")

    mixin._select_embedding_client = lambda _ctx: EmbedClient()
    state = {
        "needs_retrieval": True,
        "active_query": "",
        "ctx": object(),
        "store": object(),
        "where": {},
        "force_retrieve": True,
    }

    out = await mixin._rag_route_category(state, step_context=None)

    assert out["category_hits"] == []
    assert out["category_summary_lookup"] == {}
    assert out["query_vector"] is None


@pytest.mark.asyncio
async def test_category_sufficiency_uses_second_step_mental_health_query():
    mixin = RetrieveMixin()
    mixin._select_chat_client = lambda _ctx: object()
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
async def test_force_retrieve_uses_sufficiency_ai_query_for_items():
    mixin = RetrieveMixin()
    mixin._select_chat_client = lambda _ctx: object()
    captured: dict[str, object] = {}

    class EmbedClient:
        async def embed(self, values):  # type: ignore[no-untyped-def]
            assert values == ["ai-written item query"]
            return [[0.1, 0.2]]

    mixin._select_embedding_client = lambda _ctx: EmbedClient()
    state = {
        "needs_retrieval": True,
        "new_message": "raw current message",
        "active_query": "",
        "context_queries": [],
        "category_pool": {},
        "category_hits": [],
        "store": object(),
        "where": {},
        "mental_health_enabled": True,
        "force_retrieve": True,
    }

    async def _fake_decide(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return True, "ai-written item query", (
            "<active_query>ai-written item query</active_query>"
            "<mental_health_query>sleep boundaries</mental_health_query>"
        )

    mixin._decide_if_retrieval_needed = _fake_decide  # type: ignore[method-assign]

    out = await mixin._rag_category_sufficiency(state, step_context=None)

    assert captured["require_decision"] is False
    assert "<decision>" not in str(captured["system_prompt"])
    assert "Always write an active_query" in str(captured["system_prompt"])
    assert out["active_query"] == "ai-written item query"
    assert out["mental_health_query"] == "sleep boundaries"
    assert out["proceed_to_items"] is True
    assert out["query_vector"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_category_sufficiency_preserves_mental_health_when_second_step_omits_it():
    mixin = RetrieveMixin()
    mixin._select_chat_client = lambda _ctx: object()
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

    async def _fake_decide(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return False, "second-step-rewrite", "<decision>NO_RETRIEVE</decision><active_query></active_query>"

    mixin._decide_if_retrieval_needed = _fake_decide  # type: ignore[method-assign]

    out = await mixin._rag_category_sufficiency(state, step_context=None)

    assert out["mental_health_query"] == "first-step-query"


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


@pytest.mark.asyncio
async def test_decide_if_retrieval_needed_accepts_mistyped_active_query_close_tag():
    mixin = RetrieveMixin()
    mixin._escape_prompt_value = lambda text: text

    class Client:
        async def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return (
                "<decision>RETRIEVE</decision>"
                "<active_query>Annie Gottlieb relationship history</active_health_query>"
                "<mental_health_query></mental_health_query>"
            )

    needs_retrieval, active_query, _raw = await mixin._decide_if_retrieval_needed(
        "raw current message",
        [],
        llm_client=Client(),
    )

    assert needs_retrieval is True
    assert active_query == "Annie Gottlieb relationship history"


@pytest.mark.asyncio
async def test_query_only_retrieve_requires_active_query_without_decision():
    mixin = RetrieveMixin()
    mixin._escape_prompt_value = lambda text: text

    class Client:
        async def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "<active_query>Marcos Mexico travel re-entry</active_query>"

    needs_retrieval, active_query, raw = await mixin._decide_if_retrieval_needed(
        "raw current message",
        [],
        llm_client=Client(),
        require_decision=False,
    )

    assert needs_retrieval is True
    assert active_query == "Marcos Mexico travel re-entry"
    assert "<decision>" not in raw


@pytest.mark.asyncio
async def test_query_only_retrieve_rejects_missing_active_query():
    mixin = RetrieveMixin()
    mixin._escape_prompt_value = lambda text: text

    class Client:
        async def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "<mental_health_query></mental_health_query>"

    with pytest.raises(ValueError, match="missing active_query"):
        await mixin._decide_if_retrieval_needed(
            "raw current message",
            [],
            llm_client=Client(),
            require_decision=False,
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
