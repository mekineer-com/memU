from memu.app.retrieve import RetrieveMixin


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
