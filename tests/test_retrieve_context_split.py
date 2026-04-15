from memu.app.retrieve import RetrieveMixin


def test_split_context_queries_route_keeps_twox_downstream_keeps_onex():
    queries = [
        {"role": "all_categories_summary", "content": {"text": "cats"}},
        {"role": "history_from_second_chat_x", "content": {"text": "history-2x"}},
        {"role": "history_from_chat_x", "content": {"text": "history-1x"}},
        {"role": "memory_cache", "content": {"text": "cache"}},
    ]
    route_ctx, downstream_ctx = RetrieveMixin._split_context_queries(queries)
    assert [q.get("role") for q in route_ctx] == [
        "all_categories_summary",
        "history_from_second_chat_x",
        "memory_cache",
    ]
    assert [q.get("role") for q in downstream_ctx] == [
        "all_categories_summary",
        "history_from_chat_x",
        "memory_cache",
    ]


def test_split_context_queries_no_history_unchanged():
    queries = [
        {"role": "all_categories_summary", "content": {"text": "cats"}},
        {"role": "intentions", "content": {"text": "intentions"}},
    ]
    route_ctx, downstream_ctx = RetrieveMixin._split_context_queries(queries)
    assert route_ctx == queries
    assert downstream_ctx == queries
