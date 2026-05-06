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
