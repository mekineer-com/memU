import logging

import pytest

from memu.database.vector import cosine_topk


def test_cosine_topk_logs_error_on_dim_mismatch(caplog):
    corpus = [
        ("good", [1.0, 0.0]),
        ("bad_dim", [1.0, 0.0, 0.0]),  # wrong dim
    ]
    with caplog.at_level(logging.ERROR, logger="memu.database.vector"):
        results = cosine_topk([1.0, 0.0], corpus, k=5)

    assert len(results) == 1
    assert results[0][0] == "good"
    assert any("skipped" in r.message and "1" in r.message for r in caplog.records)
