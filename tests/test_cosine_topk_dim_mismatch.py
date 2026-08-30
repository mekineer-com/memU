import pytest

from memu.database.vector import cosine_topk


def test_cosine_topk_fails_on_dim_mismatch():
    corpus = [
        ("good", [1.0, 0.0]),
        ("bad_dim", [1.0, 0.0, 0.0]),  # wrong dim
    ]
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_topk([1.0, 0.0], corpus, k=5)
