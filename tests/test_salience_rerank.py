from memu.database.vector import rerank_by_salience, salience_score


def test_salience_score_returns_reflection_salience():
    assert salience_score(0.8) == 0.8
    assert salience_score(0.0) == 0.0


def test_rerank_by_salience_boosts_high_salience():
    candidates = [
        ("low_sal", 0.9, 1, None, 0.1),
        ("high_sal", 0.85, 1, None, 0.9),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "high_sal"
    assert ranked[1][0] == "low_sal"


def test_rerank_by_salience_preserves_order_when_salience_equal():
    candidates = [
        ("a", 0.9, 1, None, 0.5),
        ("b", 0.8, 1, None, 0.5),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "a"
    assert ranked[1][0] == "b"
