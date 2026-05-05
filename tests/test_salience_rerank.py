from datetime import datetime, timezone

from memu.database.vector import rerank_by_salience, salience_score


def test_salience_score_combines_reinforcement_and_recency():
    score = salience_score(3, None, 0.5)
    # log(3+1) + 0.5 = 1.886, * 0.5 (unknown recency) = 0.943
    assert 0.9 < score < 1.0

    # Higher reinforcement → higher score
    assert salience_score(10, None, 0.5) > salience_score(1, None, 0.5)

    # Higher reflection_salience → higher score
    assert salience_score(1, None, 0.9) > salience_score(1, None, 0.1)


def test_salience_score_recency_decay():
    now = datetime.now(timezone.utc)
    recent = salience_score(1, now, 0.5)
    # None last_reinforced_at gets 0.5 recency factor
    unknown = salience_score(1, None, 0.5)
    assert recent > unknown


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
