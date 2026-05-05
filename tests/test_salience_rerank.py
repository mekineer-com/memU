from datetime import datetime, timezone, timedelta

from memu.database.vector import rerank_by_salience, salience_score


def test_salience_score_recency_decay():
    now = datetime.now(timezone.utc)
    recent = salience_score(now, 0.5)
    old = salience_score(now - timedelta(days=60), 0.5)
    unknown = salience_score(None, 0.5)
    assert recent > old
    assert recent > unknown


def test_salience_score_high_salience_decays_slower():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    high_sal = salience_score(old, 0.9)
    low_sal = salience_score(old, 0.1)
    assert high_sal > low_sal


def test_rerank_by_salience_boosts_recent():
    now = datetime.now(timezone.utc)
    candidates = [
        ("old", 0.9, now - timedelta(days=90), 0.5),
        ("new", 0.85, now, 0.5),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "new"


def test_rerank_by_salience_preserves_order_when_equal():
    candidates = [
        ("a", 0.9, None, 0.5),
        ("b", 0.8, None, 0.5),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "a"
    assert ranked[1][0] == "b"
