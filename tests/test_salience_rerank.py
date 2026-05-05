from datetime import datetime, timezone, timedelta

from memu.database.vector import rerank_by_salience, salience_score


def test_salience_score_recency_decay():
    now = datetime.now(timezone.utc)
    recent = salience_score(now, 0.5)
    old = salience_score(now - timedelta(days=60), 0.5)
    assert recent > old


def test_salience_score_high_salience_decays_slower():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    assert salience_score(old, 0.9) > salience_score(old, 0.1)


def test_salience_score_emotional_intensity_boosts():
    now = datetime.now(timezone.utc)
    without_emotion = salience_score(now, 0.5, 0.0)
    with_emotion = salience_score(now, 0.5, 0.8)
    assert with_emotion > without_emotion


def test_rerank_by_salience_boosts_recent():
    now = datetime.now(timezone.utc)
    candidates = [
        ("old", 0.9, now - timedelta(days=90), 0.5, 0.0),
        ("new", 0.85, now, 0.5, 0.0),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "new"


def test_rerank_by_salience_emotional_tiebreak():
    now = datetime.now(timezone.utc)
    candidates = [
        ("calm", 0.9, now, 0.5, 0.0),
        ("intense", 0.9, now, 0.5, 0.9),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "intense"


def test_rerank_by_salience_preserves_order_when_equal():
    now = datetime.now(timezone.utc)
    candidates = [
        ("a", 0.9, now, 0.5, 0.0),
        ("b", 0.8, now, 0.5, 0.0),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "a"
    assert ranked[1][0] == "b"
