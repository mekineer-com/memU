from datetime import datetime, timezone, timedelta

from memu.database.vector import rerank_by_salience, salience_score


def test_salience_score_recency_decay():
    now = datetime.now(timezone.utc)
    _, recent_recency = salience_score(now, 0.5)
    _, old_recency = salience_score(now - timedelta(days=60), 0.5)
    assert recent_recency > old_recency


def test_salience_score_high_salience_decays_slower():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    _, recency_high = salience_score(old, 0.9)
    _, recency_low = salience_score(old, 0.1)
    assert recency_high > recency_low


def test_salience_score_emotional_intensity_boosts():
    now = datetime.now(timezone.utc)
    imp_without, _ = salience_score(now, 0.5, 0.0)
    imp_with, _ = salience_score(now, 0.5, 0.8)
    assert imp_with > imp_without


def test_salience_score_importance_capped_at_one():
    now = datetime.now(timezone.utc)
    imp, _ = salience_score(now, 0.9, 1.0)
    assert imp == 1.0


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


def test_rerank_none_salience_uses_similarity_as_importance():
    now = datetime.now(timezone.utc)
    candidates = [
        ("extracted", 0.7, now, 0.8, 0.0),
        ("non_extracted", 0.9, now, None, None),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "non_extracted"
