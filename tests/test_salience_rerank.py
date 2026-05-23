from datetime import datetime, timezone, timedelta

from memu.database.vector import normalize_score_with_percentiles, rerank_by_salience, salience_score


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
        ("old", 0.9, now - timedelta(days=90), 0.5, 0.0, None),
        ("new", 0.85, now, 0.5, 0.0, None),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "new"


def test_rerank_by_salience_emotional_tiebreak():
    now = datetime.now(timezone.utc)
    candidates = [
        ("calm", 0.9, now, 0.5, 0.0, None),
        ("intense", 0.9, now, 0.5, 0.9, None),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "intense"


def test_rerank_by_salience_preserves_order_when_equal():
    now = datetime.now(timezone.utc)
    candidates = [
        ("a", 0.9, now, 0.5, 0.0, None),
        ("b", 0.8, now, 0.5, 0.0, None),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "a"
    assert ranked[1][0] == "b"


def test_rerank_none_salience_uses_similarity_as_importance():
    now = datetime.now(timezone.utc)
    candidates = [
        ("extracted", 0.7, now, 0.8, 0.0, "claude-opus-4-6"),
        ("non_extracted", 0.9, now, None, None, None),
    ]
    ranked = rerank_by_salience(candidates)
    assert ranked[0][0] == "non_extracted"


def test_normalize_score_with_percentiles_linear_piecewise():
    params = {"p10": 0.2, "p25": 0.3, "p50": 0.5, "p75": 0.7, "p90": 0.8}
    assert normalize_score_with_percentiles(0.1, params) == 0.1
    assert normalize_score_with_percentiles(0.9, params) == 0.9
    mid = normalize_score_with_percentiles(0.6, params)
    assert 0.5 < mid < 0.75


def test_rerank_by_salience_applies_model_calibration():
    now = datetime.now(timezone.utc)
    candidates = [
        ("raw_high", 0.6, now, 0.9, 0.1, "model-a"),
        ("raw_mid", 0.6, now, 0.6, 0.1, "model-b"),
    ]
    calibrations = {
        ("model-a", "reflection_salience"): {"p10": 0.9, "p25": 0.93, "p50": 0.95, "p75": 0.97, "p90": 0.99},
        ("model-b", "reflection_salience"): {"p10": 0.2, "p25": 0.3, "p50": 0.5, "p75": 0.6, "p90": 0.7},
    }
    ranked = rerank_by_salience(candidates, calibrations=calibrations)
    assert ranked[0][0] == "raw_mid"
