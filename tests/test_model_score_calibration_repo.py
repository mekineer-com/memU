from __future__ import annotations

from pydantic import BaseModel

from memu.app.service import MemoryService


class _Scope(BaseModel):
    user_id: str


def _service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": _Scope},
    )


def _create_item(*, service: MemoryService, idx: int, model: str, salience: float, intensity: float, confidence: float) -> None:
    store = service._get_database()
    store.memory_item_repo.create_item(
        memory_type="profile",
        summary=f"item-{idx}",
        embedding=[0.1, 0.2, 0.3],
        user_data={"user_id": "u1"},
        confidence=confidence,
        reflection_salience=salience,
        emotional_intensity=intensity,
        extra={"model": model},
    )


def test_model_score_calibration_refresh_uses_thresholds() -> None:
    service = _service()
    store = service._get_database()

    for i in range(3):
        _create_item(
            service=service,
            idx=i,
            model="model-a",
            salience=0.2 + (i * 0.1),
            intensity=0.3 + (i * 0.1),
            confidence=0.4 + (i * 0.1),
        )

    samples = store.memory_item_repo.refresh_model_score_calibration(model="model-a")
    assert samples["confidence"] == 3
    assert samples["reflection_salience"] == 3
    assert samples["emotional_intensity"] == 3

    with store._sessions.session() as session:
        rows = session.connection().exec_driver_sql(
            "SELECT field, sample_size FROM model_score_calibration WHERE model = ? AND version = 1",
            ("model-a",),
        ).fetchall()
    assert sorted((str(field), int(size)) for field, size in rows) == [
        ("confidence", 3),
        ("emotional_intensity", 3),
        ("reflection_salience", 3),
    ]

    for i in range(3, 50):
        _create_item(
            service=service,
            idx=i,
            model="model-a",
            salience=0.4,
            intensity=0.5,
            confidence=0.6,
        )

    samples = store.memory_item_repo.refresh_model_score_calibration(model="model-a")
    assert samples["confidence"] == 50
    assert samples["reflection_salience"] == 50
    assert samples["emotional_intensity"] == 50

    with store._sessions.session() as session:
        rows = session.connection().exec_driver_sql(
            "SELECT field, sample_size FROM model_score_calibration WHERE model = ? AND version = 1",
            ("model-a",),
        ).fetchall()
    assert sorted((str(field), int(size)) for field, size in rows) == [
        ("confidence", 50),
        ("emotional_intensity", 50),
        ("reflection_salience", 50),
    ]
