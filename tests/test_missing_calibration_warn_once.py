from __future__ import annotations

import logging

from memu.database.sqlite.repositories import memory_item_repo as repo_mod
from memu.database.sqlite.repositories.memory_item_repo import SQLiteMemoryItemRepo


def test_missing_calibration_warning_deduped_per_model_field(caplog) -> None:
    repo_mod._MISSING_CALIBRATION_WARNED.clear()
    caplog.set_level(logging.WARNING)

    SQLiteMemoryItemRepo._warn_missing_calibration_once("model-a", "reflection_salience")
    SQLiteMemoryItemRepo._warn_missing_calibration_once("model-a", "reflection_salience")
    SQLiteMemoryItemRepo._warn_missing_calibration_once("model-a", "emotional_intensity")

    warnings = [
        rec.message for rec in caplog.records if "No score calibration found" in rec.message
    ]
    assert warnings == [
        "No score calibration found for model=model-a field=reflection_salience; using identity mapping",
        "No score calibration found for model=model-a field=emotional_intensity; using identity mapping",
    ]
