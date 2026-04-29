from __future__ import annotations

from memu.app.service import MemoryService
from memu.database.models import MemoryItem
from memu.prompts.memory_type import CUSTOM_PROMPTS, DEFAULT_MEMORY_TYPES, PROMPTS


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_affect_is_active_in_default_memory_type_registry() -> None:
    assert "affect" in DEFAULT_MEMORY_TYPES
    assert "affect" in PROMPTS
    assert "affect" in CUSTOM_PROMPTS


def test_affect_memory_item_round_trips() -> None:
    item = MemoryItem(resource_id=None, memory_type="affect", summary="Marcos feels shame when he needs rest")
    assert item.memory_type == "affect"


def test_build_affect_prompt_injects_soul_context() -> None:
    service = _service()

    prompt = service._build_memory_type_prompt(
        memory_type="affect",
        resource_text="[0] [user]: I felt sudden relief when the test was over.",
        categories_str="health",
        soul_context_str="## Health\nMarcos has ongoing medical anxiety.",
    )

    assert "## Health\nMarcos has ongoing medical anxiety." in prompt
    assert "{soul_context}" not in prompt
