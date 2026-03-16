import pytest

from memu.app.service import MemoryService
from memu.database.models import MemoryType


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "inmemory"}})


@pytest.mark.parametrize("memory_type", ["knowledge", "behavior"])
def test_build_memory_type_prompt_injects_soul_context_for_activated_types(memory_type: MemoryType) -> None:
    service = _service()

    prompt = service._build_memory_type_prompt(
        memory_type=memory_type,
        resource_text="[0] [user]: I learned something important.",
        categories_str="health",
        soul_context_str="## Health\nMarcos has ongoing circulation concerns.",
    )

    assert "## Health\nMarcos has ongoing circulation concerns." in prompt
    assert "{soul_context}" not in prompt


@pytest.mark.parametrize("memory_type", ["knowledge", "behavior"])
def test_parse_structured_entries_keeps_salience_and_source_ids_for_activated_types(memory_type: MemoryType) -> None:
    service = _service()

    response = """
<item>
    <memory>
        <content>Marcos may benefit from pausing before replying when a thought spans multiple messages</content>
        <source_role>user</source_role>
        <confidence>0.8</confidence>
        <reflection_salience>0.72</reflection_salience>
        <source_message_ids>
            <id>3</id>
            <id>5</id>
        </source_message_ids>
        <categories>
            <category>Communication</category>
        </categories>
    </memory>
</item>
""".strip()

    entries = service._parse_structured_entries(
        [memory_type],
        [response],
        default_source_message_ids=[3, 5, 8],
    )

    assert len(entries) == 1
    assert entries[0].memory_type == memory_type
    assert entries[0].categories == ["communication"]
    assert entries[0].source_role == "user"
    assert entries[0].confidence == pytest.approx(0.8)
    assert entries[0].source_message_ids == [3, 5]
    assert entries[0].reflection_salience == pytest.approx(0.72)
