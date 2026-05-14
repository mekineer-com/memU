import pytest

from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


class _RouterStub:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    async def chat(self, _prompt: str) -> str:
        return self.payload


@pytest.mark.asyncio
async def test_route_episode_uses_excluded_types_model() -> None:
    service = _service()
    client = _RouterStub(
        '{"memorable": true, "excluded_types": ["knowledge", "social"], "episode_summary": "S", "episode_item": "I"}'
    )

    routed, summary, item = await service._route_episode(
        "episode text",
        ["profile", "knowledge", "behavior", "social"],
        llm_client=client,
    )

    assert routed == ["profile", "behavior"]
    assert summary == "S"
    assert item == "I"


def test_parse_structured_entries_requires_episode_ref_when_requested() -> None:
    service = _service()
    missing_ref = """
<item>
  <memory>
    <source_role>user</source_role>
    <content>Marcos values consistency in system behavior</content>
    <categories><category>Identity</category></categories>
  </memory>
</item>
""".strip()
    with_ref = """
<item>
  <memory>
    <episode_ref>2</episode_ref>
    <source_role>user</source_role>
    <content>Marcos values consistency in system behavior</content>
    <categories><category>Identity</category></categories>
  </memory>
</item>
""".strip()

    dropped = service._parse_structured_entries(
        ["profile"],
        [missing_ref],
        require_episode_ref=True,
    )
    kept = service._parse_structured_entries(
        ["profile"],
        [with_ref],
        require_episode_ref=True,
    )

    assert dropped == []
    assert len(kept) == 1
    assert kept[0].episode_ref == 2
