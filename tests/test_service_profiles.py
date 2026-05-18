import pytest

from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_get_llm_base_client_raises_on_unknown_step_profile() -> None:
    service = _service()
    with pytest.raises(KeyError, match="Step profile 'memory_extract' not found in config"):
        service._get_llm_base_client("memory_extract")

