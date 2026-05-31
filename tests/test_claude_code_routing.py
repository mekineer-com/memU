from __future__ import annotations

import memu.app.service as service_module
from memu.app.service import MemoryService
from memu.llm.http_client import HTTPLLMClient


class _FakeClaudeCLIClient:
    provider = "claude_code"

    def __init__(self, *, model: str, effort: str | None = None) -> None:
        self.chat_model = model
        self.effort = effort
        self.embed_model = None

    async def chat(self, prompt: str, **_: object):  # pragma: no cover - not needed in this seam test
        return prompt, {"provider": self.provider}


def _service(**kwargs) -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        **kwargs,
    )


def test_claude_code_retrieve_only_uses_claude_for_retrieve_steps(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_scope="retrieve_only", claude_code_model="claude-opus-4-7")

    retrieve_client = service._get_step_llm_client({"workflow_name": "retrieve_rag", "step_id": "route_intention"})
    memorize_client = service._get_step_llm_client({"workflow_name": "memorize", "step_id": "memory_extract"})

    assert isinstance(retrieve_client._client, _FakeClaudeCLIClient)
    assert isinstance(memorize_client._client, HTTPLLMClient)


def test_claude_code_all_chat_routes_all_workflow_chat_steps(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_scope="all_chat", claude_code_model="claude-opus-4-7")

    retrieve_client = service._get_step_llm_client({"workflow_name": "retrieve_rag", "step_id": "route_intention"})
    memorize_client = service._get_step_llm_client({"workflow_name": "memorize", "step_id": "memory_extract"})

    assert isinstance(retrieve_client._client, _FakeClaudeCLIClient)
    assert isinstance(memorize_client._client, _FakeClaudeCLIClient)


def test_claude_code_does_not_change_embedding_client(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_scope="all_chat", claude_code_model="claude-opus-4-7")

    embedding_client = service._get_step_embedding_client(
        {"workflow_name": "retrieve_rag", "step_id": "route_category", "step_config": {"embed_llm_profile": "default"}}
    )

    assert isinstance(embedding_client._client, HTTPLLMClient)
