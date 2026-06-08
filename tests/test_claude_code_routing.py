from __future__ import annotations

import pytest

import memu.app.service as service_module
from memu.app.service import MemoryService
from memu.llm.http_client import HTTPLLMClient


class _FakeClaudeCLIClient:
    provider = "claude_code"

    def __init__(
        self,
        *,
        model: str,
        effort: str | None = None,
        permission_mode: str | None = None,
        settings: str | None = None,
        workspace: str | None = None,
        timeout_seconds: int = 900,
    ) -> None:
        self.chat_model = model
        self.effort = effort
        self.permission_mode = permission_mode
        self.settings = settings
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.embed_model = None
        self.last_chat_kwargs: dict[str, object] = {}

    async def chat(self, prompt: str, **kwargs: object):
        self.last_chat_kwargs = dict(kwargs)
        return prompt, {"provider": self.provider}


def _service(**kwargs) -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        **kwargs,
    )


def test_claude_code_routes_all_chat_workflow_steps(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_model="claude-opus-4-7")

    retrieve_client = service._get_step_llm_client({"workflow_name": "retrieve_rag", "step_id": "route_intention"})
    memorize_client = service._get_step_llm_client({"workflow_name": "memorize", "step_id": "memory_extract"})

    assert isinstance(retrieve_client._client, _FakeClaudeCLIClient)
    assert isinstance(memorize_client._client, _FakeClaudeCLIClient)


def test_claude_code_does_not_change_embedding_client(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_model="claude-opus-4-7")

    embedding_client = service._get_step_embedding_client(
        {"workflow_name": "retrieve_rag", "step_id": "route_category", "step_config": {"embed_llm_profile": "default"}}
    )

    assert isinstance(embedding_client._client, HTTPLLMClient)


@pytest.mark.asyncio
async def test_service_chat_uses_claude_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_model="claude-opus-4-7")
    out = await service.chat("hello", op="manual_chat")
    assert out == "hello"


@pytest.mark.asyncio
async def test_service_chat_passes_claude_session_args_through_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_model="claude-opus-4-7")

    out = await service.chat("hello", op="manual_chat", session_id="turn-123")

    assert out == "hello"
    assert service._get_claude_cli_client().last_chat_kwargs["session_id"] == "turn-123"


@pytest.mark.asyncio
async def test_service_chat_rejects_claude_session_args_without_claude(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=False)

    with pytest.raises(ValueError, match="claude_code=True"):
        await service.chat("hello", session_id="turn-123")


def test_claude_code_passes_workspace(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(
        claude_code=True,
        claude_code_model="claude-opus-4-7",
        claude_code_workspace="/tmp/siri-workspace",
    )

    client = service._get_claude_cli_client()

    assert client.workspace == "/tmp/siri-workspace"


def test_claude_code_passes_permission_mode(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(
        claude_code=True,
        claude_code_model="claude-opus-4-7",
        claude_code_permission_mode="bypassPermissions",
    )

    client = service._get_claude_cli_client()

    assert client.permission_mode == "bypassPermissions"


def test_claude_code_passes_settings(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(
        claude_code=True,
        claude_code_model="claude-opus-4-7",
        claude_code_settings="/tmp/siri-settings.json",
    )

    client = service._get_claude_cli_client()

    assert client.settings == "/tmp/siri-settings.json"


def test_claude_code_passes_timeout_seconds(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(
        claude_code=True,
        claude_code_model="claude-opus-4-7",
        claude_code_timeout_seconds=3600,
    )

    client = service._get_claude_cli_client()

    assert client.timeout_seconds == 3600
