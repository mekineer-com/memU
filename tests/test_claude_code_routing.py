from __future__ import annotations

import pytest

import memu.app.service as service_module
from memu.app.service import MemoryService
from memu.llm.http_client import HTTPLLMClient
from memu.llm.wrapper import LLMClientWrapper


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


def test_claude_code_routes_chat_workflow_steps_to_neutral_workspace(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    monkeypatch.setenv("HOME", str(tmp_path))
    service = _service(
        claude_code=True,
        claude_code_model="claude-opus-4-7",
        claude_code_workspace="/tmp/siri-workspace",
    )

    retrieve_client = service._get_step_llm_client({"workflow_name": "retrieve_rag", "step_id": "route_intention"})
    memorize_client = service._get_step_llm_client({"workflow_name": "memorize", "step_id": "memory_extract"})

    assert isinstance(retrieve_client._client, _FakeClaudeCLIClient)
    assert isinstance(memorize_client._client, _FakeClaudeCLIClient)
    assert retrieve_client._client.workspace == tmp_path / ".cache" / "memu-claude-internal"
    assert memorize_client._client.workspace == tmp_path / ".cache" / "memu-claude-internal"


def test_claude_code_does_not_change_embedding_client(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_model="claude-opus-4-7")

    embedding_client = service._get_step_embedding_client(
        {"workflow_name": "retrieve_rag", "step_id": "route_category", "step_config": {"embed_llm_profile": "default"}}
    )

    assert isinstance(embedding_client._client, HTTPLLMClient)


@pytest.mark.asyncio
async def test_service_chat_without_session_uses_neutral_workspace(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    monkeypatch.setenv("HOME", str(tmp_path))
    service = _service(
        claude_code=True,
        claude_code_model="claude-opus-4-7",
        claude_code_workspace="/tmp/siri-workspace",
    )

    out = await service.chat("hello", op="manual_chat")

    assert out == "hello"
    assert service._claude_cli_client is None
    assert service._claude_cli_internal_client is not None
    assert service._claude_cli_internal_client.workspace == tmp_path / ".cache" / "memu-claude-internal"


@pytest.mark.asyncio
async def test_service_chat_passes_claude_session_args_through_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(
        claude_code=True,
        claude_code_model="claude-opus-4-7",
        claude_code_workspace="/tmp/siri-workspace",
    )

    out = await service.chat("hello", op="manual_chat", session_id="turn-123")

    assert out == "hello"
    assert service._get_claude_cli_client().last_chat_kwargs["session_id"] == "turn-123"
    assert service._get_claude_cli_client().workspace == "/tmp/siri-workspace"
    assert service._claude_cli_internal_client is None


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


def test_get_step_llm_client_with_explicit_profile_uses_claude_in_claude_code_mode(monkeypatch) -> None:
    """_get_step_llm_client with a profile override must still return the Claude CLI client
    when claude_code=True, and must return the profile's HTTP client otherwise."""
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)

    # claude_code=True: profile arg must be ignored in favour of Claude CLI
    svc_claude = _service(claude_code=True, claude_code_model="claude-opus-4-7")
    chat_client = svc_claude._get_step_llm_client(
        {"operation": "memorize", "step_id": "extract_items_batch"},
        profile="some_profile",
    )
    assert isinstance(chat_client, LLMClientWrapper)
    assert isinstance(chat_client._client, _FakeClaudeCLIClient)

    # Embedding acquisition must NOT use Claude CLI even when claude_code=True
    embed_client = svc_claude._get_step_embedding_client(
        {"operation": "memorize", "step_id": "embed", "step_config": {"embed_llm_profile": "default"}}
    )
    assert isinstance(embed_client, LLMClientWrapper)
    assert isinstance(embed_client._client, HTTPLLMClient)

    # claude_code=False: profile arg selects the HTTP client (falls back to "default" profile)
    svc_plain = _service(claude_code=False)
    chat_client_plain = svc_plain._get_step_llm_client(
        {"operation": "memorize", "step_id": "extract_items_batch"},
        profile="default",
    )
    assert isinstance(chat_client_plain, LLMClientWrapper)
    assert isinstance(chat_client_plain._client, HTTPLLMClient)


@pytest.mark.asyncio
async def test_background_rollup_uses_claude_code_step_client(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "ClaudeCLIClient", _FakeClaudeCLIClient)
    service = _service(claude_code=True, claude_code_model="claude-opus-4-7")

    out = await service.summarize_background_chat_rollup(
        prior_summary=None,
        messages=[
            {
                "role": "user",
                "speaker": "Marcos",
                "content": "This should route through Claude Code.",
                "source_conversation_index": 1,
            }
        ],
        soul_name="Siri",
    )

    assert "This should route through Claude Code." in out
    assert isinstance(service._claude_cli_internal_client, _FakeClaudeCLIClient)
    assert service._claude_cli_internal_client.chat_model == "claude-opus-4-7"
