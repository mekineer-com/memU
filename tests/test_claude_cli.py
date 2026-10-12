from __future__ import annotations

import asyncio
import locale
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import memu.llm.claude_cli as claude_cli
import pytest
from memu.llm.claude_cli import ClaudeCLIClient
from memu.llm.wrapper import LLMClientWrapper, LLMInterceptorRegistry


@pytest.fixture
def mock_cli_spawn(monkeypatch):
    def install(run):
        async def spawn(*cmd, cwd, stdin, stdout, stderr):
            assert stdin == stdout == stderr == asyncio.subprocess.PIPE
            process = SimpleNamespace(returncode=None)

            async def communicate(data):
                completed = await run(list(cmd), cwd=cwd, input=data.decode(locale.getpreferredencoding(False)))
                process.returncode = completed.returncode
                return completed.stdout.encode(locale.getpreferredencoding(False)), completed.stderr.encode(
                    locale.getpreferredencoding(False)
                )

            process.communicate = communicate
            return process

        monkeypatch.setattr(claude_cli.asyncio, "create_subprocess_exec", spawn)

    return install


@pytest.mark.asyncio
async def test_claude_cli_uses_workspace_and_cleans_prompt_file(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        prompt_file = Path(cmd[cmd.index("--system-prompt-file") + 1])
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        seen["prompt_file"] = prompt_file
        seen["system_prompt"] = prompt_file.read_text(encoding="utf-8")
        seen["input"] = input
        return subprocess.CompletedProcess(cmd, 0, stdout="reply", stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    text, raw = await client._run_claude(prompt="hello", system_prompt="system")

    assert text == "reply"
    assert raw["provider"] == "claude_code"
    assert seen["cwd"] == tmp_path
    assert seen["input"] == "hello"
    assert "--safe-mode" in seen["cmd"]
    assert seen["system_prompt"] == "system"
    assert not seen["prompt_file"].exists()
    assert list((tmp_path / ".prompts").iterdir()) == []


@pytest.mark.asyncio
async def test_claude_cli_payload_logging_is_request_local(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        if input == "slow":
            await asyncio.sleep(0.05)
        return subprocess.CompletedProcess(cmd, 0, stdout=f"reply:{input}", stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    async def call(prompt: str) -> dict[str, object] | None:
        await client.chat(prompt, system_prompt=f"system:{prompt}")
        return client.get_last_payload()

    slow_task = asyncio.create_task(call("slow"))
    await asyncio.sleep(0.01)
    fast_payload = await call("fast")
    slow_payload = await slow_task

    assert fast_payload is not None
    assert slow_payload is not None
    assert fast_payload["messages"][1]["content"] == "fast"
    assert slow_payload["messages"][1]["content"] == "slow"


@pytest.mark.asyncio
async def test_claude_cli_passes_permission_mode(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="reply", stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(
        model="claude-opus-4-7",
        permission_mode="bypassPermissions",
        workspace=tmp_path,
    )

    await client._run_claude(prompt="hello", system_prompt="system")

    assert "--permission-mode" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--permission-mode") + 1] == "bypassPermissions"


@pytest.mark.asyncio
async def test_claude_cli_passes_settings(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="reply", stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(
        model="claude-opus-4-7",
        settings=str(tmp_path / "fictional-settings.json"),
        workspace=tmp_path,
    )

    await client._run_claude(prompt="hello", system_prompt="system")

    assert "--settings" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--settings") + 1] == str(tmp_path / "fictional-settings.json")


@pytest.mark.asyncio
async def test_claude_cli_passes_session_id(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="reply", stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    _text, raw = await client._run_claude(prompt="hello", system_prompt="system", session_id="turn-123")

    assert "--session-id" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--session-id") + 1] == "turn-123"
    assert "--resume" not in seen["cmd"]
    assert "--system-prompt-snapshot" not in seen["cmd"]
    assert raw["session_id"] == "turn-123"


@pytest.mark.asyncio
async def test_claude_cli_passes_resume_session_id(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        seen["input"] = input
        prompt_file = Path(cmd[cmd.index("--system-prompt-file") + 1])
        seen["system_prompt"] = prompt_file.read_text(encoding="utf-8")
        seen["prompt_file"] = prompt_file
        return subprocess.CompletedProcess(cmd, 0, stdout="reply", stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    _text, raw = await client._run_claude(
        prompt="Continue the fictional task using prior tool results.",
        system_prompt="Fictional real identity.",
        resume_session_id="turn-123",
    )

    assert "--resume" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--resume") + 1] == "turn-123"
    assert seen["cmd"][seen["cmd"].index("--system-prompt-snapshot") + 1] == "off"
    assert "--session-id" not in seen["cmd"]
    assert seen["cwd"] == tmp_path
    assert seen["input"] == "Continue the fictional task using prior tool results."
    assert seen["system_prompt"] == "Fictional real identity."
    assert not seen["prompt_file"].exists()
    assert raw["resume_session_id"] == "turn-123"


@pytest.mark.asyncio
async def test_claude_cli_passes_json_output_format(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"type":"result","result":"{\\"response_target\\":\\"respond\\",\\"response\\":\\"hi\\"}","usage":{"input_tokens":1}}',
            stderr="",
        )

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    text, raw = await client._run_claude(
        prompt="hello",
        system_prompt="system",
        response_format={"type": "json_object"},
    )

    assert "--output-format" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--output-format") + 1] == "json"
    assert text == '{"response_target":"respond","response":"hi"}'
    assert raw["usage"] == {"input_tokens": 1}


@pytest.mark.asyncio
async def test_claude_cli_json_output_requires_result_field(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        return subprocess.CompletedProcess(cmd, 0, stdout='{"type":"result","usage":{}}', stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    with pytest.raises(RuntimeError, match="missing string result"):
        await client._run_claude(
            prompt="hello",
            system_prompt="system",
            response_format={"type": "json_object"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "outer_timeout", "configured_timeout"])
async def test_claude_cli_stops_real_child_through_wrapper(monkeypatch, tmp_path: Path, stop) -> None:
    ready = tmp_path / "ready"
    code = (
        "import sys,time; from pathlib import Path; sys.stdin.buffer.read(); "
        f"Path({str(ready)!r}).write_text('ready'); "
        "sys.stdout.buffer.write(b'x' * 262144); sys.stdout.buffer.flush(); time.sleep(30)"
    )
    real_spawn = asyncio.create_subprocess_exec
    process = None

    async def spawn(*_cmd, **kwargs):
        nonlocal process
        process = await real_spawn(sys.executable, "-c", code, **kwargs)
        return process

    monkeypatch.setattr(claude_cli.asyncio, "create_subprocess_exec", spawn)
    client = ClaudeCLIClient(
        model="fictional", claude_binary=sys.executable, workspace=tmp_path,
        timeout_seconds=1 if stop == "configured_timeout" else 30, min_call_gap_seconds=0,
    )
    wrapper = LLMClientWrapper(client, registry=LLMInterceptorRegistry())
    task = asyncio.create_task(wrapper.chat("fictional input", system_prompt="fictional system"))
    try:
        async with asyncio.timeout(3):
            while not ready.exists():
                await asyncio.sleep(0.005)
        assert list(client._prompt_dir.iterdir())
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        elif stop == "outer_timeout":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 0.02)
        else:
            with pytest.raises(TimeoutError, match="Claude CLI timed out after 1s for model=fictional"):
                await asyncio.wait_for(task, 3)
        assert process is not None
        assert process.returncode is not None
        assert await asyncio.wait_for(process.wait(), 1) != 0
        assert list(client._prompt_dir.iterdir()) == []
        assert client._last_call_monotonic == 0
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if process is not None:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await asyncio.wait_for(process.communicate(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_claude_cli_real_child_preserves_text_semantics(monkeypatch, tmp_path: Path, exit_code) -> None:
    code = (
        "import sys; data=sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(data+b'\\r\\nlast\\rline\\n'); "
        "sys.stderr.buffer.write(b'error\\r\\ndetail\\rline\\n'); "
        f"sys.exit({exit_code})"
    )
    prompt = "caf\u00e9\nsecond\r\nthird\rfourth"
    expected = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, input=prompt,
        text=True, capture_output=True, timeout=3, check=False,
    )
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*_cmd, **kwargs):
        return await real_spawn(sys.executable, "-c", code, **kwargs)

    monkeypatch.setattr(claude_cli.asyncio, "create_subprocess_exec", spawn)
    client = ClaudeCLIClient(
        model="fictional", claude_binary=sys.executable, workspace=tmp_path, timeout_seconds=3,
    )
    if exit_code:
        with pytest.raises(RuntimeError) as error:
            await client.chat(prompt, system_prompt="fictional system")
        assert str(error.value) == f"Claude CLI failed (exit {exit_code}) for model=fictional: {expected.stderr.strip()}"
    else:
        text, raw = await client.chat(prompt, system_prompt="fictional system")
        assert text == expected.stdout
        assert raw["exit_code"] == expected.returncode
    assert list(client._prompt_dir.iterdir()) == []
