from __future__ import annotations

import asyncio
import json
import locale
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import memu.llm.claude_cli as claude_cli
import pytest
from memu.llm.claude_cli import ClaudeCLIClient
from memu.llm.wrapper import LLMClientWrapper, LLMInterceptorRegistry


def _result_envelope(result="reply", **changes):
    return {
        "type": "result", "subtype": "success", "is_error": False,
        "result": result, "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        **changes,
    }


def test_claude_session_path_uses_effective_home_and_workspace(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_CODE_PROJECT_DIR_NAME", raising=False)
    home = tmp_path / "native-home"
    workspace = (tmp_path / "Soul space" / "under_score").resolve()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    native_id = "11111111-2222-4333-8444-555555555555"
    escaped = claude_cli.re.sub(r"[^a-zA-Z0-9]", "-", str(workspace))
    assert claude_cli.claude_session_path(workspace, native_id) == home / "projects" / escaped / f"{native_id}.jsonl"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "relative-home")
    assert claude_cli.claude_session_path(workspace, native_id) == workspace / "relative-home" / "projects" / escaped / f"{native_id}.jsonl"
    with pytest.raises(ValueError):
        claude_cli.claude_session_path(workspace, "../another-session")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "native-cafe\u0301"))
    monkeypatch.setenv("CLAUDE_CODE_PROJECT_DIR_NAME", "Fixed_Key")
    assert claude_cli.claude_session_path(workspace, native_id) == tmp_path / "native-caf\u00e9" / "projects" / "Fixed_Key" / f"{native_id}.jsonl"


@pytest.mark.parametrize("workspace,expected", [
    ("/tmp/" + "a" * 195, "-tmp-" + "a" * 195),
    ("/tmp/" + "a" * 196, "-tmp-" + "a" * 195 + "-r0traf"),
    ("/tmp/caf\u00e9/\U0001f600", "-tmp-caf----"),
])
def test_claude_project_key_matches_vendor_fixtures(workspace, expected):
    assert claude_cli._claude_project_name(workspace) == expected


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
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_result_envelope(stop_reason=None)), stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    text, raw = await client._run_claude(prompt="hello", system_prompt="system")

    assert text == "reply"
    assert raw["provider"] == "claude_code"
    assert raw["stop_reason"] is None
    assert raw["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert seen["cwd"] == tmp_path
    assert seen["input"] == "hello"
    assert "--safe-mode" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--output-format") + 1] == "json"
    assert seen["system_prompt"] == "system"
    assert not seen["prompt_file"].exists()
    assert list((tmp_path / ".prompts").iterdir()) == []


@pytest.mark.asyncio
async def test_claude_cli_payload_logging_is_request_local(monkeypatch, tmp_path: Path, mock_cli_spawn) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    async def fake_run(cmd, *, cwd, input):
        if input == "slow":
            await asyncio.sleep(0.05)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_result_envelope(f"reply:{input}")), stderr="")

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
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_result_envelope()), stderr="")

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
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_result_envelope()), stderr="")

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
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_result_envelope()), stderr="")

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
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_result_envelope()), stderr="")

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
            stdout=json.dumps(_result_envelope('{"response_target":"respond","response":"hi"}', usage={"input_tokens": 1})),
            stderr="",
        )

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    text, raw = await client.chat(
        "hello",
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
        envelope = _result_envelope()
        del envelope["result"]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(envelope), stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    with pytest.raises(RuntimeError, match="missing string result"):
        await client.chat(
            "hello",
            system_prompt="system",
            response_format={"type": "json_object"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("result,response_format,stop_reason", [
    pytest.param(" \nordinary prose\n ", None, None, id="prose"),
    pytest.param('<memories><memory>2 < 3 & caf\u00e9</memory></memories>\n', None, "end_turn", id="extraction-xml"),
    pytest.param('{"response":"hi"}\n', {"type": "json_object"}, "end_turn", id="json-content"),
])
async def test_claude_cli_envelope_preserves_success(monkeypatch, tmp_path, mock_cli_spawn, result, response_format, stop_reason):
    calls = []
    after = []
    errors = []
    envelope = _result_envelope(result, stop_reason=stop_reason)

    async def fake_run(cmd, *, cwd, input):
        calls.append(cmd)
        assert cmd[cmd.index("--output-format") + 1] == "json"
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(envelope), stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    registry = LLMInterceptorRegistry(strict=True)
    registry.register_after(lambda ctx, request, response, usage: after.append((response, usage)))
    registry.register_on_error(lambda *args: errors.append(args))
    wrapper = LLMClientWrapper(client, registry=registry)
    assert await wrapper.chat("fictional prompt", response_format=response_format) == result
    assert len(calls) == len(after) == 1
    assert errors == []
    response, usage = after[0]
    assert response.content == result
    assert usage.status == "success"
    assert list((client._workspace / ".prompts").iterdir()) == []


@pytest.fixture
def stream_child(monkeypatch, tmp_path):
    def install(code):
        script = tmp_path / "fictional-child.py"
        script.write_text(code, encoding="utf-8")
        spawn_real = asyncio.create_subprocess_exec
        calls = []

        async def spawn(*cmd, **kwargs):
            mcp_path = Path(cmd[cmd.index("--mcp-config") + 1])
            call = {
                "cmd": cmd, "cwd": kwargs["cwd"], "env": kwargs["env"],
                "mcp": json.loads(mcp_path.read_text()), "mode": stat.S_IMODE(mcp_path.stat().st_mode),
                "prompt": Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text()
                if "--system-prompt-file" in cmd else None,
            }
            process = await spawn_real(sys.executable, str(script), **kwargs)
            call["process"] = process
            calls.append(call)
            return process

        monkeypatch.setattr(claude_cli.asyncio, "create_subprocess_exec", spawn)
        return calls

    return install


def _scoped_tools():
    return {
        "mcp_servers": {"atomic": {"type": "http", "url": "http://127.0.0.1:1/mcp",
                                   "headers": {"Authorization": "Bearer fictional-secret"}}},
        "allowed_tools": ["mcp__atomic__search_atoms"],
    }


@pytest.mark.asyncio
async def test_scoped_input_checkpoint_failure_prevents_input_delivery(monkeypatch, tmp_path, stream_child):
    marker = tmp_path / "input-observed"
    calls = stream_child("import sys,pathlib; data=sys.stdin.read(); pathlib.Path(" + repr(str(marker)) + ").write_text(data)")
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    writes = []
    async def checkpoint(event):
        if event.get("state") == "started":
            writer = calls[0]["process"].stdin
            write = writer.write
            def observed(data):
                writes.append(data)
                return write(data)
            monkeypatch.setattr(writer, "write", observed)
        if event.get("state") == "input_started":
            raise RuntimeError("fictional checkpoint failure")
    with pytest.raises(RuntimeError, match="checkpoint failure"):
        await client.chat("must not arrive", session_id="fictional", system_prompt="Presence",
                          workspace=tmp_path, cli_tools=_scoped_tools(), on_event=checkpoint)
    assert not marker.exists()
    assert writes == [], "even a fast cleanup must not hide premature stdin writes"
    assert calls[0]["process"].returncode is not None


@pytest.mark.asyncio
async def test_scoped_stream_large_split_events_final_usage_and_isolated_calls(tmp_path, stream_child):
    code = '''import sys,json,time
prompt=sys.stdin.buffer.read().decode("utf-8")
events=[
 {"type":"stream_event","event":{"type":"message_start","message":{"id":"previous","usage":{"input_tokens":900,"cache_creation_input_tokens":None,"output_tokens":1}}}},
 {"type":"stream_event","event":{"type":"message_stop"}},
 {"type":"stream_event","event":{"type":"message_start","message":{"id":"primary","usage":{"input_tokens":2,"cache_creation_input_tokens":5,"cache_read_input_tokens":30,"output_tokens":0}}}},
 {"type":"stream_event","parent_tool_use_id":"nested","event":{"type":"message_start","message":{"id":"nested","usage":{"input_tokens":999}}}},
 {"type":"assistant","message":{"content":[{"type":"text","text":"caf\\u00e9"+"x"*100000}],"usage":{"output_tokens":8}}},
 {"type":"stream_event","event":{"type":"message_delta","usage":{"input_tokens":None,"cache_creation_input_tokens":0,"cache_read_input_tokens":None,"output_tokens":16}}},
 {"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":None}}},
 {"type":"stream_event","event":{"type":"message_stop"}},
 {"type":"stream_event","event":{"type":"message_start","message":{"id":"unknown-input","usage":{"input_tokens":None}}}},
 {"type":"stream_event","event":{"type":"message_stop"}},
 {"type":"stream_event","event":{"type":"message_start","message":{"id":"unknown-output","usage":{"input_tokens":1,"output_tokens":None}}}},
 {"type":"stream_event","event":{"type":"message_stop"}},
 {"type":"result","subtype":"success","is_error":False,"result":prompt,"stop_reason":"end_turn","usage":{"input_tokens":9999}}]
sys.stderr.buffer.write(b"x"*150000);sys.stderr.flush()
for e in events:
 raw=(json.dumps(e,ensure_ascii=False)+"\\n").encode()
 cut=raw.index(b"\\xc3")+1 if b"\\xc3" in raw else len(raw)//2
 sys.stdout.buffer.write(raw[:cut]);sys.stdout.flush();time.sleep(.001)
 sys.stdout.buffer.write(raw[cut:]);sys.stdout.flush()
'''
    calls = stream_child(code)
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable,
                             workspace=tmp_path / "shared", min_call_gap_seconds=0,
                             permission_mode="bypassPermissions", settings="unwanted-settings.json")
    registry = LLMInterceptorRegistry()
    logged = []
    registry.register_after(lambda _ctx, request, _response, _usage: logged.append(request.metadata))
    wrapped = LLMClientWrapper(client, registry=registry)
    events = {"birch": [], "cedar": []}

    async def turn(name):
        async def capture(event):
            events[name].append(event)
            if event["type"] == "result":
                assert list((tmp_path / name / ".prompts").iterdir()) == []
                assert all(call["process"].returncode is not None for call in calls if call["cwd"] == tmp_path / name)
        return await wrapped.chat(name, session_id=name, system_prompt="Presence:"+name,
                                  workspace=tmp_path / name, cli_tools=_scoped_tools(), on_event=capture)

    assert await asyncio.gather(turn("birch"), turn("cedar")) == ["birch", "cedar"]
    assert client._workspace == tmp_path / "shared"
    assert len(calls) == 2
    for call in calls:
        cmd = call["cmd"]
        assert "--safe-mode" not in cmd and "unwanted-settings.json" not in cmd
        assert cmd[cmd.index("--settings")+1] == "{}"
        assert cmd[cmd.index("--permission-mode")+1] == "default"
        assert cmd[cmd.index("--system-prompt-snapshot")+1] == "on"
        assert cmd[cmd.index("--tools")+1] == ""
        assert "--restricted" in cmd
        assert "--strict-mcp-config" in cmd
        assert cmd[cmd.index("--setting-sources")+1] == ""
        assert cmd[cmd.index("--max-turns")+1] == "10"
        assert "--include-partial-messages" in cmd
        assert cmd[cmd.index("--allowedTools")+1] == "mcp__atomic__search_atoms"
        assert call["env"]["DISABLE_AUTO_COMPACT"] == "1"
        assert call["mode"] == 0o600
        assert call["mcp"]["mcpServers"]["atomic"]["headers"]["Authorization"] == "Bearer fictional-secret"
        assert call["prompt"] == "Presence:" + call["cwd"].name
    for received in events.values():
        assert received[0]["type"] == "process" and received[0]["state"] == "started"
        reaped = next(i for i, event in enumerate(received) if event.get("state") == "reaped")
        completed = next(i for i, event in enumerate(received) if event["type"] == "result")
        assert reaped < completed
        snapshots = [e for e in received if e["type"] == "context_usage"]
        starts = [e["message_id"] for e in received if e["type"] == "context_start"]
        assert starts == ["previous", "primary", "unknown-input", "unknown-output"]
        assert [e["message_id"] for e in snapshots] == ["previous", "primary"]
        assert snapshots[0]["usage"]["cache_creation_input_tokens"] == 0
        assert snapshots[-1] == {"type": "context_usage", "message_id": "primary",
                                 "usage": {"input_tokens": 2, "cache_creation_input_tokens":0, "cache_read_input_tokens": 30, "output_tokens": 16}}
        assert received[-1]["type"] == "result"
    assert "fictional-secret" not in repr(logged)
    assert "cli_tools" not in repr(logged) and "on_event" not in repr(logged)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["nonzero", "native_error", "native_api_error", "malformed", "malformed_diagnostic", "missing", "callback", "timeout", "stdin_timeout", "refusal"])
async def test_scoped_stream_failure_reaps_child_and_removes_files(tmp_path, stream_child, failure):
    terminal = _result_envelope(stop_reason="refusal") if failure == "refusal" else _result_envelope()
    if failure == "native_error":
        terminal = _result_envelope(subtype="error_max_turns", is_error=True, errors=["fictional-secret quota exhausted"])
    elif failure == "native_api_error":
        terminal = _result_envelope(result="fictional rate limit", is_error=True, api_error_status=429)
    code = "import sys,json,time;sys.stdin.buffer.read();"
    if failure == "stdin_timeout":
        code = "import time;time.sleep(30)"
    elif failure == "malformed_diagnostic":
        code += "sys.stderr.write('fictional-secret parse detail');sys.stderr.flush();time.sleep(.02);"
        code += "print('{broken',flush=True);time.sleep(30)"
    elif failure == "malformed":
        code += "print('{broken',flush=True);time.sleep(30)"
    elif failure in {"callback", "timeout"}:
        code += "print(json.dumps({'type':'system','subtype':'init'}),flush=True);time.sleep(30)"
    elif failure == "missing":
        code += "print(json.dumps({'type':'system','subtype':'init'}))"
    else:
        code += f"print(json.dumps({terminal!r}),flush=True);sys.exit({7 if failure in {'nonzero', 'native_error'} else 0})"
    calls = stream_child(code)
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path,
                             timeout_seconds=0.1 if failure in {"timeout", "stdin_timeout"} else 3)
    hooks = []
    registry = LLMInterceptorRegistry()
    registry.register_before(lambda *_args: hooks.append("before"))
    registry.register_after(lambda *_args: hooks.append("after"))
    registry.register_on_error(lambda *_args: hooks.append("error"))
    wrapped = LLMClientWrapper(client, registry=registry)
    events = []
    async def capture(event):
        events.append(event)
        if failure == "callback":
            raise RuntimeError("fictional callback failure")
    with pytest.raises((RuntimeError, TimeoutError)) as error:
        await wrapped.chat("x" * 500000 if failure == "stdin_timeout" else "question",
                          session_id="fictional-session", system_prompt="fictional Presence",
                          workspace=tmp_path, cli_tools=_scoped_tools(), on_event=capture)
    if failure == "native_error":
        assert "error_max_turns" in str(error.value) and "quota exhausted" in str(error.value)
        assert "fictional-secret" not in str(error.value)
    if failure == "malformed_diagnostic":
        assert "parse detail" in str(error.value) and "fictional-secret" not in str(error.value)
    assert calls[0]["process"].returncode is not None
    assert hooks == ["before", "error"]
    assert any(e["type"] == "result" for e in events) == (failure in {"native_error", "native_api_error"})
    if failure in {"native_error", "native_api_error"}:
        assert events[-1]["is_error"] is True
        assert events[-2]["state"] == "reaped"
    if failure == "native_api_error":
        assert events[-1]["subtype"] == "success" and events[-1]["api_error_status"] == 429
    assert list((client._workspace / ".prompts").iterdir()) == []


@pytest.mark.asyncio
async def test_scoped_resume_is_tool_free_and_does_not_resend_presence(tmp_path, stream_child):
    calls = stream_child(f"import sys,json;sys.stdin.buffer.read();print(json.dumps({_result_envelope()!r}))")
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    assert (await client.chat("recap", resume_session_id="fictional-session",
                              workspace=tmp_path, cli_tools={"mcp_servers": {}, "allowed_tools": []}))[0] == "reply"
    assert calls[0]["prompt"] is None
    assert calls[0]["mcp"] == {"mcpServers": {}}
    assert calls[0]["cmd"][calls[0]["cmd"].index("--allowedTools")+1] == ""
    assert list((client._workspace / ".prompts").iterdir()) == []


@pytest.mark.asyncio
async def test_scoped_stream_spawn_failure_cleans_files(monkeypatch, tmp_path):
    async def fail_spawn(*_cmd, **_kwargs):
        raise OSError("fictional spawn failure")
    monkeypatch.setattr(claude_cli.asyncio, "create_subprocess_exec", fail_spawn)
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    with pytest.raises(OSError, match="fictional spawn"):
        await client.chat("question", session_id="fictional", system_prompt="Presence",
                          workspace=tmp_path, cli_tools=_scoped_tools())
    assert list((client._workspace / ".prompts").iterdir()) == []


@pytest.mark.asyncio
async def test_scoped_terminal_delivery_does_not_relabel_completed_reply_as_timeout(tmp_path, stream_child):
    calls = stream_child(f"import sys,json;sys.stdin.buffer.read();print(json.dumps({_result_envelope()!r}))")
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path,
                             timeout_seconds=0.2)
    async def delayed_terminal(event):
        if event["type"] == "result":
            await asyncio.sleep(0.25)
    assert (await client.chat("question", session_id="fictional", system_prompt="Presence",
                              workspace=tmp_path, cli_tools=_scoped_tools(), on_event=delayed_terminal))[0] == "reply"
    assert calls[0]["cwd"] == tmp_path
    cmd = calls[0]["cmd"]
    assert Path(cmd[cmd.index("--system-prompt-file")+1]).is_absolute()
    assert Path(cmd[cmd.index("--mcp-config")+1]).is_absolute()
    assert list((tmp_path / ".prompts").iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", [None, "relative-workspace"])
async def test_scoped_workspace_must_be_explicit_and_absolute(tmp_path, workspace):
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    with pytest.raises(ValueError, match="explicit absolute workspace"):
        await client.chat("question", session_id="fictional", system_prompt="Presence",
                          workspace=workspace, cli_tools=_scoped_tools())


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["stderr", "result"])
async def test_scoped_errors_redact_mcp_credentials(tmp_path, stream_child, channel):
    if channel == "stderr":
        code = "import sys;sys.stdin.buffer.read();sys.stderr.write('Bearer fictional-secret');sys.exit(1)"
    else:
        envelope = _result_envelope(subtype="error_during_execution", is_error=True,
                                    errors=["Bearer fictional-secret"])
        code = f"import sys,json;sys.stdin.buffer.read();print(json.dumps({envelope!r}))"
    stream_child(code)
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    with pytest.raises(RuntimeError) as error:
        await client.chat("question", session_id="fictional", system_prompt="Presence",
                          workspace=tmp_path, cli_tools=_scoped_tools())
    assert "fictional-secret" not in str(error.value)
    assert "[redacted]" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
async def test_scoped_timeout_reaps_inherited_pipe_group(tmp_path, stream_child):
    calls = stream_child("import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])")
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path,
                             timeout_seconds=0.2)
    with pytest.raises(TimeoutError, match="Claude CLI timed out"):
        await asyncio.wait_for(client.chat("question", session_id="fictional", system_prompt="Presence",
                                           workspace=tmp_path, cli_tools=_scoped_tools()), 2)
    assert calls[0]["process"].returncode is not None
    assert list((client._workspace / ".prompts").iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
async def test_scoped_success_does_not_leave_owned_background_child(monkeypatch, tmp_path, stream_child):
    pid_file = tmp_path / "descendant.pid"
    calls = stream_child(
        "import subprocess,sys,json,pathlib;sys.stdin.buffer.read();"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid));"
        f"print(json.dumps({_result_envelope()!r}))"
    )
    kill_group = os.killpg
    killed = []
    def capture_kill(pgid, sig):
        assert os.getpgid(int(pid_file.read_text())) == pgid
        killed.append(pgid)
        kill_group(pgid, sig)
    monkeypatch.setattr(os, "killpg", capture_kill)
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    try:
        assert (await client.chat("question", session_id="fictional", system_prompt="Presence",
                                   workspace=tmp_path, cli_tools=_scoped_tools()))[0] == "reply"
        assert killed == [calls[0]["process"].pid]
        if sys.platform.startswith("linux"):
            status = Path(f"/proc/{int(pid_file.read_text())}/status")
            for _ in range(100):
                try:
                    state = next(line.split()[1] for line in status.read_text().splitlines() if line.startswith("State:"))
                except FileNotFoundError:
                    break
                if state in {"Z", "X"}:  # A zombie cannot keep writing before init reaps it.
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("owned background child survived cleanup")
    finally:
        if calls:
            try:
                kill_group(calls[0]["process"].pid, claude_cli.signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_scoped_large_input_keeps_early_launch_failure_reason(tmp_path, stream_child):
    stream_child("import sys;sys.stderr.write('fictional launch rejected');sys.exit(1)")
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    with pytest.raises(RuntimeError, match="fictional launch rejected"):
        await client.chat("x" * 500000, session_id="fictional", system_prompt="Presence",
                          workspace=tmp_path, cli_tools=_scoped_tools())


@pytest.mark.asyncio
@pytest.mark.parametrize("output,message", [
    *[pytest.param(json.dumps(_result_envelope(subtype=subtype, is_error=True, errors=["fictional failure"])),
                   "fictional failure", id=subtype)
      for subtype in ("error_during_execution", "error_max_turns", "error_max_budget_usd", "error_max_structured_output_retries")],
    pytest.param(json.dumps({key: value for key, value in _result_envelope(
        subtype="error_during_execution", is_error=True, errors=["fictional failure"]
    ).items() if key != "result"}), "fictional failure", id="error-without-result"),
    pytest.param(json.dumps(_result_envelope(is_error=True)), "Claude CLI failed", id="success-is-error"),
    pytest.param(json.dumps(_result_envelope(subtype="error_max_turns")), "Claude CLI failed", id="error-subtype-false-flag"),
    pytest.param(json.dumps(_result_envelope(stop_reason="refusal")), "refused by provider", id="refusal"),
    pytest.param(json.dumps(_result_envelope(stop_reason="max_tokens")), "truncated at the output token limit", id="max-tokens"),
    pytest.param(json.dumps(_result_envelope(" \n\t")), "returned empty result", id="blank-result"),
    pytest.param(json.dumps(_result_envelope(None)), "missing string result field", id="nonstring-result"),
    pytest.param("{", "not valid JSON", id="malformed-json"),
    pytest.param("[]", "missing result envelope", id="nonobject-envelope"),
    pytest.param(json.dumps(_result_envelope(type="assistant")), "missing result envelope", id="wrong-envelope-type"),
    pytest.param(json.dumps({"result": "reply"}), "missing result envelope", id="missing-envelope-type"),
    pytest.param(json.dumps({key: value for key, value in _result_envelope().items() if key != "subtype"}),
                 "Claude CLI failed", id="missing-status"),
    pytest.param(json.dumps({key: value for key, value in _result_envelope().items() if key != "is_error"}),
                 "Claude CLI failed", id="missing-error-flag"),
])
async def test_claude_cli_envelope_fails_before_parser(monkeypatch, tmp_path, mock_cli_spawn, output, message):
    calls = []
    after = []
    errors = []
    parsed = []

    async def fake_run(cmd, *, cwd, input):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    mock_cli_spawn(fake_run)
    client = ClaudeCLIClient(model="fictional", claude_binary=sys.executable, workspace=tmp_path)
    registry = LLMInterceptorRegistry(strict=True)
    registry.register_after(lambda *args: after.append(args))
    registry.register_on_error(lambda ctx, request, error, usage: errors.append((error, usage)))
    wrapper = LLMClientWrapper(client, registry=registry)

    with pytest.raises(RuntimeError, match=message) as raised:
        reply = await wrapper.chat("fictional extraction prompt")
        parsed.append(reply)
    assert len(calls) == len(errors) == 1
    assert errors[0][0] is raised.value
    assert errors[0][1].status == "error"
    assert after == parsed == []
    assert client._last_call_monotonic == 0
    assert list((client._workspace / ".prompts").iterdir()) == []


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

    async def spawn(*cmd, **kwargs):
        nonlocal process
        assert cmd[cmd.index("--output-format") + 1] == "json"
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
        assert list((client._workspace / ".prompts").iterdir())
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
        assert list((client._workspace / ".prompts").iterdir()) == []
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
        "import sys,json,locale; data=sys.stdin.buffer.read(); "
        "result=(data+b'\\r\\nlast\\rline\\n').decode(locale.getpreferredencoding(False)); "
        f"envelope={_result_envelope()!r}; envelope['result']=result; "
        "sys.stdout.write(json.dumps(envelope)+'\\r\\n'); "
        "sys.stderr.buffer.write(b'error\\r\\ndetail\\rline\\n'); "
        f"sys.exit({exit_code})"
    )
    prompt = "caf\u00e9\nsecond\r\nthird\rfourth"
    expected = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, input=prompt,
        text=True, capture_output=True, timeout=3, check=False,
    )
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*cmd, **kwargs):
        assert cmd[cmd.index("--output-format") + 1] == "json"
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
        assert text == json.loads(expected.stdout)["result"]
        assert raw["exit_code"] == expected.returncode
        assert raw["usage"] == {"input_tokens": 1, "output_tokens": 2}
        assert raw["stop_reason"] == "end_turn"
    assert list((client._workspace / ".prompts").iterdir()) == []
