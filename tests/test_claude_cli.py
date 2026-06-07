from __future__ import annotations

import subprocess
from pathlib import Path

import memu.llm.claude_cli as claude_cli
from memu.llm.claude_cli import ClaudeCLIClient


def test_claude_cli_uses_workspace_and_cleans_prompt_file(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    def fake_run(cmd, *, cwd, input, text, capture_output, timeout, check):
        prompt_file = Path(cmd[cmd.index("--system-prompt-file") + 1])
        seen["cwd"] = cwd
        seen["prompt_file"] = prompt_file
        seen["system_prompt"] = prompt_file.read_text(encoding="utf-8")
        seen["input"] = input
        return subprocess.CompletedProcess(cmd, 0, stdout="reply", stderr="")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    client = ClaudeCLIClient(model="claude-opus-4-7", workspace=tmp_path)

    text, raw = client._run_claude(prompt="hello", system_prompt="system")

    assert text == "reply"
    assert raw["provider"] == "claude_code"
    assert seen["cwd"] == tmp_path
    assert seen["input"] == "hello"
    assert seen["system_prompt"] == "system"
    assert not seen["prompt_file"].exists()
    assert list((tmp_path / ".prompts").iterdir()) == []


def test_claude_cli_passes_permission_mode(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(claude_cli.shutil, "which", lambda _binary: "/bin/claude")

    def fake_run(cmd, *, cwd, input, text, capture_output, timeout, check):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="reply", stderr="")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    client = ClaudeCLIClient(
        model="claude-opus-4-7",
        permission_mode="bypassPermissions",
        workspace=tmp_path,
    )

    client._run_claude(prompt="hello", system_prompt="system")

    assert "--permission-mode" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--permission-mode") + 1] == "bypassPermissions"
