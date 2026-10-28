from __future__ import annotations

import asyncio
import copy
import contextvars
import json
import locale
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypedDict


class ClaudeCLITools(TypedDict):
    mcp_servers: dict[str, Any]
    allowed_tools: list[str]


ClaudeCLIEventHandler = Callable[[dict[str, Any]], Awaitable[None]]


def claude_session_path(workspace: str | Path, session_id: str) -> Path:
    """Resolve Claude's canonical transcript using the same home as the child."""
    uuid.UUID(session_id)
    workspace = Path(workspace).expanduser().resolve()
    home = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()
    if not home.is_absolute():
        home = workspace / home
    project = re.sub(r"[^a-zA-Z0-9]", "-", str(workspace))
    return home / "projects" / project / f"{session_id}.jsonl"


class ClaudeCLIClient:
    """Claude Code CLI adapter with HTTPLLMClient-compatible chat surface."""

    provider = "claude_code"

    def __init__(
        self,
        *,
        model: str,
        effort: str | None = None,
        timeout_seconds: int = 900,
        min_call_gap_seconds: float = 2.0,
        claude_binary: str = "claude",
        permission_mode: str | None = None,
        settings: str | Path | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if min_call_gap_seconds < 0:
            raise ValueError("min_call_gap_seconds must be >= 0")

        resolved = shutil.which(claude_binary)
        if not resolved:
            raise RuntimeError(f"Claude binary not found: {claude_binary}")

        self.chat_model = model
        self.embed_model = None
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self._min_call_gap_seconds = min_call_gap_seconds
        self._claude_binary = resolved
        self.permission_mode = str(permission_mode or "").strip() or None
        self.settings = str(settings or "").strip() or None
        self._workspace = (
            Path(workspace).expanduser() if workspace else (Path.home() / ".cache" / "memu-claude-workspace")
        )
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._last_call_monotonic = 0.0
        self._last_payload_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
            "memu_claude_cli_last_payload",
            default=None,
        )

    async def _throttle(self) -> None:
        if self._min_call_gap_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_call_monotonic
        if elapsed < self._min_call_gap_seconds:
            await asyncio.sleep(self._min_call_gap_seconds - elapsed)

    async def chat(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        session_id: str | None = None,
        resume_session_id: str | None = None,
        on_event: ClaudeCLIEventHandler | None = None,
        cli_tools: ClaudeCLITools | None = None,
        workspace: str | Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        session_id_clean = str(session_id or "").strip() or None
        resume_session_id_clean = str(resume_session_id or "").strip() or None
        if session_id_clean and resume_session_id_clean:
            raise ValueError("session_id and resume_session_id are mutually exclusive")
        if on_event is not None and cli_tools is None:
            raise ValueError("CLI events require an explicit cli_tools policy")
        if cli_tools is not None:
            if workspace is None or not Path(workspace).expanduser().is_absolute():
                raise ValueError("Scoped CLI calls require an explicit absolute workspace")
            if not (session_id_clean or resume_session_id_clean):
                raise ValueError("Scoped CLI calls require an explicit session")
            if session_id_clean and not system_prompt:
                raise ValueError("A new scoped CLI session requires a system prompt")
            if resume_session_id_clean and system_prompt:
                raise ValueError("Scoped CLI resume uses its original system prompt")
        await self._throttle()
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "provider": self.provider,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ],
            "effort": self.effort,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
            "session_id": session_id_clean,
            "resume_session_id": resume_session_id_clean,
        }
        self._last_payload_var.set(copy.deepcopy(payload))

        run_kwargs: dict[str, Any] = {}
        runner = self._run_claude
        if cli_tools is not None:
            runner = self._run_claude_stream
            run_kwargs.update(on_event=on_event, cli_tools=cli_tools)
            run_kwargs["workspace"] = workspace
        elif workspace is not None:
            raise ValueError("Per-call workspace requires cli_tools")
        text, raw = await runner(
            prompt=prompt,
            system_prompt=system_prompt or "",
            session_id=session_id_clean,
            resume_session_id=resume_session_id_clean,
            **run_kwargs,
        )
        self._last_call_monotonic = time.monotonic()
        return text, raw

    async def _run_claude(
        self,
        *,
        prompt: str,
        system_prompt: str,
        session_id: str | None = None,
        resume_session_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        cwd, prompt_dir = self._call_paths(None)
        system_prompt_file = prompt_dir / f"system-prompt-{uuid.uuid4().hex}.txt"
        try:
            system_prompt_file.write_text(system_prompt, encoding="utf-8")
            cmd = self._command(system_prompt_file, session_id, resume_session_id)
            cmd.extend(["--safe-mode", "--output-format", "json"])
            if resume_session_id:
                cmd.extend(["--system-prompt-snapshot", "off"])
            if self.permission_mode:
                cmd.extend(["--permission-mode", self.permission_mode])
            if self.settings:
                cmd.extend(["--settings", self.settings])
            # Match subprocess.run(text=True): locale encoding and newline translation.
            encoding = locale.getpreferredencoding(False)
            input_bytes = prompt.replace("\n", os.linesep).encode(encoding)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_bytes), timeout=self.timeout_seconds,
                )
            except (asyncio.CancelledError, TimeoutError) as exc:
                await self._kill_and_reap(process)
                if isinstance(exc, TimeoutError):
                    raise TimeoutError(
                        f"Claude CLI timed out after {self.timeout_seconds}s for model={self.chat_model}"
                    ) from exc
                raise
            stdout, stderr = (
                data.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
                for data in (stdout, stderr)
            )
            completed = subprocess.CompletedProcess(cmd, process.returncode, stdout=stdout, stderr=stderr)
        finally:
            system_prompt_file.unlink(missing_ok=True)

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            details = stderr or stdout or "(no subprocess output)"
            raise RuntimeError(
                f"Claude CLI failed (exit {completed.returncode}) for model={self.chat_model}: {details}"
            )
        response_text = completed.stdout
        if not response_text or not response_text.strip():
            raise RuntimeError(f"Claude CLI returned empty output for model={self.chat_model}")
        try:
            envelope = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Claude CLI JSON output was not valid JSON") from exc
        if not isinstance(envelope, dict) or envelope.get("type") != "result":
            raise RuntimeError("Claude CLI JSON output missing result envelope")
        return self._result(envelope, completed.returncode, session_id, resume_session_id)

    def _call_paths(self, workspace: str | Path | None) -> tuple[Path, Path]:
        cwd = (Path(workspace).expanduser() if workspace is not None else self._workspace).resolve()
        prompt_dir = cwd / ".prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        return cwd, prompt_dir

    def _command(self, prompt_file: Path | None, session_id: str | None, resume_session_id: str | None) -> list[str]:
        cmd = [self._claude_binary, "-p", "--model", self.chat_model]
        if prompt_file is not None:
            cmd.extend(["--system-prompt-file", str(prompt_file)])
        if session_id:
            cmd.extend(["--session-id", session_id])
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        if self.effort:
            cmd.extend(["--effort", self.effort])
        return cmd

    @staticmethod
    async def _kill_and_reap(process: asyncio.subprocess.Process, *, process_group: bool = False) -> None:
        if process_group and os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        # Pipe inheritance must not turn timeout cleanup into an unbounded wait.
        try:
            await asyncio.wait_for(process.communicate(), 5)
        except TimeoutError:
            process._transport.close()
            await process.wait()

    def _result(
        self, envelope: dict[str, Any], exit_code: int,
        session_id: str | None, resume_session_id: str | None,
    ) -> tuple[str, dict[str, Any]]:
        if envelope.get("subtype") != "success" or envelope.get("is_error") is not False:
            raise RuntimeError(
                f"Claude CLI failed for model={self.chat_model} "
                f"(subtype={envelope.get('subtype')!r}, is_error={envelope.get('is_error')!r}): "
                f"{envelope.get('errors') or '(no error details)'}"
            )
        if envelope.get("stop_reason") == "max_tokens":
            raise RuntimeError("Claude CLI response truncated at the output token limit")
        if envelope.get("stop_reason") == "refusal":
            raise RuntimeError("Claude CLI response refused by provider")
        if not isinstance(envelope.get("result"), str):
            raise RuntimeError("Claude CLI JSON output missing string result field")
        response_text = envelope["result"]
        if not response_text.strip():
            raise RuntimeError(f"Claude CLI returned empty result for model={self.chat_model}")

        raw = {
            "provider": self.provider,
            "model": self.chat_model,
            "effort": self.effort,
            "exit_code": exit_code,
            "session_id": session_id,
            "resume_session_id": resume_session_id,
            "usage": envelope.get("usage"),
            "stop_reason": envelope.get("stop_reason"),
        }
        return response_text, raw

    async def _run_claude_stream(
        self, *, prompt: str, system_prompt: str,
        session_id: str | None, resume_session_id: str | None,
        cli_tools: ClaudeCLITools, on_event: ClaudeCLIEventHandler | None,
        workspace: str | Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        tools = copy.deepcopy(cli_tools)
        servers, allowed = tools["mcp_servers"], tools["allowed_tools"]
        if not isinstance(servers, dict) or not isinstance(allowed, list) or any(
            not isinstance(name, str) or not re.fullmatch(r"mcp__[A-Za-z0-9_-]+__[A-Za-z0-9_-]+", name)
            or not any(name.startswith(f"mcp__{server}__") for server in servers)
            for name in allowed
        ):
            raise ValueError("cli_tools requires MCP servers and exact MCP tool names")
        cwd, prompt_dir = self._call_paths(workspace)
        prompt_file = prompt_dir / f"system-prompt-{uuid.uuid4().hex}.txt" if session_id else None
        mcp_file = prompt_dir / f"mcp-{uuid.uuid4().hex}.json"
        process = None
        tasks: list[asyncio.Task[Any]] = []
        terminal: dict[str, Any] | None = None
        stderr_tail = bytearray()
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        secrets = set()
        for server in servers.values():
            for name, value in server.get("headers", {}).items():
                if name.lower() in {"authorization", "proxy-authorization", "x-api-key"} and isinstance(value, str):
                    secrets.update((value, value.split(" ", 1)[-1]))

        def redact(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, str):
                for secret in sorted(secrets, key=len, reverse=True):
                    if secret:
                        value = value.replace(secret, "[redacted]")
            return value

        async def emit(event: dict[str, Any]) -> None:
            if on_event is not None:
                await on_event(event)

        async def write_input() -> None:
            try:
                await emit({"type": "process", "state": "input_started", "pid": process.pid})
                process.stdin.write(prompt.encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                # Report the child's launch error, not the closed input pipe.
                pass

        async def drain_stderr() -> None:
            while chunk := await process.stderr.read(65536):
                stderr_tail.extend(chunk)
                del stderr_tail[:-65536]

        async def read_events() -> None:
            nonlocal terminal
            message_id: str | None = None
            usage: dict[str, Any] = {}
            while True:
                try:
                    line = await process.stdout.readline()
                except ValueError as exc:
                    raise RuntimeError("Claude CLI event exceeds the 8 MiB line limit") from exc
                if not line:
                    return
                if not line.strip():
                    continue
                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    details = redact(stderr_tail.decode("utf-8", errors="replace").strip())
                    raise RuntimeError(
                        "Claude CLI stream output was not valid UTF-8 JSON" + (f": {details}" if details else "")
                    ) from exc
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise RuntimeError("Claude CLI stream event missing type")
                if event["type"] == "result":
                    if terminal is not None:
                        raise RuntimeError("Claude CLI emitted multiple result envelopes")
                    terminal = event
                    continue
                await emit(redact(event))
                if event["type"] != "stream_event" or event.get("parent_tool_use_id") is not None:
                    continue
                delta = event.get("event", {})
                if delta.get("type") == "message_start":
                    message = delta["message"]
                    message_id = message["id"]
                    usage = dict(message.get("usage") or {})
                elif delta.get("type") == "message_delta" and message_id is not None:
                    usage.update(delta.get("usage") or {})
                elif delta.get("type") == "message_stop" and message_id is not None:
                    await emit({"type": "context_usage", "message_id": message_id, "usage": dict(usage)})
                    message_id = None

        try:
            if prompt_file is not None:
                prompt_file.write_text(system_prompt, encoding="utf-8")
            # Credentials stay in an owner-only file, never argv or prompt logs.
            fd = os.open(mcp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({"mcpServers": servers}, output)
            cmd = self._command(prompt_file, session_id, resume_session_id)
            cmd.extend([
                "--restricted", "--permission-mode", "default", "--tools", "",
                "--strict-mcp-config", "--mcp-config", str(mcp_file),
                "--allowedTools", ",".join(allowed), "--setting-sources", "",
                "--settings", "{}", "--system-prompt-snapshot", "on",
                "--output-format", "stream-json", "--include-partial-messages", "--verbose",
                "--max-turns", "10",
            ])
            # Provisional API-loop parity; native round semantics remain a D check.
            process = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, env={**os.environ, "DISABLE_AUTO_COMPACT": "1"},
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=8 * 1024 * 1024,
                start_new_session=os.name == "posix",
            )
            # The owner persists child identity before we deliver any input.
            await emit({"type": "process", "state": "started", "pid": process.pid})
            tasks = [asyncio.create_task(fn()) for fn in (write_input, read_events, drain_stderr, process.wait)]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), max(0, deadline - asyncio.get_running_loop().time()))
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Claude CLI timed out after {self.timeout_seconds}s for model={self.chat_model}"
                ) from exc
            if process.returncode != 0:
                details = stderr_tail.decode("utf-8", errors="replace").strip()
                if terminal is not None:
                    details = f"{terminal.get('subtype')}: {terminal.get('errors') or terminal.get('result') or ''}\n{details}"
                details = redact(details.strip()) or "(no subprocess output)"
                raise RuntimeError(f"Claude CLI failed (exit {process.returncode}) for model={self.chat_model}: {details}")
            if terminal is None:
                raise RuntimeError("Claude CLI stream missing result envelope")
            result, raw = self._result(redact(terminal), process.returncode, session_id, resume_session_id)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                if process is not None:
                    await self._kill_and_reap(process, process_group=True)
                    await emit({"type": "process", "state": "reaped", "pid": process.pid})
            finally:
                if prompt_file is not None:
                    prompt_file.unlink(missing_ok=True)
                mcp_file.unlink(missing_ok=True)
        # This in-process hook hands off a completed result, not another model
        # operation. B2 queues delivery; do not relabel success as model timeout.
        await emit(redact(terminal))
        return result, raw

    async def embed(self, inputs: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        raise NotImplementedError("ClaudeCLIClient does not support embeddings")

    async def transcribe(
        self,
        audio_path: str,
        *,
        prompt: str | None = None,
        language: str | None = None,
        response_format: str = "text",
    ) -> tuple[str, dict[str, Any] | None]:
        raise NotImplementedError("claude_code backend has no audio transcription")

    def get_last_payload(self) -> dict[str, Any] | None:
        payload = self._last_payload_var.get()
        if not isinstance(payload, dict):
            return None
        return copy.deepcopy(payload)
