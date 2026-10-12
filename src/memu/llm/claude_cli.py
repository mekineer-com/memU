from __future__ import annotations

import asyncio
import copy
import contextvars
import json
import locale
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


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
        self._prompt_dir = self._workspace / ".prompts"
        self._prompt_dir.mkdir(parents=True, exist_ok=True)
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
    ) -> tuple[str, dict[str, Any]]:
        session_id_clean = str(session_id or "").strip() or None
        resume_session_id_clean = str(resume_session_id or "").strip() or None
        if session_id_clean and resume_session_id_clean:
            raise ValueError("session_id and resume_session_id are mutually exclusive")
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

        text, raw = await self._run_claude(
            prompt=prompt,
            system_prompt=system_prompt or "",
            session_id=session_id_clean,
            resume_session_id=resume_session_id_clean,
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
        system_prompt_file = self._prompt_dir / f"system-prompt-{uuid.uuid4().hex}.txt"
        system_prompt_file.write_text(system_prompt, encoding="utf-8")
        try:
            cmd = [
                self._claude_binary,
                "-p",
                "--safe-mode",
                "--model",
                self.chat_model,
                "--system-prompt-file",
                str(system_prompt_file),
                "--output-format",
                "json",
            ]
            if session_id:
                cmd.extend(["--session-id", session_id])
            if resume_session_id:
                cmd.extend(["--resume", resume_session_id, "--system-prompt-snapshot", "off"])
            if self.effort:
                cmd.extend(["--effort", self.effort])
            if self.permission_mode:
                cmd.extend(["--permission-mode", self.permission_mode])
            if self.settings:
                cmd.extend(["--settings", self.settings])
            # Match subprocess.run(text=True): locale encoding and newline translation.
            encoding = locale.getpreferredencoding(False)
            input_bytes = prompt.replace("\n", os.linesep).encode(encoding)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self._workspace,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_bytes), timeout=self.timeout_seconds,
                )
            except (asyncio.CancelledError, TimeoutError) as exc:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.communicate()
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
            "exit_code": completed.returncode,
            "session_id": session_id,
            "resume_session_id": resume_session_id,
            "usage": envelope.get("usage"),
            "stop_reason": envelope.get("stop_reason"),
        }
        return response_text, raw

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
