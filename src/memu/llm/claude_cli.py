from __future__ import annotations

import asyncio
import copy
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
        self._last_payload: dict[str, Any] | None = None

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
        self._last_payload = copy.deepcopy(payload)

        text, raw = await asyncio.to_thread(
            self._run_claude,
            prompt=prompt,
            system_prompt=system_prompt or "",
            session_id=session_id_clean,
            resume_session_id=resume_session_id_clean,
        )
        self._last_call_monotonic = time.monotonic()
        return text, raw

    async def summarize(
        self,
        text: str,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        return await self.chat(
            text,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=None,
            response_format=None,
        )

    def _run_claude(
        self,
        *,
        prompt: str,
        system_prompt: str,
        session_id: str | None = None,
        resume_session_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if session_id and resume_session_id:
            raise ValueError("session_id and resume_session_id are mutually exclusive")
        system_prompt_file = self._prompt_dir / f"system-prompt-{uuid.uuid4().hex}.txt"
        system_prompt_file.write_text(system_prompt, encoding="utf-8")
        try:
            cmd = [
                self._claude_binary,
                "-p",
                "--model",
                self.chat_model,
                "--system-prompt-file",
                str(system_prompt_file),
            ]
            if session_id:
                cmd.extend(["--session-id", session_id])
            if resume_session_id:
                cmd.extend(["--resume", resume_session_id])
            if self.effort:
                cmd.extend(["--effort", self.effort])
            if self.permission_mode:
                cmd.extend(["--permission-mode", self.permission_mode])
            if self.settings:
                cmd.extend(["--settings", self.settings])
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=self._workspace,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"Claude CLI timed out after {self.timeout_seconds}s for model={self.chat_model}"
                ) from exc
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

        raw = {
            "provider": self.provider,
            "model": self.chat_model,
            "effort": self.effort,
            "exit_code": completed.returncode,
            "session_id": session_id,
            "resume_session_id": resume_session_id,
            "usage": None,
        }
        return response_text, raw

    async def embed(self, inputs: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        raise NotImplementedError("ClaudeCLIClient does not support embeddings")

    def get_last_payload(self) -> dict[str, Any] | None:
        if not isinstance(self._last_payload, dict):
            return None
        return copy.deepcopy(self._last_payload)
