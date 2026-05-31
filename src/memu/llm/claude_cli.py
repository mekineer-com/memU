from __future__ import annotations

import asyncio
import copy
import shutil
import subprocess
import tempfile
import time
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
        sandbox_root: str | Path | None = None,
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
        self._sandbox_root = Path(sandbox_root) if sandbox_root else (Path.home() / ".cache" / "memu-claude-sandbox")
        self._sandbox_root.mkdir(parents=True, exist_ok=True)
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
    ) -> tuple[str, dict[str, Any]]:
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
        }
        self._last_payload = copy.deepcopy(payload)

        text, raw = await asyncio.to_thread(
            self._run_claude,
            prompt=prompt,
            system_prompt=system_prompt or "",
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

    def _run_claude(self, *, prompt: str, system_prompt: str) -> tuple[str, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="run-", dir=str(self._sandbox_root)) as run_dir:
            system_prompt_file = Path(run_dir) / "system_prompt.txt"
            system_prompt_file.write_text(system_prompt, encoding="utf-8")
            cmd = [
                self._claude_binary,
                "-p",
                "--model",
                self.chat_model,
                "--system-prompt-file",
                str(system_prompt_file),
            ]
            if self.effort:
                cmd.extend(["--effort", self.effort])
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=run_dir,
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
            "usage": None,
        }
        return response_text, raw

    async def embed(self, inputs: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        raise NotImplementedError("ClaudeCLIClient does not support embeddings")

    def get_last_payload(self) -> dict[str, Any] | None:
        if not isinstance(self._last_payload, dict):
            return None
        return copy.deepcopy(self._last_payload)
