from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from memu.workflow.step import WorkflowStep


@dataclass
class PipelineRevision:
    name: str
    revision: int
    steps: list[WorkflowStep]
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineManager:
    def __init__(self, *, available_capabilities: set[str] | None = None, llm_profiles: set[str] | None = None):
        self.available_capabilities = available_capabilities or set()
        self.llm_profiles = llm_profiles or {"default"}
        self._pipelines: dict[str, list[PipelineRevision]] = {}

    def register(
        self,
        name: str,
        steps: Iterable[WorkflowStep],
        *,
        initial_state_keys: set[str] | None = None,
    ) -> None:
        steps_list = list(steps)
        meta = {"initial_state_keys": set(initial_state_keys or set())}
        self._validate_steps(steps_list, initial_state_keys=meta["initial_state_keys"])
        self._pipelines[name] = [
            PipelineRevision(
                name=name,
                revision=1,
                steps=steps_list,
                created_at=time.time(),
                metadata=meta,
            )
        ]

    def build(self, name: str) -> list[WorkflowStep]:
        revision = self._current_revision(name)
        return [step.copy() for step in revision.steps]

    def _current_revision(self, name: str) -> PipelineRevision:
        revisions = self._pipelines.get(name)
        if not revisions:
            msg = f"Pipeline '{name}' not registered"
            raise KeyError(msg)
        return revisions[-1]

    def _validate_steps(self, steps: list[WorkflowStep], *, initial_state_keys: set[str] | None) -> None:
        seen: set[str] = set()
        available_keys = set(initial_state_keys or set())

        for step in steps:
            if step.step_id in seen:
                msg = f"Duplicate step_id '{step.step_id}' found"
                raise ValueError(msg)
            seen.add(step.step_id)

            if self.available_capabilities:
                unknown_caps = step.capabilities - self.available_capabilities
                if unknown_caps:
                    msg = f"Step '{step.step_id}' requests unavailable capabilities: {', '.join(sorted(unknown_caps))}"
                    raise ValueError(msg)

            if getattr(step, "config", None):
                profile_name = step.config.get("llm_profile")
                if profile_name and profile_name not in self.llm_profiles:
                    msg = (
                        f"Step '{step.step_id}' references unknown llm_profile '{profile_name}'. "
                        f"Available profiles: {', '.join(sorted(self.llm_profiles))}"
                    )
                    raise ValueError(msg)

            missing = step.requires - available_keys
            if missing:
                msg = (
                    f"Step '{step.step_id}' requires missing state keys: {', '.join(sorted(missing))}. "
                    "Ensure previous steps produce them or initial_state_keys contains them."
                )
                raise ValueError(msg)

            available_keys.update(step.produces)

    def revision_token(self) -> str:
        parts: list[str] = []
        for name, revisions in sorted(self._pipelines.items()):
            parts.append(f"{name}:v{revisions[-1].revision}")
        return "|".join(parts)
