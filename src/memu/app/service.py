from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from memu.app.memorize import MemorizeMixin
from memu.app.retrieve import RetrieveMixin
from memu.app.settings import (
    BlobConfig,
    CategoryConfig,
    DatabaseConfig,
    LLMConfig,
    LLMProfilesConfig,
    MemorizeConfig,
    RetrieveConfig,
    UserConfig,
)
from memu.blob.local_fs import LocalFS
from memu.database.factory import build_database
from memu.database.interfaces import Database
from memu.llm.claude_cli import ClaudeCLIClient
from memu.llm.http_client import HTTPLLMClient
from memu.llm.wrapper import (
    LLMCallMetadata,
    LLMClientWrapper,
    LLMInterceptorHandle,
    LLMInterceptorRegistry,
)
from memu.workflow.pipeline import PipelineManager
from memu.workflow.runner import WorkflowRunner, resolve_workflow_runner
from memu.workflow.step import WorkflowState, WorkflowStep

TConfigModel = TypeVar("TConfigModel", bound=BaseModel)


@dataclass
class Context:
    categories_ready: bool = False
    category_ids: list[str] = field(default_factory=list)
    category_name_to_id: dict[str, str] = field(default_factory=dict)
    category_scope_key: str | None = None
    _init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MemoryService(MemorizeMixin, RetrieveMixin):
    def __init__(
        self,
        *,
        llm_profiles: LLMProfilesConfig | dict[str, Any] | None = None,
        blob_config: BlobConfig | dict[str, Any] | None = None,
        database_config: DatabaseConfig | dict[str, Any] | None = None,
        memorize_config: MemorizeConfig | dict[str, Any] | None = None,
        retrieve_config: RetrieveConfig | dict[str, Any] | None = None,
        workflow_runner: WorkflowRunner | str | None = None,
        user_config: UserConfig | dict[str, Any] | None = None,
        claude_code: bool = False,
        claude_code_model: str = "claude-opus-4-7",
        claude_code_effort: str = "medium",
        claude_code_permission_mode: str | None = None,
        claude_code_settings: str | None = None,
        claude_code_workspace: str | None = None,
        claude_code_timeout_seconds: int = 900,
    ):
        self.llm_profiles = self._validate_config(llm_profiles, LLMProfilesConfig)
        self.user_config = self._validate_config(user_config, UserConfig)
        self.user_model = self.user_config.model
        self.llm_config = self._validate_config(self.llm_profiles.default, LLMConfig)
        self.blob_config = self._validate_config(blob_config, BlobConfig)
        self.database_config = self._validate_config(database_config, DatabaseConfig)
        self.memorize_config = self._validate_config(memorize_config, MemorizeConfig)
        self.retrieve_config = self._validate_config(retrieve_config, RetrieveConfig)
        self._claude_code = bool(claude_code)
        self._claude_code_model = str(claude_code_model or "claude-opus-4-7").strip() or "claude-opus-4-7"
        self._claude_code_effort = str(claude_code_effort or "").strip() or None
        self._claude_code_permission_mode = str(claude_code_permission_mode or "").strip() or None
        self._claude_code_settings = str(claude_code_settings or "").strip() or None
        self._claude_code_workspace = str(claude_code_workspace or "").strip() or None
        self._claude_code_internal_workspace = Path.home() / ".cache" / "memu-claude-internal"
        self._claude_code_timeout_seconds = int(claude_code_timeout_seconds)

        self.fs = LocalFS(self.blob_config.resources_dir)
        self.category_configs: list[CategoryConfig] = list(self.memorize_config.memory_categories or [])
        self.category_config_map: dict[str, CategoryConfig] = {cfg.name: cfg for cfg in self.category_configs}
        self._category_prompt_str = self._format_categories_for_prompt(self.category_configs)

        self._context = Context(categories_ready=not bool(self.category_configs))
        self._category_summary_embedding_cache: dict[str, tuple[str, list[float]]] = {}

        self.database: Database = build_database(
            config=self.database_config,
            user_model=self.user_model,
        )

        # Initialize client caches (lazy creation on first use)
        self._llm_clients: dict[str, Any] = {}
        self._claude_cli_client: ClaudeCLIClient | None = None
        self._claude_cli_internal_client: ClaudeCLIClient | None = None
        self._llm_interceptors = LLMInterceptorRegistry()

        self._workflow_runner = resolve_workflow_runner(workflow_runner)

        self._pipelines = PipelineManager(
            available_capabilities={"llm", "vector", "db", "io", "vision"},
            llm_profiles=set(self.llm_profiles.profiles.keys()),
        )
        self._register_pipelines()

    def _init_llm_client(self, config: LLMConfig | None = None) -> Any:
        cfg = config or self.llm_config
        return HTTPLLMClient(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            chat_model=cfg.chat_model,
            provider=cfg.provider,
            endpoint_overrides=cfg.endpoint_overrides,
            embed_model=cfg.embed_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def _get_llm_base_client(self, profile: str | None = None) -> Any:
        """
        Lazily initialize and cache LLM clients per profile to avoid eager network setup.
        """
        name = profile or "default"
        client = self._llm_clients.get(name)
        if client is not None:
            return client
        cfg: LLMConfig | None = self.llm_profiles.profiles.get(name)
        if cfg is None:
            msg = f"Step profile '{name}' not found in config"
            raise KeyError(msg)
        client = self._init_llm_client(cfg)
        self._llm_clients[name] = client
        return client

    @staticmethod
    def _llm_call_metadata(profile: str, step_context: Mapping[str, Any] | None) -> LLMCallMetadata:
        if not isinstance(step_context, Mapping):
            return LLMCallMetadata(profile)
        operation = None
        for key in ("operation", "workflow_name"):
            value = step_context.get(key)
            if isinstance(value, str) and value.strip():
                operation = value.strip()
                break
        step_id = step_context.get("step_id") if isinstance(step_context.get("step_id"), str) else None
        trace_id = step_context.get("trace_id") if isinstance(step_context.get("trace_id"), str) else None
        tags = step_context.get("tags") if isinstance(step_context.get("tags"), Mapping) else None
        return LLMCallMetadata(profile=profile, operation=operation, step_id=step_id, trace_id=trace_id, tags=tags)

    def _wrap_llm_client(
        self,
        client: Any,
        *,
        profile: str | None = None,
        step_context: Mapping[str, Any] | None = None,
    ) -> Any:
        cfg: LLMConfig | None = self.llm_profiles.profiles.get(profile or "default")
        provider = cfg.provider if cfg is not None else getattr(client, "provider", None)
        metadata = self._llm_call_metadata(profile or "default", step_context)
        return LLMClientWrapper(
            client,
            registry=self._llm_interceptors,
            metadata=metadata,
            provider=provider,
            chat_model=getattr(client, "chat_model", None),
            embed_model=getattr(client, "embed_model", None),
        )

    def _get_llm_client(self, profile: str | None = None, step_context: Mapping[str, Any] | None = None) -> Any:
        base_client = self._get_llm_base_client(profile)
        return self._wrap_llm_client(base_client, profile=profile, step_context=step_context)

    @property
    def llm_client(self) -> Any:
        """Default LLM client (lazy)."""
        return self._get_llm_client()

    async def chat(
        self,
        prompt: str,
        *,
        profile: str | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        op: str | None = None,
        step: str | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
        resume_session_id: str | None = None,
    ) -> Any:
        session_id_clean = str(session_id or "").strip() or None
        resume_session_id_clean = str(resume_session_id or "").strip() or None
        if session_id_clean and resume_session_id_clean:
            raise ValueError("session_id and resume_session_id are mutually exclusive")
        if (session_id_clean or resume_session_id_clean) and not self._claude_code:
            raise ValueError("Claude session arguments require claude_code=True")
        trace_id_clean = str(trace_id or "").strip()
        step_context = {"operation": op, "step_id": step} if (op or step or trace_id_clean) else None
        if step_context is not None and trace_id_clean:
            step_context["trace_id"] = trace_id_clean
        if self._claude_code:
            claude_client = (
                self._get_claude_cli_client()
                if (session_id_clean or resume_session_id_clean)
                else self._get_claude_cli_internal_client()
            )
            client = self._wrap_llm_client(
                claude_client,
                profile="claude_code",
                step_context=step_context,
            )
        else:
            client = self._get_llm_client(profile, step_context=step_context)
        return await client.chat(
            prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format=response_format,
            session_id=session_id_clean,
            resume_session_id=resume_session_id_clean,
        )

    async def embed(self, texts: list[str], *, profile: str | None = None) -> Any:
        return await self._get_llm_client(profile).embed(texts)

    @staticmethod
    def _llm_profile_from_context(
        step_context: Mapping[str, Any] | None, task: Literal["chat", "embedding"] = "chat"
    ) -> str | None:
        if not isinstance(step_context, Mapping):
            return None
        step_cfg = step_context.get("step_config")
        if not isinstance(step_cfg, Mapping):
            return None
        if task == "chat":
            profile = step_cfg.get("chat_llm_profile", step_cfg.get("llm_profile"))
        elif task == "embedding":
            profile = step_cfg.get("embed_llm_profile", step_cfg.get("llm_profile"))
        else:
            raise ValueError(task)
        if isinstance(profile, str) and profile.strip():
            return profile.strip()
        return None

    def _get_step_llm_client(self, step_context: Mapping[str, Any] | None) -> Any:
        if self._claude_code:
            return self._wrap_llm_client(
                self._get_claude_cli_internal_client(),
                profile="claude_code",
                step_context=step_context,
            )
        profile = self._llm_profile_from_context(step_context, task="chat") or "default"
        return self._get_llm_client(profile, step_context=step_context)

    def _get_step_embedding_client(self, step_context: Mapping[str, Any] | None) -> Any:
        profile = self._llm_profile_from_context(step_context, task="embedding") or "embedding"
        return self._get_llm_client(profile, step_context=step_context)

    def _get_claude_cli_client(self) -> ClaudeCLIClient:
        if self._claude_cli_client is None:
            self._claude_cli_client = ClaudeCLIClient(
                model=self._claude_code_model,
                effort=self._claude_code_effort,
                permission_mode=self._claude_code_permission_mode,
                settings=self._claude_code_settings,
                workspace=self._claude_code_workspace,
                timeout_seconds=self._claude_code_timeout_seconds,
            )
        return self._claude_cli_client

    def _get_claude_cli_internal_client(self) -> ClaudeCLIClient:
        if self._claude_cli_internal_client is None:
            self._claude_cli_internal_client = ClaudeCLIClient(
                model=self._claude_code_model,
                effort=self._claude_code_effort,
                permission_mode=self._claude_code_permission_mode,
                settings=self._claude_code_settings,
                workspace=self._claude_code_internal_workspace,
                timeout_seconds=self._claude_code_timeout_seconds,
            )
        return self._claude_cli_internal_client

    def intercept_before_llm_call(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        priority: int = 0,
        where: Mapping[str, Any] | Callable[..., Any] | None = None,
    ) -> LLMInterceptorHandle:
        return self._llm_interceptors.register_before(fn, name=name, priority=priority, where=where)

    def intercept_after_llm_call(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        priority: int = 0,
        where: Mapping[str, Any] | Callable[..., Any] | None = None,
    ) -> LLMInterceptorHandle:
        return self._llm_interceptors.register_after(fn, name=name, priority=priority, where=where)

    def intercept_on_error_llm_call(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        priority: int = 0,
        where: Mapping[str, Any] | Callable[..., Any] | None = None,
    ) -> LLMInterceptorHandle:
        return self._llm_interceptors.register_on_error(fn, name=name, priority=priority, where=where)

    def _get_context(self) -> Context:
        return self._context

    def _get_database(self) -> Database:
        return self.database

    def _register_pipelines(self) -> None:
        memo_workflow = self._build_memorize_workflow()
        memo_initial_keys = self._list_memorize_initial_keys()
        self._pipelines.register("memorize", memo_workflow, initial_state_keys=memo_initial_keys)
        rag_workflow = self._build_rag_retrieve_workflow()
        retrieve_initial_keys = self._list_retrieve_initial_keys()
        self._pipelines.register("retrieve_rag", rag_workflow, initial_state_keys=retrieve_initial_keys)

    async def _run_workflow(self, workflow_name: str, initial_state: WorkflowState) -> WorkflowState:
        """Execute a workflow through the configured runner backend."""
        steps = self._pipelines.build(workflow_name)
        runner_context = {"workflow_name": workflow_name}
        trace_id = initial_state.get("trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            runner_context["trace_id"] = trace_id.strip()
        return await self._workflow_runner.run(
            workflow_name,
            steps,
            initial_state,
            runner_context,
        )

    @staticmethod
    def _extract_json_blob(raw: str) -> str:
        start_obj = raw.find("{")
        start_arr = raw.find("[")
        candidates: list[tuple[int, str]] = []
        if start_obj != -1:
            candidates.append((start_obj, "}"))
        if start_arr != -1:
            candidates.append((start_arr, "]"))
        if not candidates:
            msg = "No JSON object or array found"
            raise ValueError(msg)
        start, closing = min(candidates, key=lambda pair: pair[0])
        end = raw.rfind(closing)
        if end == -1 or end <= start:
            msg = "No complete JSON object or array found"
            raise ValueError(msg)
        return raw[start : end + 1]

    @staticmethod
    def _escape_prompt_value(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}").replace("<", "&lt;").replace(">", "&gt;")

    def _model_dump_without_embeddings(self, obj: BaseModel) -> dict[str, Any]:
        data = obj.model_dump(exclude={"embedding"})
        return data

    @staticmethod
    def _validate_config(
        config: Mapping[str, Any] | BaseModel | None,
        model_type: type[TConfigModel],
    ) -> TConfigModel:
        if isinstance(config, model_type):
            return config
        if config is None:
            return model_type()
        return model_type.model_validate(config)

    def configure_pipeline(self, *, step_id: str, configs: Mapping[str, Any], pipeline: str = "memorize") -> int:
        revision = self._pipelines.config_step(pipeline, step_id, dict(configs))
        return revision

    def insert_step_after(
        self,
        *,
        target_step_id: str,
        new_step: WorkflowStep,
        pipeline: str = "memorize",
    ) -> int:
        revision = self._pipelines.insert_after(pipeline, target_step_id, new_step)
        return revision

    def insert_step_before(
        self,
        *,
        target_step_id: str,
        new_step: WorkflowStep,
        pipeline: str = "memorize",
    ) -> int:
        revision = self._pipelines.insert_before(pipeline, target_step_id, new_step)
        return revision

    def replace_step(
        self,
        *,
        target_step_id: str,
        new_step: WorkflowStep,
        pipeline: str = "memorize",
    ) -> int:
        revision = self._pipelines.replace_step(pipeline, target_step_id, new_step)
        return revision

    def remove_step(self, *, target_step_id: str, pipeline: str = "memorize") -> int:
        revision = self._pipelines.remove_step(pipeline, target_step_id)
        return revision
