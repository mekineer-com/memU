from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, RootModel, StringConstraints, model_validator

from memu.prompts.memory_type import (
    DEFAULT_MEMORY_CUSTOM_PROMPT_ORDINAL,
    DEFAULT_MEMORY_TYPES,
)
from memu.prompts.memory_type import (
    PROMPTS as DEFAULT_MEMORY_TYPE_PROMPTS,
)


def normalize_value(v: str) -> str:
    if isinstance(v, str):
        return v.strip().lower()
    return v


Normalize = BeforeValidator(normalize_value)


def _default_memory_types() -> list[str]:
    return list(DEFAULT_MEMORY_TYPES)


def _default_memory_type_prompts() -> "dict[str, str | CustomPrompt]":
    return dict(DEFAULT_MEMORY_TYPE_PROMPTS)


class PromptBlock(BaseModel):
    label: str | None = None
    ordinal: int = Field(default=0)
    prompt: str | None = None


class CustomPrompt(RootModel[dict[str, PromptBlock]]):
    root: dict[str, PromptBlock] = Field(default_factory=dict)

    def get(self, key: str, default: PromptBlock | None = None) -> PromptBlock | None:
        return self.root.get(key, default)

    def items(self) -> list[tuple[str, PromptBlock]]:
        return list(self.root.items())


def complete_prompt_blocks(prompt: CustomPrompt, default_blocks: Mapping[str, int]) -> CustomPrompt:
    for key, ordinal in default_blocks.items():
        if key not in prompt.root:
            prompt.root[key] = PromptBlock(ordinal=ordinal)
    return prompt


CompleteMemoryTypePrompt = AfterValidator(lambda v: complete_prompt_blocks(v, DEFAULT_MEMORY_CUSTOM_PROMPT_ORDINAL))


class LLMConfig(BaseModel):
    provider: str = Field(
        default="openai",
        description="Identifier for the LLM provider implementation (used by HTTP client backend).",
    )
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="OPENAI_API_KEY")
    chat_model: str = Field(default="")
    endpoint_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Optional overrides for HTTP endpoints (keys: 'chat'/'summary').",
    )
    embed_model: str = Field(
        default="text-embedding-3-large",
        description="Default embedding model used for vectorization.",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature. None = provider default (usually 1.0).",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Default max output tokens for all calls using this profile. None = provider default.",
    )

    @model_validator(mode="after")
    def set_provider_defaults(self) -> "LLMConfig":
        if self.provider == "grok":
            if self.base_url == "https://api.openai.com/v1":
                self.base_url = "https://api.x.ai/v1"
            if self.api_key == "OPENAI_API_KEY":
                self.api_key = "XAI_API_KEY"
            if not self.chat_model:
                self.chat_model = "grok-2-latest"
        return self


class BlobConfig(BaseModel):
    provider: str = Field(default="local")
    resources_dir: str = Field(default="./data/resources")


class RetrieveCategoryConfig(BaseModel):
    top_k: int = Field(default=5, description="Total number of categories to retrieve.")
    max_count: int = Field(default=4, ge=1, le=4, description="Maximum dossiers returned after autocut.")


class RetrieveItemConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable item retrieval.")
    top_k: int = Field(default=5, description="Total number of items to retrieve.")
    # Salience-aware retrieval settings
    ranking: Literal["similarity", "salience"] = Field(
        default="salience",
        description="Ranking strategy: 'similarity' (cosine only) or 'salience' (weighted by reflection salience).",
    )
    recency_decay_days: float = Field(
        default=30.0,
        description="Half-life in days for recency decay in salience scoring. After this many days, recency factor is ~0.5.",
    )
    fts_enabled: bool = Field(
        default=True,
        description="Enable FTS5 BM25 keyword search alongside vector search, fused via RRF.",
    )
    fts_top_k: int = Field(
        default=20,
        description="Number of FTS5 candidates to retrieve for RRF fusion (should be >= top_k).",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant k (Cormack et al. 2009). Higher values reduce impact of high-ranked items.",
    )


class RetrieveResourceConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable resource retrieval.")
    top_k: int = Field(default=5, description="Total number of resources to retrieve.")


class RetrieveGraphConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable graph-based retrieval alongside vector search.")
    max_graph_results: int = Field(default=3, description="Max graph-expanded results per retrieval.")
    min_entity_similarity: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="Minimum query cosine similarity for memories reached through an entity mention.",
    )


class RetrieveConfig(BaseModel):
    method: Annotated[Literal["rag"], Normalize] = "rag"
    category: RetrieveCategoryConfig = Field(default=RetrieveCategoryConfig())
    item: RetrieveItemConfig = Field(default=RetrieveItemConfig())
    resource: RetrieveResourceConfig = Field(default=RetrieveResourceConfig())
    graph: RetrieveGraphConfig = Field(default=RetrieveGraphConfig())
    sufficiency_check_llm_profile: str = Field(default="default")


class MemorizeConfig(BaseModel):
    multimodal_preprocess_prompts: dict[str, str | CustomPrompt] = Field(
        default_factory=dict,
        description="Optional mapping of modality -> preprocess system prompt.",
    )
    preprocess_llm_profile: str = Field(default="default", description="LLM profile for preprocess.")
    memory_types: list[str] = Field(
        default_factory=_default_memory_types,
        description="Ordered list of memory types (profile/knowledge/behavior/social by default).",
    )
    memory_type_prompts: dict[str, str | Annotated[CustomPrompt, CompleteMemoryTypePrompt]] = Field(
        default_factory=_default_memory_type_prompts,
        description="User prompt overrides for each memory type extraction.",
    )
    memory_extract_llm_profile: str = Field(default="default", description="LLM profile for memory extract.")
    episodes_per_segment: int = Field(
        default=3,
        ge=1,
        description="Maximum number of episodes the extraction router can return for one conversation segment.",
    )
    min_chunk_tokens: int = Field(
        default=4000,
        description="Configured memorize chunk size used to set per-memory-type extraction targets.",
    )
    background_extra_messages_tokens: int = Field(
        default=100,
        description=(
            "During segment preprocessing, if the total token estimate across unsummarized background tails "
            "is below this threshold, skip background-tail summarization and keep raw lines."
        ),
    )
    enable_confidence_normalization: bool = Field(
        default=False,
        description="When true, redistribute clustered confidence scores via z-score rescaling.",
    )
    dynamic_category_cluster_size: int = Field(
        default=10,
        description="Minimum number of homeless items that must cluster together (by embedding similarity) before that cluster becomes a new dynamic category.",
    )
    active_dossiers_per_kind: int = Field(
        default=30,
        ge=0,
        description="Maximum active non-anchor dossiers retained per dossier kind.",
    )
    category_summary_target_words: int = Field(
        default=300,
        description="Target maximum word count for auto-generated category summaries.",
    )
    category_update_llm_profile: str = Field(default="default", description="LLM profile for category summary.")
    semantic_dedupe_enabled: bool = Field(
        default=True,
        description="Enable conservative post-persist semantic dedupe in memorize workflow.",
    )
    semantic_dedupe_similarity_threshold: float = Field(
        default=0.85,
        description="Cosine similarity threshold for semantic dedupe auto-merge decisions.",
    )

    @model_validator(mode="after")
    def reject_removed_conversation_preprocess_prompts(self) -> "MemorizeConfig":
        removed = {
            str(key).strip().lower()
            for key in self.multimodal_preprocess_prompts
            if str(key).strip().lower() in {"conversation", "cross_conversation"}
        }
        if removed:
            names = ", ".join(sorted(removed))
            msg = (
                f"Conversation preprocess prompt override(s) are no longer supported: {names}. "
                "Conversation segments are passed whole to extraction."
            )
            raise ValueError(msg)
        return self


class DefaultUserModel(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


class UserConfig(BaseModel):
    model: type[BaseModel] = Field(default=DefaultUserModel)


Key = Annotated[str, StringConstraints(min_length=1)]


class LLMProfilesConfig(RootModel[dict[Key, LLMConfig]]):
    root: dict[str, LLMConfig] = Field(default_factory=lambda: {"default": LLMConfig()})

    def get(self, key: str, default: LLMConfig | None = None) -> LLMConfig | None:
        return self.root.get(key, default)

    @model_validator(mode="before")
    @classmethod
    def ensure_default(cls, data: Any) -> Any:
        if data is None:
            data = {}
        elif isinstance(data, dict):
            data = dict(data)
        else:
            return data
        if "default" not in data:
            data["default"] = LLMConfig()
        if "embedding" not in data:
            data["embedding"] = data["default"]
        return data

    @property
    def profiles(self) -> dict[str, LLMConfig]:
        return self.root

    @property
    def default(self) -> LLMConfig:
        return self.root.get("default", LLMConfig())


class MetadataStoreConfig(BaseModel):
    provider: Annotated[Literal["sqlite"], Normalize] = "sqlite"
    ddl_mode: Annotated[Literal["create", "validate"], Normalize] = "create"
    dsn: str | None = Field(default=None, description="Database connection string for sqlite.")


class VectorIndexConfig(BaseModel):
    provider: Annotated[Literal["bruteforce", "none"], Normalize] = "bruteforce"
    dsn: str | None = Field(default=None, description="Reserved for future non-sqlite vector backends.")


class DatabaseConfig(BaseModel):
    metadata_store: MetadataStoreConfig = Field(default_factory=MetadataStoreConfig)
    vector_index: VectorIndexConfig | None = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.vector_index is None:
            self.vector_index = VectorIndexConfig(provider="bruteforce")
