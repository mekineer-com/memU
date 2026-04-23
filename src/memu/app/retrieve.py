from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from memu.database.vector import cosine_topk
from memu.prompts.retrieve.llm_category_ranker import PROMPT as LLM_CATEGORY_RANKER_PROMPT
from memu.prompts.retrieve.llm_item_ranker import PROMPT as LLM_ITEM_RANKER_PROMPT
from memu.prompts.retrieve.llm_resource_ranker import PROMPT as LLM_RESOURCE_RANKER_PROMPT
from memu.prompts.retrieve.pre_retrieval_decision import SYSTEM_PROMPT as PRE_RETRIEVAL_SYSTEM_PROMPT
from memu.prompts.retrieve.pre_retrieval_decision import USER_PROMPT as PRE_RETRIEVAL_USER_PROMPT
from memu.prompts.retrieve.pre_retrieval_decision import system_prompt_for_angle as _system_prompt_for_angle
from memu.workflow.step import WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from memu.app.service import Context
    from memu.app.settings import RetrieveConfig
    from memu.database.interfaces import Database


class RetrieveMixin:
    if TYPE_CHECKING:
        retrieve_config: RetrieveConfig
        _run_workflow: Callable[..., Awaitable[WorkflowState]]
        _get_context: Callable[[], Context]
        _get_database: Callable[[], Database]
        _ensure_categories_ready: Callable[[Context, Database], Awaitable[None]]
        _get_step_llm_client: Callable[[Mapping[str, Any] | None], Any]
        _get_step_embedding_client: Callable[[Mapping[str, Any] | None], Any]
        _get_llm_client: Callable[..., Any]
        _model_dump_without_embeddings: Callable[[BaseModel], dict[str, Any]]
        _extract_json_blob: Callable[[str], str]
        _escape_prompt_value: Callable[[str], str]
        _category_summary_embedding_cache: dict[str, tuple[str, list[float]]]
        user_model: type[BaseModel]

    async def retrieve(
        self,
        queries: list[dict[str, Any]],
        where: dict[str, Any] | None = None,
        as_of: datetime | None = None,
        rewrite_angle: int = 0,
    ) -> dict[str, Any]:
        if not queries:
            raise ValueError("empty_queries")
        ctx = self._get_context()
        store = self._get_database()
        original_query = self._extract_query_text(queries[-1])
        where_filters = self._normalize_where(where)

        context_queries_objs = queries[:-1] if len(queries) > 1 else []
        route_context_queries, downstream_context_queries = self._split_context_queries(context_queries_objs)

        route_intention = self.retrieve_config.route_intention
        retrieve_category = self.retrieve_config.category.enabled
        retrieve_item = self.retrieve_config.item.enabled
        retrieve_resource = self.retrieve_config.resource.enabled
        sufficiency_check = self.retrieve_config.sufficiency_check

        workflow_name = "retrieve_llm" if self.retrieve_config.method == "llm" else "retrieve_rag"

        state: WorkflowState = {
            "method": self.retrieve_config.method,
            "original_query": original_query,
            "context_queries": downstream_context_queries,
            "route_context_queries": route_context_queries,
            "route_intention": route_intention,
            "retrieve_category": retrieve_category,
            "retrieve_item": retrieve_item,
            "retrieve_resource": retrieve_resource,
            "sufficiency_check": sufficiency_check,
            "rewrite_angle": int(rewrite_angle) if rewrite_angle is not None else 0,
            "ctx": ctx,
            "store": store,
            "where": where_filters,
            "as_of": as_of,
        }

        result = await self._run_workflow(workflow_name, state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "Retrieve workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    def _normalize_where(self, where: Mapping[str, Any] | None) -> dict[str, Any]:
        if not where:
            return {}

        valid_fields = set(self.user_model.model_fields.keys())
        cleaned: dict[str, Any] = {}

        for raw_key, value in where.items():
            if value is None:
                continue
            field = raw_key.split("__", 1)[0]
            if field not in valid_fields:
                msg = f"Unknown filter field '{field}' for current user scope"
                raise ValueError(msg)
            cleaned[raw_key] = value

        return cleaned

    def _build_rag_retrieve_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="route_intention",
                role="route_intention",
                handler=self._rag_route_intention,
                requires={"route_intention", "original_query", "context_queries"},
                produces={"needs_retrieval", "rewritten_query", "active_query", "next_step_query"},
                capabilities={"llm"},
                config={"chat_llm_profile": self.retrieve_config.sufficiency_check_llm_profile},
            ),
            WorkflowStep(
                step_id="route_category",
                role="route_category",
                handler=self._rag_route_category,
                requires={"retrieve_category", "needs_retrieval", "active_query", "ctx", "store", "where"},
                produces={"category_hits", "category_summary_lookup", "query_vector"},
                capabilities={"vector"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="sufficiency_after_category",
                role="sufficiency_check",
                handler=self._rag_category_sufficiency,
                requires={
                    "retrieve_category",
                    "needs_retrieval",
                    "active_query",
                    "context_queries",
                    "category_hits",
                    "ctx",
                    "store",
                    "where",
                },
                produces={"next_step_query", "proceed_to_items", "query_vector"},
                capabilities={"llm"},
                config={
                    "chat_llm_profile": self.retrieve_config.sufficiency_check_llm_profile,
                    "embed_llm_profile": "embedding",
                },
            ),
            WorkflowStep(
                step_id="recall_items",
                role="recall_items",
                handler=self._rag_recall_items,
                requires={
                    "needs_retrieval",
                    "proceed_to_items",
                    "ctx",
                    "store",
                    "where",
                    "active_query",
                    "query_vector",
                },
                produces={"item_hits", "query_vector"},
                capabilities={"vector"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="sufficiency_after_items",
                role="sufficiency_check",
                handler=self._rag_item_sufficiency,
                requires={
                    "needs_retrieval",
                    "active_query",
                    "context_queries",
                    "item_hits",
                    "ctx",
                    "store",
                    "where",
                },
                produces={"next_step_query", "proceed_to_resources", "query_vector"},
                capabilities={"llm"},
                config={
                    "chat_llm_profile": self.retrieve_config.sufficiency_check_llm_profile,
                    "embed_llm_profile": "embedding",
                },
            ),
            WorkflowStep(
                step_id="recall_resources",
                role="recall_resources",
                handler=self._rag_recall_resources,
                requires={
                    "needs_retrieval",
                    "proceed_to_resources",
                    "ctx",
                    "store",
                    "where",
                    "active_query",
                    "query_vector",
                },
                produces={"resource_hits", "query_vector"},
                capabilities={"vector"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="build_context",
                role="build_context",
                handler=self._rag_build_context,
                requires={"needs_retrieval", "original_query", "rewritten_query", "ctx", "store", "where"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    def _list_retrieve_initial_keys(self) -> set[str]:
        return {
            "method",
            "original_query",
            "context_queries",
            "route_intention",
            "retrieve_category",
            "retrieve_item",
            "retrieve_resource",
            "sufficiency_check",
            "ctx",
            "store",
            "where",
            "as_of",
        }

    async def _rag_route_intention(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("route_intention"):
            state.update({
                "needs_retrieval": True,
                "rewritten_query": state["original_query"],
                "active_query": state["original_query"],
                "next_step_query": None,
                "proceed_to_items": False,
                "proceed_to_resources": False,
            })
            return state

        llm_client = self._get_step_llm_client(step_context)
        # Prompt-diversity: rotate among topic / relation / counterpoint-hint
        # lenses on consecutive RETRIEVE turns. Server picks the angle.
        angle_prompt = _system_prompt_for_angle(state.get("rewrite_angle"))
        needs_retrieval, rewritten_query, raw_response = await self._decide_if_retrieval_needed(
            state["original_query"],
            state.get("route_context_queries", state["context_queries"]),
            retrieved_content=None,
            system_prompt=angle_prompt,
            llm_client=llm_client,
        )
        mental_health_query = self._extract_mental_health_query(raw_response)

        state.update({
            "needs_retrieval": needs_retrieval,
            "rewritten_query": rewritten_query,
            "active_query": rewritten_query,
            "mental_health_query": mental_health_query,
            "next_step_query": None,
            "proceed_to_items": False,
            "proceed_to_resources": False,
        })
        return state

    async def _rag_route_category(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("retrieve_category") or not state.get("needs_retrieval"):
            state["category_hits"] = []
            state["category_summary_lookup"] = {}
            state["query_vector"] = None
            return state

        embed_client = self._get_step_embedding_client(step_context)
        store = state["store"]
        where_filters = state["where"]
        category_pool = store.memory_category_repo.list_categories(where_filters)
        qvec = (await embed_client.embed([state["active_query"]]))[0]
        hits, summary_lookup = await self._rank_categories_by_summary(
            qvec,
            self.retrieve_config.category.top_k,
            store,
            embed_client=embed_client,
            categories=category_pool,
        )
        cat_cfg = self.retrieve_config.category
        if hits:
            top_score = hits[0][1]
            hits = [h for h in hits if h[1] >= cat_cfg.min_score and h[1] >= (top_score - cat_cfg.score_window)][:cat_cfg.max_count]
        state.update({
            "query_vector": qvec,
            "category_hits": hits,
            "category_summary_lookup": summary_lookup,
            "category_pool": category_pool,
        })
        return state

    async def _rag_category_sufficiency(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        # Near-duplicate of _llm_category_sufficiency. The only structural difference is how
        # retrieved_content is formatted: RAG uses _format_category_content (needs store +
        # summary_lookup + category_pool); LLM uses _format_llm_category_content (hits only).
        # Intentionally kept separate — injecting a formatter callable adds more coupling
        # than the duplication costs.
        if not state.get("needs_retrieval"):
            state["proceed_to_items"] = False
            return state

        retrieved_content = ""
        store = state["store"]
        where_filters = state["where"]
        category_pool = state.get("category_pool") or store.memory_category_repo.list_categories(where_filters)
        hits = state.get("category_hits") or []
        if hits:
            retrieved_content = self._format_category_content(
                hits,
                state.get("category_summary_lookup", {}),
                store,
                categories=category_pool,
            )

        if not state.get("retrieve_category") or not state.get("sufficiency_check"):
            # Rewrite-only mode: gating disabled, LLM called for query rewrite only,
            # retrieval proceeds regardless of what the LLM would decide about sufficiency.
            llm_client = self._get_step_llm_client(step_context)
            _needs_more, rewritten_query, _ = await self._decide_if_retrieval_needed(
                state["active_query"],
                state["context_queries"],
                retrieved_content=retrieved_content or "No content retrieved yet.",
                llm_client=llm_client,
            )
            state["next_step_query"] = rewritten_query
            state["active_query"] = rewritten_query
            state["proceed_to_items"] = True
            embed_client = self._get_step_embedding_client(step_context)
            state["query_vector"] = (await embed_client.embed([state["active_query"]]))[0]
            return state

        llm_client = self._get_step_llm_client(step_context)
        needs_more, rewritten_query, _ = await self._decide_if_retrieval_needed(
            state["active_query"],
            state["context_queries"],
            retrieved_content=retrieved_content or "No content retrieved yet.",
            llm_client=llm_client,
        )
        state["next_step_query"] = rewritten_query
        state["active_query"] = rewritten_query
        state["proceed_to_items"] = needs_more
        if needs_more:
            embed_client = self._get_step_embedding_client(step_context)
            state["query_vector"] = (await embed_client.embed([state["active_query"]]))[0]
        return state

    def _find_entity_matches(self, text: str, store: Database, where: Mapping[str, Any] | None = None) -> list[Any]:
        all_entities = store.entity_repo.list_all(where)
        if not all_entities:
            return []
        text_lower = text.lower()
        return [e for e in all_entities if e.name.lower() in text_lower]

    def _find_superseded_at(
        self,
        store: Database,
        item_id: str,
        where: Mapping[str, Any] | None = None,
    ) -> datetime | None:
        edges = store.triple_repo.get_edges_from(item_id, predicate="evolved_into", where=where)
        if not edges:
            return None
        return edges[0].valid_from

    def _get_entity_seed_memory_ids(
        self,
        entities: list[Any],
        store: Database,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        memory_ids: list[str] = []
        seen: set[str] = set()
        provenance: dict[str, str] = {}
        for entity in entities:
            triples = store.triple_repo.get_edges_to(entity.id, predicate="mentions", where=where, as_of=as_of)
            for t in triples:
                if t.subject_id not in seen:
                    seen.add(t.subject_id)
                    memory_ids.append(t.subject_id)
                    provenance[t.subject_id] = f"via {entity.name}"
        return memory_ids, provenance

    async def _rag_recall_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("retrieve_item") or not state.get("needs_retrieval") or not state.get("proceed_to_items"):
            state["item_hits"] = []
            return state

        store = state["store"]
        where_filters = state["where"]
        include_superseded = state.get("as_of") is not None
        items_pool = store.memory_item_repo.list_items(where_filters, include_superseded=include_superseded)
        qvec = state.get("query_vector")
        if qvec is None:
            embed_client = self._get_step_embedding_client(step_context)
            qvec = (await embed_client.embed([state["active_query"]]))[0]
            state["query_vector"] = qvec
        item_cfg = self.retrieve_config.item

        vector_hits = store.memory_item_repo.vector_search_items(
            qvec,
            item_cfg.top_k,
            where=where_filters,
            ranking=item_cfg.ranking,
            recency_decay_days=item_cfg.recency_decay_days,
            fts_query=state.get("active_query"),
            fts_enabled=item_cfg.fts_enabled,
            fts_top_k=item_cfg.fts_top_k,
            rrf_k=item_cfg.rrf_k,
            include_superseded=include_superseded,
        )

        graph_cfg = self.retrieve_config.graph
        graph_provenance: dict[str, str] = {}
        graph_edges: dict[str, tuple[str, str]] = {}

        if graph_cfg.enabled:
            entity_seed_ids: list[str] = []
            expanded_ids: list[str] = []
            matched_entities = self._find_entity_matches(state["active_query"], store, where_filters)
            if matched_entities:
                entity_seed_ids, provenance = self._get_entity_seed_memory_ids(
                    matched_entities,
                    store,
                    where_filters,
                    as_of=state.get("as_of"),
                )
                graph_provenance.update(provenance)

            vector_ids = [item_id for item_id, _ in vector_hits]
            all_seed_ids = list(dict.fromkeys(vector_ids + entity_seed_ids))
            if all_seed_ids:
                # Exclude "mentions" so expansion follows only semantic edges
                sem_predicates = [
                    "caused_by", "evokes", "conflicts_with", "parallels", "shaped_by",
                ]
                expanded_edges = store.triple_repo.get_connected_memory_edges(
                    all_seed_ids,
                    predicates=sem_predicates,
                    max_per_source=3,
                    where=where_filters,
                    as_of=state.get("as_of"),
                )
                expanded_ids = [mid for (mid, _, _) in expanded_edges]
                for mid, predicate, seed_id in expanded_edges:
                    if mid not in graph_edges:
                        graph_edges[mid] = (predicate, seed_id)

            vector_id_set = {item_id for item_id, _ in vector_hits}
            graph_only = [
                mid for mid in (entity_seed_ids + expanded_ids)
                if mid not in vector_id_set
            ]
            seen: set[str] = set()
            deduped: list[str] = []
            for mid in graph_only:
                if mid not in seen:
                    seen.add(mid)
                    deduped.append(mid)
            deduped = deduped[:graph_cfg.max_graph_results]
            # score=0.0 is a sentinel for graph-expanded hits (no cosine score); not a weak match
            vector_hits = list(vector_hits) + [(mid, 0.0) for mid in deduped]

        state["item_hits"] = vector_hits
        state["item_pool"] = items_pool
        state["graph_provenance"] = graph_provenance
        state["graph_edges"] = graph_edges
        return state

    async def _rag_item_sufficiency(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        state["next_step_query"] = state.get("active_query")
        state["proceed_to_resources"] = False
        return state

    async def _rag_recall_resources(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if (
            not state.get("needs_retrieval")
            or not state.get("retrieve_resource")
            or not state.get("proceed_to_resources")
        ):
            state["resource_hits"] = []
            return state

        store = state["store"]
        where_filters = state["where"]
        resource_pool = store.resource_repo.list_resources(where_filters)
        state["resource_pool"] = resource_pool
        corpus = self._resource_caption_corpus(store, resources=resource_pool)
        if not corpus:
            state["resource_hits"] = []
            return state

        qvec = state.get("query_vector")
        if qvec is None:
            embed_client = self._get_step_embedding_client(step_context)
            qvec = (await embed_client.embed([state["active_query"]]))[0]
            state["query_vector"] = qvec
        state["resource_hits"] = cosine_topk(qvec, corpus, k=self.retrieve_config.resource.top_k)
        return state

    def _rag_build_context(self, state: WorkflowState, _: Any) -> WorkflowState:
        response = {
            "needs_retrieval": bool(state.get("needs_retrieval")),
            "original_query": state["original_query"],
            "rewritten_query": state.get("rewritten_query", state["original_query"]),
            "mental_health_query": state.get("mental_health_query"),
            "next_step_query": state.get("next_step_query"),
            "categories": [],
            "items": [],
            "resources": [],
        }
        if state.get("needs_retrieval"):
            store = state["store"]
            where_filters = state["where"]
            include_superseded = state.get("as_of") is not None
            categories_pool = state.get("category_pool") or store.memory_category_repo.list_categories(where_filters)
            items_pool = state.get("item_pool") or store.memory_item_repo.list_items(
                where_filters, include_superseded=include_superseded
            )
            resources_pool = state.get("resource_pool") or store.resource_repo.list_resources(where_filters)
            response["categories"] = self._materialize_hits(
                state.get("category_hits", []),
                categories_pool,
            )
            response["items"] = self._materialize_hits(state.get("item_hits", []), items_pool)
            # Exclude narrative_self from retrieval — the current narrative_self is
            # already delivered through the soul_card / self-model path; pulling
            # paragraph-sized self-identity prose in as a retrieved "memory" is
            # pure bloat. Evolution awareness is a TODO for a separate surface.
            response["items"] = [
                it for it in response["items"]
                if (it.get("memory_type") or "") != "narrative_self"
            ]
            graph_provenance = state.get("graph_provenance") or {}
            graph_edges = state.get("graph_edges") or {}
            for item_data in response["items"]:
                item_id = item_data.get("id")
                if not item_id:
                    continue
                if item_id in graph_provenance:
                    item_data["via_graph"] = graph_provenance[item_id]
                if item_id in graph_edges:
                    predicate, seed_id = graph_edges[item_id]
                    seed = store.memory_item_repo.get_item(seed_id, include_superseded=include_superseded)
                    if seed is None or (seed.memory_type or "") == "narrative_self":
                        continue
                    item_data["shaped_by"] = {
                        "predicate": predicate,
                        "id": seed.id,
                        "memory_type": seed.memory_type,
                        "summary": seed.summary,
                        "happened_at": seed.happened_at,
                        "extra": seed.extra,
                        "superseded_at": self._find_superseded_at(store, seed.id, where_filters),
                    }
                evolved_at = self._find_superseded_at(store, item_id, where_filters)
                if evolved_at is not None:
                    item_data["superseded_at"] = evolved_at
            response["resources"] = self._materialize_hits(
                state.get("resource_hits", []),
                resources_pool,
            )
        state["response"] = response
        return state

    def _build_llm_retrieve_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="route_intention",
                role="route_intention",
                handler=self._llm_route_intention,
                requires={"original_query", "context_queries"},
                produces={"needs_retrieval", "rewritten_query", "active_query", "next_step_query"},
                capabilities={"llm"},
                config={"llm_profile": self.retrieve_config.sufficiency_check_llm_profile},
            ),
            WorkflowStep(
                step_id="route_category",
                role="route_category",
                handler=self._llm_route_category,
                requires={"needs_retrieval", "active_query", "ctx", "store", "where"},
                produces={"category_hits"},
                capabilities={"llm"},
                config={"llm_profile": self.retrieve_config.llm_ranking_llm_profile},
            ),
            WorkflowStep(
                step_id="sufficiency_after_category",
                role="sufficiency_check",
                handler=self._llm_category_sufficiency,
                requires={"needs_retrieval", "active_query", "context_queries", "category_hits"},
                produces={"next_step_query", "proceed_to_items"},
                capabilities={"llm"},
                config={"llm_profile": self.retrieve_config.sufficiency_check_llm_profile},
            ),
            WorkflowStep(
                step_id="recall_items",
                role="recall_items",
                handler=self._llm_recall_items,
                requires={
                    "needs_retrieval",
                    "proceed_to_items",
                    "ctx",
                    "store",
                    "where",
                    "active_query",
                    "category_hits",
                },
                produces={"item_hits"},
                capabilities={"llm"},
                config={"llm_profile": self.retrieve_config.llm_ranking_llm_profile},
            ),
            WorkflowStep(
                step_id="sufficiency_after_items",
                role="sufficiency_check",
                handler=self._llm_item_sufficiency,
                requires={"needs_retrieval", "active_query", "context_queries", "item_hits"},
                produces={"next_step_query", "proceed_to_resources"},
                capabilities={"llm"},
                config={"llm_profile": self.retrieve_config.sufficiency_check_llm_profile},
            ),
            WorkflowStep(
                step_id="recall_resources",
                role="recall_resources",
                handler=self._llm_recall_resources,
                requires={
                    "needs_retrieval",
                    "proceed_to_resources",
                    "active_query",
                    "ctx",
                    "store",
                    "where",
                    "item_hits",
                    "category_hits",
                },
                produces={"resource_hits"},
                capabilities={"llm"},
                config={"llm_profile": self.retrieve_config.llm_ranking_llm_profile},
            ),
            WorkflowStep(
                step_id="build_context",
                role="build_context",
                handler=self._llm_build_context,
                requires={"needs_retrieval", "original_query", "rewritten_query"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    async def _llm_route_intention(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("route_intention"):
            state.update({
                "needs_retrieval": True,
                "rewritten_query": state["original_query"],
                "active_query": state["original_query"],
                "next_step_query": None,
                "proceed_to_items": False,
                "proceed_to_resources": False,
            })
            return state

        llm_client = self._get_step_llm_client(step_context)
        needs_retrieval, rewritten_query, _ = await self._decide_if_retrieval_needed(
            state["original_query"],
            state.get("route_context_queries", state["context_queries"]),
            retrieved_content=None,
            llm_client=llm_client,
        )

        state.update({
            "needs_retrieval": needs_retrieval,
            "rewritten_query": rewritten_query,
            "active_query": rewritten_query,
            "next_step_query": None,
            "proceed_to_items": False,
            "proceed_to_resources": False,
        })
        return state

    async def _llm_route_category(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval"):
            state["category_hits"] = []
            return state
        llm_client = self._get_step_llm_client(step_context)
        store = state["store"]
        where_filters = state["where"]
        category_pool = store.memory_category_repo.list_categories(where_filters)
        hits = await self._llm_rank_categories(
            state["active_query"],
            self.retrieve_config.category.top_k,
            store,
            llm_client=llm_client,
            categories=category_pool,
        )
        state["category_hits"] = hits
        state["category_pool"] = category_pool
        return state

    async def _llm_category_sufficiency(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval"):
            state["proceed_to_items"] = False
            return state

        retrieved_content = ""
        hits = state.get("category_hits") or []
        if hits:
            retrieved_content = self._format_llm_category_content(hits)

        if not state.get("retrieve_category") or not state.get("sufficiency_check"):
            llm_client = self._get_step_llm_client(step_context)
            _needs_more, rewritten_query, _ = await self._decide_if_retrieval_needed(
                state["active_query"],
                state["context_queries"],
                retrieved_content=retrieved_content or "No content retrieved yet.",
                llm_client=llm_client,
            )
            state["next_step_query"] = rewritten_query
            state["active_query"] = rewritten_query
            state["proceed_to_items"] = True
            return state

        llm_client = self._get_step_llm_client(step_context)
        needs_more, rewritten_query, _ = await self._decide_if_retrieval_needed(
            state["active_query"],
            state["context_queries"],
            retrieved_content=retrieved_content or "No content retrieved yet.",
            llm_client=llm_client,
        )
        state["next_step_query"] = rewritten_query
        state["active_query"] = rewritten_query
        state["proceed_to_items"] = needs_more
        return state

    async def _llm_recall_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval") or not state.get("proceed_to_items"):
            state["item_hits"] = []
            return state

        where_filters = state["where"]
        category_hits = state.get("category_hits", [])
        category_ids = [cat["id"] for cat in category_hits]
        llm_client = self._get_step_llm_client(step_context)
        store = state["store"]

        use_refs = self.retrieve_config.item.use_category_references
        ref_ids: list[str] = []
        if use_refs and category_hits:
            from memu.utils.references import extract_references

            for cat in category_hits:
                summary = cat.get("summary") or ""
                ref_ids.extend(extract_references(summary))
        if ref_ids:
            items_pool = store.memory_item_repo.list_items_by_ref_ids(ref_ids, where_filters)
        else:
            items_pool = store.memory_item_repo.list_items(where_filters)

        relations = store.category_item_repo.list_relations(where_filters)
        category_pool = state.get("category_pool") or store.memory_category_repo.list_categories(where_filters)
        state["item_hits"] = await self._llm_rank_items(
            state["active_query"],
            self.retrieve_config.item.top_k,
            category_ids,
            state.get("category_hits", []),
            store,
            llm_client=llm_client,
            categories=category_pool,
            items=items_pool,
            relations=relations,
        )
        state["item_pool"] = items_pool
        return state

    async def _llm_item_sufficiency(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval"):
            state["proceed_to_resources"] = False
            return state
        if not state.get("retrieve_item") or not state.get("sufficiency_check"):
            state["proceed_to_resources"] = True
            return state

        retrieved_content = ""
        hits = state.get("item_hits") or []
        if hits:
            retrieved_content = self._format_llm_item_content(hits)

        llm_client = self._get_step_llm_client(step_context)
        needs_more, rewritten_query, _ = await self._decide_if_retrieval_needed(
            state["active_query"],
            state["context_queries"],
            retrieved_content=retrieved_content or "No content retrieved yet.",
            llm_client=llm_client,
        )
        state["next_step_query"] = rewritten_query
        state["active_query"] = rewritten_query
        state["proceed_to_resources"] = needs_more
        return state

    async def _llm_recall_resources(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval") or not state.get("proceed_to_resources"):
            state["resource_hits"] = []
            return state

        llm_client = self._get_step_llm_client(step_context)
        store = state["store"]
        where_filters = state["where"]
        resource_pool = store.resource_repo.list_resources(where_filters)
        item_pool_state = state.get("item_pool")
        items_pool = item_pool_state if isinstance(item_pool_state, dict) else {}
        state["resource_hits"] = await self._llm_rank_resources(
            state["active_query"],
            self.retrieve_config.resource.top_k,
            state.get("category_hits", []),
            state.get("item_hits", []),
            store,
            llm_client=llm_client,
            items=items_pool,
            resources=resource_pool,
        )
        state["resource_pool"] = resource_pool
        return state

    def _llm_build_context(self, state: WorkflowState, _: Any) -> WorkflowState:
        response = {
            "needs_retrieval": bool(state.get("needs_retrieval")),
            "original_query": state["original_query"],
            "rewritten_query": state.get("rewritten_query", state["original_query"]),
            "mental_health_query": state.get("mental_health_query"),
            "next_step_query": state.get("next_step_query"),
            "categories": [],
            "items": [],
            "resources": [],
        }
        if state.get("needs_retrieval"):
            response["categories"] = list(state.get("category_hits") or [])
            response["items"] = list(state.get("item_hits") or [])
            response["resources"] = list(state.get("resource_hits") or [])
        state["response"] = response
        return state

    async def _rank_categories_by_summary(
        self,
        query_vec: list[float],
        top_k: int,
        store: Database,
        embed_client: Any | None = None,
        categories: Mapping[str, Any] | None = None,
    ) -> tuple[list[tuple[str, float]], dict[str, str]]:
        category_pool = categories if categories is not None else store.memory_category_repo.categories
        entries = [(cid, cat.summary) for cid, cat in category_pool.items() if cat.summary]
        if not entries:
            return [], {}
        cache = self._category_summary_embedding_cache
        missing_entries: list[tuple[str, str]] = []
        for cid, summary in entries:
            cached = cache.get(cid)
            if cached is None or cached[0] != summary:
                missing_entries.append((cid, summary))

        if missing_entries:
            client = embed_client or self._get_llm_client()
            missing_embeddings = await client.embed([summary for _, summary in missing_entries])
            for (cid, summary), emb in zip(missing_entries, missing_embeddings, strict=True):
                cache[cid] = (summary, emb)

        corpus = [(cid, cache[cid][1]) for cid, _ in entries]
        hits = cosine_topk(query_vec, corpus, k=top_k)
        summary_lookup = dict(entries)
        return hits, summary_lookup

    async def _decide_if_retrieval_needed(
        self,
        query: str,
        context_queries: list[dict[str, Any]] | None,
        retrieved_content: str | None = None,
        system_prompt: str | None = None,
        llm_client: Any | None = None,
    ) -> tuple[bool, str, str]:
        history_text = self._format_query_context(context_queries)
        content_text = retrieved_content or "No content retrieved yet."

        prompt = self.retrieve_config.sufficiency_check_prompt or PRE_RETRIEVAL_USER_PROMPT
        user_prompt = prompt.format(
            query=self._escape_prompt_value(query),
            conversation_history=self._escape_prompt_value(history_text),
            retrieved_content=self._escape_prompt_value(content_text),
        )

        sys_prompt = system_prompt or PRE_RETRIEVAL_SYSTEM_PROMPT
        client = llm_client or self._get_llm_client()
        response = await client.chat(user_prompt, system_prompt=sys_prompt)
        decision = self._extract_decision(response)
        rewritten = self._extract_rewritten_query(response) or query

        # Caller can pull <mental_health_query> out of `response` if it cares.
        return decision == "RETRIEVE", rewritten, response

    def _format_query_context(self, queries: list[dict[str, Any]] | None) -> str:
        if not queries:
            return "No query context."

        lines = []
        for q in queries:
            if isinstance(q, str):
                # Backward compatibility
                lines.append(f"- {q}")
            elif isinstance(q, dict):
                role = q.get("role", "user")
                content = q.get("content")
                if isinstance(content, dict):
                    text = content.get("text", "")
                elif isinstance(content, str):
                    text = content
                else:
                    text = str(content)
                role_text = str(role or "").strip().lower()
                if role_text == "identity_context":
                    lines.append(text)
                    continue
                lines.append(f"- [{role}]: {text}")
            else:
                lines.append(f"- {q!s}")

        return "\n".join(lines)

    @staticmethod
    def _split_context_queries(context_queries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        route_context = list(context_queries)
        # "history_from_second_chat_x" must stay in sync with _build_retrieve_soul_context_queries()
        # in mcp-memu-server/app/main.py, which assigns this role server-side before calling retrieve.
        downstream_context = [
            q
            for q in context_queries
            if not (
                isinstance(q, dict)
                and str(q.get("role") or "").strip().lower() == "history_from_second_chat_x"
            )
        ]
        return route_context, downstream_context

    @staticmethod
    def _extract_query_text(query: dict[str, Any]) -> str:
        if isinstance(query, str):
            return query
        if not isinstance(query, dict):
            raise TypeError("INVALID")
        content = query.get("content")
        if isinstance(content, dict):
            text = content.get("text", "")
            if not text:
                raise ValueError("EMPTY")
            return str(text)
        elif isinstance(content, str):
            return content
        else:
            raise TypeError("INVALID")

    def _extract_decision(self, raw: str) -> str:
        if not raw:
            return "RETRIEVE"

        match = re.search(r"<decision>(.*?)</decision>", raw, re.IGNORECASE | re.DOTALL)
        if match:
            decision = match.group(1).strip().upper()
            if "NO_RETRIEVE" in decision or "NO RETRIEVE" in decision:
                return "NO_RETRIEVE"
            if "RETRIEVE" in decision:
                return "RETRIEVE"

        upper = raw.strip().upper()
        if "NO_RETRIEVE" in upper or "NO RETRIEVE" in upper:
            return "NO_RETRIEVE"

        return "RETRIEVE"

    def _extract_rewritten_query(self, raw: str) -> str | None:
        match = re.search(r"<rewritten_query>(.*?)</rewritten_query>", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_mental_health_query(self, raw: str) -> str | None:
        match = re.search(r"<mental_health_query>(.*?)</mental_health_query>", raw, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            return text or None
        return None

    def _materialize_hits(self, hits: Sequence[tuple[str, float]], pool: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for _id, score in hits:
            obj = pool.get(_id)
            if not obj:
                continue
            data = self._model_dump_without_embeddings(obj)
            if "memory_type" in data:
                data["speaker_id"] = data.get("speaker_id")
                data["speaker_label"] = data.get("speaker_label")
            data["score"] = float(score)
            out.append(data)
        return out

    def _format_category_content(
        self,
        hits: list[tuple[str, float]],
        summaries: dict[str, str],
        store: Database,
        categories: Mapping[str, Any] | None = None,
    ) -> str:
        category_pool = categories if categories is not None else store.memory_category_repo.categories
        lines = []
        for cid, score in hits:
            cat = category_pool.get(cid)
            if not cat:
                continue
            summary = summaries.get(cid) or cat.summary or ""
            lines.append(f"Category: {cat.name}\nSummary: {summary}\nScore: {score:.3f}")
        return "\n\n".join(lines).strip()

    def _resource_caption_corpus(
        self, store: Database, resources: Mapping[str, Any] | None = None
    ) -> list[tuple[str, list[float]]]:
        resource_pool = resources if resources is not None else store.resource_repo.resources
        corpus: list[tuple[str, list[float]]] = []
        for rid, res in resource_pool.items():
            if res.embedding:
                corpus.append((rid, res.embedding))
        return corpus

    def _format_categories_for_llm(
        self,
        store: Database,
        category_ids: list[str] | None = None,
        categories: Mapping[str, Any] | None = None,
    ) -> str:
        categories_to_format = categories if categories is not None else store.memory_category_repo.categories
        if category_ids:
            categories_to_format = {cid: cat for cid, cat in categories_to_format.items() if cid in category_ids}

        if not categories_to_format:
            return "No categories available."

        lines = []
        for cid, cat in categories_to_format.items():
            lines.append(f"ID: {cid}")
            lines.append(f"Name: {cat.name}")
            if cat.description:
                lines.append(f"Description: {cat.description}")
            if cat.summary:
                lines.append(f"Summary: {cat.summary}")
            lines.append("---")

        return "\n".join(lines)

    def _format_items_for_llm(
        self,
        store: Database,
        category_ids: list[str] | None = None,
        *,
        items: Mapping[str, Any],
        relations: Sequence[Any] | None = None,
    ) -> str:
        # items is required (scoped by the caller). Removed a fallback
        # `list_items()` that would've run a full-table scan under a
        # per-soul DB — all production call sites pass a scoped pool.
        item_pool = items
        relation_pool = relations if relations is not None else store.category_item_repo.relations
        items_to_format = []
        seen_item_ids = set()

        if category_ids:
            for rel in relation_pool:
                if rel.category_id in category_ids:
                    item = item_pool.get(rel.item_id)
                    if item and item.id not in seen_item_ids:
                        items_to_format.append(item)
                        seen_item_ids.add(item.id)
        else:
            items_to_format = list(item_pool.values())

        if not items_to_format:
            return "No memory items available."

        lines = []
        for item in items_to_format:
            lines.append(f"ID: {item.id}")
            lines.append(f"Type: {item.memory_type}")
            lines.append(f"Summary: {item.summary}")
            lines.append("---")

        return "\n".join(lines)

    def _format_resources_for_llm(
        self,
        store: Database,
        item_ids: list[str] | None = None,
        *,
        items: Mapping[str, Any],
        resources: Mapping[str, Any] | None = None,
    ) -> str:
        # items is required; same rationale as _format_items_for_llm above.
        resource_pool = resources if resources is not None else store.resource_repo.resources
        item_pool = items
        resources_to_format = []

        if item_ids:
            resource_ids = {item_pool[iid].resource_id for iid in item_ids if iid in item_pool}
            resources_to_format = [
                resource_pool[rid] for rid in resource_ids if rid in resource_pool and rid is not None
            ]
        else:
            resources_to_format = list(resource_pool.values())

        if not resources_to_format:
            return "No resources available."

        lines = []
        for res in resources_to_format:
            lines.append(f"ID: {res.id}")
            lines.append(f"URL: {res.url}")
            lines.append(f"Modality: {res.modality}")
            if res.caption:
                lines.append(f"Caption: {res.caption}")
            lines.append("---")

        return "\n".join(lines)

    async def _llm_rank_categories(
        self,
        query: str,
        top_k: int,
        store: Database,
        llm_client: Any | None = None,
        categories: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        category_pool = categories if categories is not None else store.memory_category_repo.categories
        if not category_pool:
            return []

        categories_data = self._format_categories_for_llm(store, categories=category_pool)
        prompt = LLM_CATEGORY_RANKER_PROMPT.format(
            query=self._escape_prompt_value(query),
            top_k=top_k,
            categories_data=self._escape_prompt_value(categories_data),
        )

        client = llm_client or self._get_llm_client()
        llm_response = await client.chat(prompt)
        return self._parse_llm_category_response(llm_response, store, categories=category_pool)

    async def _llm_rank_items(
        self,
        query: str,
        top_k: int,
        category_ids: list[str],
        category_hits: list[dict[str, Any]],
        store: Database,
        *,
        items: Mapping[str, Any],
        llm_client: Any | None = None,
        categories: Mapping[str, Any] | None = None,
        relations: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not category_ids:
            return []

        item_pool = items
        items_data = self._format_items_for_llm(store, category_ids, items=item_pool, relations=relations)
        if items_data == "No memory items available.":
            return []

        relevant_categories_info = "\n".join([
            f"- {cat['name']}: {cat.get('summary', cat.get('description', ''))}" for cat in category_hits
        ])

        prompt = LLM_ITEM_RANKER_PROMPT.format(
            query=self._escape_prompt_value(query),
            top_k=top_k,
            relevant_categories=self._escape_prompt_value(relevant_categories_info),
            items_data=self._escape_prompt_value(items_data),
        )

        client = llm_client or self._get_llm_client()
        llm_response = await client.chat(prompt)
        return self._parse_llm_item_response(llm_response, store, items=item_pool)

    async def _llm_rank_resources(
        self,
        query: str,
        top_k: int,
        category_hits: list[dict[str, Any]],
        item_hits: list[dict[str, Any]],
        store: Database,
        *,
        items: Mapping[str, Any],
        llm_client: Any | None = None,
        resources: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        item_ids = [item["id"] for item in item_hits]
        if not item_ids:
            return []

        item_pool = items
        resource_pool = resources if resources is not None else store.resource_repo.resources
        resources_data = self._format_resources_for_llm(store, item_ids, items=item_pool, resources=resource_pool)
        if resources_data == "No resources available.":
            return []

        context_parts = []
        if category_hits:
            context_parts.append("Relevant Categories:")
            context_parts.extend([f"- {cat['name']}" for cat in category_hits])
        if item_hits:
            context_parts.append("\nRelevant Memory Items:")
            context_parts.extend([f"- {item.get('summary', '')[:100]}..." for item in item_hits[:3]])

        context_info = "\n".join(context_parts)
        prompt = LLM_RESOURCE_RANKER_PROMPT.format(
            query=self._escape_prompt_value(query),
            top_k=top_k,
            context_info=self._escape_prompt_value(context_info),
            resources_data=self._escape_prompt_value(resources_data),
        )

        client = llm_client or self._get_llm_client()
        llm_response = await client.chat(prompt)
        return self._parse_llm_resource_response(llm_response, store, resources=resource_pool)

    def _parse_llm_id_list_response(
        self, raw_response: str, key: str, pool: Mapping[str, Any], label: str
    ) -> list[dict[str, Any]]:
        results = []
        try:
            json_blob = self._extract_json_blob(raw_response)
            parsed = json.loads(json_blob)
            if key in parsed and isinstance(parsed[key], list):
                for obj_id in parsed[key]:
                    if isinstance(obj_id, str):
                        obj = pool.get(obj_id)
                        if obj:
                            data = self._model_dump_without_embeddings(obj)
                            if "memory_type" in data:
                                data["speaker_id"] = data.get("speaker_id")
                                data["speaker_label"] = data.get("speaker_label")
                            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to parse LLM {label} ranking response: {e}")
        return results

    def _parse_llm_category_response(
        self, raw_response: str, store: Database, categories: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        pool = categories if categories is not None else store.memory_category_repo.categories
        return self._parse_llm_id_list_response(raw_response, "categories", pool, "category")

    def _parse_llm_item_response(
        self, raw_response: str, store: Database, *, items: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        # items required; caller (_llm_rank_items) owns pool scoping.
        return self._parse_llm_id_list_response(raw_response, "items", items, "item")

    def _parse_llm_resource_response(
        self, raw_response: str, store: Database, resources: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        pool = resources if resources is not None else store.resource_repo.resources
        return self._parse_llm_id_list_response(raw_response, "resources", pool, "resource")

    def _format_llm_category_content(self, hits: list[dict[str, Any]]) -> str:
        lines = []
        for cat in hits:
            summary = cat.get("summary", "") or cat.get("description", "")
            lines.append(f"Category: {cat['name']}\nSummary: {summary}")
        return "\n\n".join(lines).strip()

    def _format_llm_item_content(self, hits: list[dict[str, Any]]) -> str:
        lines = []
        for item in hits:
            lines.append(f"Memory Item ({item['memory_type']}): {item['summary']}")
        return "\n\n".join(lines).strip()

    def _format_llm_resource_content(self, hits: list[dict[str, Any]]) -> str:
        lines = []
        for res in hits:
            caption = res.get("caption", "") or f"Resource {res['url']}"
            lines.append(f"Resource: {caption}")
        return "\n\n".join(lines).strip()
