from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from memu.database.vector import autocut_first_cluster, cosine_topk, relative_score_fusion
from memu.prompts.retrieve.pre_retrieval_decision import USER_PROMPT as PRE_RETRIEVAL_USER_PROMPT
from memu.prompts.retrieve.pre_retrieval_decision import forced_query_system_prompt as _forced_query_system_prompt
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
        _select_chat_client: Callable[..., Any]
        _select_embedding_client: Callable[[Mapping[str, Any] | None], Any]
        _model_dump_without_embeddings: Callable[[BaseModel], dict[str, Any]]
        _extract_json_blob: Callable[[str], str]
        _escape_prompt_value: Callable[[str], str]
        extract_memory_refs: Callable[..., list[int]]
        user_model: type[BaseModel]

    async def retrieve(
        self,
        queries: list[dict[str, Any]],
        where: dict[str, Any] | None = None,
        as_of: datetime | None = None,
        rewrite_angle: int = 0,
        mental_health_enabled: bool = True,
        force_retrieve: bool = False,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        if not queries:
            raise ValueError("empty_queries")
        ctx = self._get_context()
        store = self._get_database()
        new_message = self._extract_query_text(queries[-1])
        where_filters = self._normalize_where(where)

        context_queries = queries[:-1] if len(queries) > 1 else []

        workflow_name = "retrieve_rag"

        state: WorkflowState = {
            "new_message": new_message,
            "context_queries": list(context_queries),
            "rewrite_angle": int(rewrite_angle) if rewrite_angle is not None else 0,
            "mental_health_enabled": bool(mental_health_enabled),
            "force_retrieve": bool(force_retrieve),
            "ctx": ctx,
            "store": store,
            "where": where_filters,
            "as_of": as_of,
        }
        trace_id_clean = str(trace_id or "").strip()
        if trace_id_clean:
            state["trace_id"] = trace_id_clean

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
                requires={"new_message", "context_queries"},
                produces={"needs_retrieval", "active_query"},
                capabilities={"llm"},
                config={"chat_llm_profile": self.retrieve_config.sufficiency_check_llm_profile},
            ),
            WorkflowStep(
                step_id="route_category",
                role="route_category",
                handler=self._rag_route_category,
                requires={"needs_retrieval", "active_query", "ctx", "store", "where"},
                produces={"category_hits", "category_summary_lookup", "query_vector"},
                capabilities={"vector"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="sufficiency_after_category",
                role="sufficiency_check",
                handler=self._rag_category_sufficiency,
                requires={
                    "needs_retrieval",
                    "new_message",
                    "active_query",
                    "context_queries",
                    "category_hits",
                    "ctx",
                    "store",
                    "where",
                },
                produces={"proceed_to_items", "query_vector", "requested_memory_refs"},
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
                    "requested_memory_refs",
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
                produces={"proceed_to_resources", "query_vector"},
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
                requires={"needs_retrieval", "new_message", "active_query", "ctx", "store", "where"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    def _list_retrieve_initial_keys(self) -> set[str]:
        return {
            "new_message",
            "context_queries",
            "ctx",
            "store",
            "where",
            "as_of",
            "mental_health_enabled",
            "force_retrieve",
        }

    async def _rag_route_intention(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if bool(state.get("force_retrieve")):
            state.update({
                "needs_retrieval": True,
                "active_query": "",
                "mental_health_query": None,
                "proceed_to_items": False,
                "proceed_to_resources": False,
            })
            return state

        llm_client = self._select_chat_client(step_context)
        mental_health_enabled = bool(state.get("mental_health_enabled", True))
        angle_prompt = _system_prompt_for_angle(
            state.get("rewrite_angle"),
            include_mental_health_query=mental_health_enabled,
        )
        needs_retrieval, active_query, raw_response = await self._decide_if_retrieval_needed(
            state["new_message"],
            state["context_queries"],
            retrieved_content=None,
            system_prompt=angle_prompt,
            include_mental_health_query=mental_health_enabled,
            llm_client=llm_client,
        )
        mental_health_query = self._extract_mental_health_query(raw_response) if mental_health_enabled else None

        state.update({
            "needs_retrieval": needs_retrieval,
            "active_query": active_query,
            "mental_health_query": mental_health_query,
            "proceed_to_items": False,
            "proceed_to_resources": False,
        })
        return state

    async def _rag_route_category(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval"):
            state["category_hits"] = []
            state["category_summary_lookup"] = {}
            state["query_vector"] = None
            return state
        if bool(state.get("force_retrieve")):
            state["category_hits"] = []
            state["category_summary_lookup"] = {}
            state["category_pool"] = {}
            state["query_vector"] = None
            return state

        embed_client = self._select_embedding_client(step_context)
        where_filters = state["where"]
        category_rows = [
            category
            for category in self.list_active_dossiers(where_filters)
            if category.anchor_role is None
        ]
        category_pool = {category.id: category for category in category_rows}
        qvec = (await embed_client.embed([state["active_query"]]))[0]
        identity = await self.search_dossiers(
            qvec,
            where=where_filters,
            view="identity",
            activity="active",
            limit=self.retrieve_config.category.top_k,
            min_score=-1.0,
            categories=category_rows,
        )
        content = await self.search_dossiers(
            qvec,
            where=where_filters,
            view="content",
            activity="active",
            limit=self.retrieve_config.category.top_k,
            min_score=-1.0,
            embedding_client=embed_client,
            categories=category_rows,
        )
        fused = relative_score_fusion(
            [(category.id, score) for category, score in identity],
            [(category.id, score) for category, score in content],
        )[: self.retrieve_config.category.max_count]
        hits = autocut_first_cluster(fused)
        summary_lookup = {
            category.id: str(category.summary or "")
            for category in category_rows
        }
        state.update({
            "query_vector": qvec,
            "category_hits": hits,
            "category_summary_lookup": summary_lookup,
            "category_pool": category_pool,
        })
        return state

    async def _rag_category_sufficiency(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval"):
            state["proceed_to_items"] = False
            state["requested_memory_refs"] = []
            return state

        retrieved_content = ""
        hits = [] if bool(state.get("force_retrieve")) else (state.get("category_hits") or [])
        if hits:
            store = state["store"]
            where_filters = state["where"]
            category_pool = state.get("category_pool") or store.memory_category_repo.list_categories(where_filters)
            retrieved_content = self._format_category_content(
                hits,
                state.get("category_summary_lookup", {}),
                store,
                categories=category_pool,
            )

        llm_client = self._select_chat_client(step_context)
        mental_health_enabled = bool(state.get("mental_health_enabled", True))
        force_retrieve = bool(state.get("force_retrieve"))
        if force_retrieve:
            needs_more, active_query, raw_response = await self._decide_if_retrieval_needed(
                state["new_message"],
                state["context_queries"],
                retrieved_content=retrieved_content,
                system_prompt=_forced_query_system_prompt(include_mental_health_query=mental_health_enabled),
                include_mental_health_query=mental_health_enabled,
                llm_client=llm_client,
                require_decision=False,
            )
        else:
            needs_more, active_query, raw_response = await self._decide_if_retrieval_needed(
                state["new_message"],
                state["context_queries"],
                retrieved_content=retrieved_content,
                system_prompt=_system_prompt_for_angle(
                    0,
                    include_mental_health_query=mental_health_enabled,
                    include_memory_refs=bool(retrieved_content),
                ),
                include_mental_health_query=mental_health_enabled,
                llm_client=llm_client,
            )
        state["requested_memory_refs"] = (
            self.extract_memory_refs(raw_response, strict=False) if retrieved_content else []
        )
        if mental_health_enabled:
            mental_health_query = self._extract_mental_health_query(raw_response)
            if mental_health_query:
                state["mental_health_query"] = mental_health_query
        state["active_query"] = active_query
        proceed_to_items = True if force_retrieve else needs_more
        state["proceed_to_items"] = proceed_to_items
        if proceed_to_items:
            embed_client = self._select_embedding_client(step_context)
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
                    provenance[t.subject_id] = f"found via entity '{entity.name}'"
        return memory_ids, provenance

    async def _rag_recall_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        requested_refs = state.get("requested_memory_refs") or []
        if not state.get("needs_retrieval") or (
            not state.get("proceed_to_items") and not requested_refs
        ):
            state["item_hits"] = []
            return state

        store = state["store"]
        where_filters = state["where"]
        include_superseded = state.get("as_of") is not None
        items_pool = store.memory_item_repo.list_items(
            where_filters,
            include_superseded=include_superseded,
            include_embeddings=False,
        )
        exact_ids: list[str] = []
        for memory_ref in requested_refs:
            item = store.memory_item_repo.get_item_by_memory_ref(memory_ref, where_filters)
            if item is not None and item.id in items_pool and item.id not in exact_ids:
                exact_ids.append(item.id)
        if not state.get("proceed_to_items"):
            state["item_hits"] = [(item_id, 0.0) for item_id in exact_ids]
            state["item_pool"] = items_pool
            state["graph_provenance"] = {}
            state["graph_edges"] = {}
            return state
        qvec = state.get("query_vector")
        if qvec is None:
            embed_client = self._select_embedding_client(step_context)
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

        exact_id_set = set(exact_ids)
        state["item_hits"] = [(item_id, 0.0) for item_id in exact_ids] + [
            hit for hit in vector_hits if hit[0] not in exact_id_set
        ]
        state["item_pool"] = items_pool
        state["graph_provenance"] = graph_provenance
        state["graph_edges"] = graph_edges
        return state

    async def _rag_item_sufficiency(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        state["proceed_to_resources"] = False
        return state

    async def _rag_recall_resources(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval") or not state.get("proceed_to_resources"):
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
            embed_client = self._select_embedding_client(step_context)
            qvec = (await embed_client.embed([state["active_query"]]))[0]
            state["query_vector"] = qvec
        state["resource_hits"] = cosine_topk(qvec, corpus, k=self.retrieve_config.resource.top_k)
        return state

    def _rag_build_context(self, state: WorkflowState, _: Any) -> WorkflowState:
        response = {
            "needs_retrieval": bool(state.get("needs_retrieval")),
            "new_message": state["new_message"],
            "active_query": state.get("active_query", ""),
            "mental_health_query": state.get("mental_health_query"),
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
                where_filters,
                include_superseded=include_superseded,
                include_embeddings=False,
            )
            resources_pool = state.get("resource_pool") or store.resource_repo.list_resources(where_filters)
            response["categories"] = self._materialize_hits(
                state.get("category_hits", []),
                categories_pool,
            )
            response["items"] = self._materialize_hits(state.get("item_hits", []), items_pool)
            # These memory rows have dedicated prompt paths; retrieving them as
            # ordinary memories duplicates or bloats the turn context.
            response["items"] = [
                it for it in response["items"]
                if (it.get("memory_type") or "") != "narrative_self"
                and not (it.get("extra") or {}).get("apimw_message_to_self")
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


    async def _decide_if_retrieval_needed(
        self,
        new_message: str,
        context_queries: list[dict[str, Any]] | None,
        retrieved_content: str | None = None,
        system_prompt: str | None = None,
        include_mental_health_query: bool = True,
        llm_client: Any | None = None,
        require_decision: bool = True,
    ) -> tuple[bool, str, str]:
        history_text = self._format_query_context(context_queries)
        retrieved_section = ""
        if retrieved_content:
            retrieved_section = f"\nRetrieved so far:\n\n{retrieved_content}\n"

        prompt = PRE_RETRIEVAL_USER_PROMPT
        user_prompt = prompt.format(
            new_message=self._escape_prompt_value(new_message),
            conversation_history=self._escape_prompt_value(history_text),
            retrieved_section=self._escape_prompt_value(retrieved_section),
        )

        sys_prompt = system_prompt or _system_prompt_for_angle(
            0,
            include_mental_health_query=include_mental_health_query,
        )
        client = llm_client or self._select_chat_client(None)
        response = await client.chat(user_prompt, system_prompt=sys_prompt)
        active_query = self._extract_active_query(response)
        if not require_decision:
            if not active_query:
                raise ValueError("retrieval query missing active_query")
            return True, active_query, response

        decision = self._extract_decision(response)
        if decision == "RETRIEVE" and not active_query:
            raise ValueError("retrieval decision missing active_query")

        # Caller can pull <mental_health_query> out of `response` if it cares.
        return decision == "RETRIEVE", active_query or "", response

    def _format_query_context(self, queries: list[dict[str, Any]] | None) -> str:
        if not queries:
            return "No query context."

        by_role: dict[str, list[str]] = {}
        passthrough: list[str] = []
        for q in queries:
            if not isinstance(q, dict):
                raise TypeError("INVALID_CONTEXT_QUERY")
            role = str(q.get("role", "user") or "user").strip()
            content = q.get("content")
            if isinstance(content, dict):
                text = str(content.get("text", "") or "").strip()
            elif isinstance(content, str):
                text = content.strip()
            else:
                raise TypeError("INVALID_CONTEXT_QUERY")
            if text:
                role_key = role.lower()
                if role_key in {
                    "identity_context",
                    "all_categories_summary",
                    "history",
                    "cross_conversation",
                    "memory_cache",
                    "intentions",
                }:
                    by_role.setdefault(role_key, []).append(text)
                else:
                    passthrough.append(f"- [{role}]:\n{text}")

        blocks: list[str] = []
        identity = "\n\n".join(by_role.get("identity_context", []))
        if identity:
            blocks.append(identity)
        all_categories = "\n\n".join(by_role.get("all_categories_summary", []))
        if all_categories:
            blocks.append(all_categories)
        history_text = "\n\n".join(by_role.get("history", []))
        if history_text:
            blocks.append(history_text)
        cross_text = "\n\n".join(by_role.get("cross_conversation", []))
        if cross_text:
            blocks.append(cross_text)
        working_text = "\n".join(by_role.get("memory_cache", []))
        if working_text:
            blocks.append(f"My Working Thoughts:\n{working_text}")
        intentions_text = "\n\n".join(by_role.get("intentions", []))
        if intentions_text:
            blocks.append(f"My Intentions:\n{intentions_text}")
        blocks.extend(passthrough)

        if not blocks:
            return "No query context."

        return "\n\n".join(blocks)

    @staticmethod
    def _extract_query_text(query: dict[str, Any]) -> str:
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
        if not raw or not str(raw).strip():
            raise ValueError("sufficiency check returned empty response")

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

    def _extract_active_query(self, raw: str) -> str | None:
        match = re.search(r"<active_query>(.*?)</active_query>", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"<active_query>(.*?)(?:</[^>]*query>|<mental_health_query>|$)", raw, re.IGNORECASE | re.DOTALL)
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
        for cid, _score in hits:
            cat = category_pool.get(cid)
            if not cat:
                continue
            summary = summaries.get(cid) or cat.summary or ""
            text = str(summary or "").strip()
            if not text:
                text = f"# {cat.name}"
            lines.append(text)
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
