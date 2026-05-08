from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from memu.database.vector import cosine_topk
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
        channel_mode: str | None = None,
    ) -> dict[str, Any]:
        if not queries:
            raise ValueError("empty_queries")
        ctx = self._get_context()
        store = self._get_database()
        original_query = self._extract_query_text(queries[-1])
        where_filters = self._normalize_where(where)

        context_queries = queries[:-1] if len(queries) > 1 else []
        route_context_queries, retrieval_context_queries = self._split_context_queries(context_queries)

        workflow_name = "retrieve_rag"

        state: WorkflowState = {
            "method": self.retrieve_config.method,
            "original_query": original_query,
            "context_queries": retrieval_context_queries,
            "route_context_queries": route_context_queries,
            "rewrite_angle": int(rewrite_angle) if rewrite_angle is not None else 0,
            "channel_mode": channel_mode,
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
                requires={"original_query", "context_queries"},
                produces={"needs_retrieval", "rewritten_query", "active_query", "next_step_query"},
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
            "ctx",
            "store",
            "where",
            "as_of",
        }

    async def _rag_route_intention(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        llm_client = self._get_step_llm_client(step_context)
        channel_mode = state.get("channel_mode")
        angle_prompt = _system_prompt_for_angle(state.get("rewrite_angle"), channel_mode=channel_mode)
        needs_retrieval, rewritten_query, raw_response = await self._decide_if_retrieval_needed(
            state["original_query"],
            state.get("route_context_queries", state["context_queries"]),
            retrieved_content=None,
            system_prompt=angle_prompt,
            llm_client=llm_client,
        )
        mental_health_query = self._extract_mental_health_query(raw_response)
        should_respond = self._extract_respond_decision(raw_response, channel_mode)

        if not should_respond:
            needs_retrieval = False

        state.update({
            "needs_retrieval": needs_retrieval,
            "should_respond": should_respond,
            "rewritten_query": rewritten_query,
            "active_query": rewritten_query,
            "mental_health_query": mental_health_query,
            "next_step_query": None,
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
                    provenance[t.subject_id] = f"found via entity '{entity.name}'"
        return memory_ids, provenance

    async def _rag_recall_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        if not state.get("needs_retrieval") or not state.get("proceed_to_items"):
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
            embed_client = self._get_step_embedding_client(step_context)
            qvec = (await embed_client.embed([state["active_query"]]))[0]
            state["query_vector"] = qvec
        state["resource_hits"] = cosine_topk(qvec, corpus, k=self.retrieve_config.resource.top_k)
        return state

    def _rag_build_context(self, state: WorkflowState, _: Any) -> WorkflowState:
        response = {
            "needs_retrieval": bool(state.get("needs_retrieval")),
            "should_respond": state.get("should_respond", True),
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

        prompt = PRE_RETRIEVAL_USER_PROMPT
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

        blocks: list[str] = []
        for q in queries:
            if isinstance(q, str):
                # Backward compatibility
                text = q.strip()
                if text:
                    blocks.append(f"- {text}")
                continue
            if isinstance(q, dict):
                role = str(q.get("role", "user") or "user").strip()
                content = q.get("content")
                if isinstance(content, dict):
                    text = str(content.get("text", "") or "").strip()
                elif isinstance(content, str):
                    text = content.strip()
                else:
                    text = str(content or "").strip()
                if not text:
                    continue
                if role.lower() == "identity_context":
                    blocks.append(text)
                    continue
                blocks.append(f"- [{role}]:\n{text}")
                continue
            text = str(q).strip()
            if text:
                blocks.append(f"- {text}")

        if not blocks:
            return "No query context."

        return "\n\n".join(blocks)

    @staticmethod
    def _split_context_queries(context_queries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return list(context_queries), list(context_queries)

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

    @staticmethod
    def _extract_respond_decision(raw: str, channel_mode: str | None) -> bool:
        match = re.search(r"<respond>(.*?)</respond>", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return "SPEAK" in match.group(1).strip().upper()
        return (channel_mode or "direct") != "group"

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
