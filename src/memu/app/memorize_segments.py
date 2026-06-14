from __future__ import annotations

import json
import logging
import pathlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from memu.utils.conversation import (
    conversation_message_indices,
    format_chat_messages,
    format_conversation_for_preprocess,
    format_grouped_chat_history,
    format_relative_time_label,
)
from memu.utils.video import VideoFrameExtractor

logger = logging.getLogger(__name__)


async def _split_into_episodes(
    *,
    local_path: str,
    text: str | None,
    modality: str,
    memorize_config: Any,
    preprocess_prompts: Mapping[str, str],
    resolve_custom_prompt: Callable[[Any, Mapping[str, str]], str],
    prepare_audio_text: Callable[..., Awaitable[str | None]],
    modality_requires_text: Callable[[str], bool],
    dispatch_preprocessor: Callable[..., Awaitable[list[dict[str, Any]]]],
    llm_client: Any | None = None,
) -> list[dict[str, Any]]:
    configured_prompt = memorize_config.multimodal_preprocess_prompts.get(modality)
    if configured_prompt is None:
        template = preprocess_prompts.get(modality)
    elif isinstance(configured_prompt, str):
        template = configured_prompt
    else:
        template = resolve_custom_prompt(configured_prompt, {})

    if not template:
        return [{"text": text, "caption": None}]

    if modality == "audio":
        text = await prepare_audio_text(local_path, text, llm_client=llm_client)
        if text is None:
            return [{"text": None, "caption": None}]

    if modality_requires_text(modality) and not text:
        return [{"text": text, "caption": None}]

    return await dispatch_preprocessor(
        modality=modality,
        local_path=local_path,
        text=text,
        template=template,
        llm_client=llm_client,
    )


async def _prepare_audio_text(
    local_path: str,
    text: str | None,
    *,
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
) -> str | None:
    if text:
        return text

    audio_extensions = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    text_extensions = {".txt", ".text", ".jsonl"}
    file_ext = pathlib.Path(local_path).suffix.lower()

    if file_ext in audio_extensions:
        try:
            client = llm_client or get_llm_client()
            transcribed = await client.transcribe(local_path)
        except Exception:
            logger.exception("Audio transcription failed for %s", local_path)
            return None
        return str(transcribed)

    if file_ext in text_extensions:
        path_obj = pathlib.Path(local_path)
        try:
            text_content = path_obj.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Failed to read text file %s", local_path)
            return None
        return text_content

    logger.warning("Unknown audio file type: %s, skipping transcription", file_ext)
    return None


def _modality_requires_text(modality: str) -> bool:
    return modality == "document"


async def _dispatch_preprocessor(
    *,
    modality: str,
    local_path: str,
    text: str | None,
    template: str,
    llm_client: Any | None,
    preprocess_video: Callable[..., Awaitable[list[dict[str, str | None]]]],
    preprocess_image: Callable[..., Awaitable[list[dict[str, str | None]]]],
    preprocess_document: Callable[..., Awaitable[list[dict[str, str | None]]]],
    preprocess_audio: Callable[..., Awaitable[list[dict[str, str | None]]]],
) -> list[dict[str, Any]]:
    if modality == "video":
        return await preprocess_video(local_path, template, llm_client=llm_client)
    if modality == "image":
        return await preprocess_image(local_path, template, llm_client=llm_client)
    if modality == "document" and text is not None:
        return await preprocess_document(text, template, llm_client=llm_client)
    if modality == "audio" and text is not None:
        return await preprocess_audio(text, template, llm_client=llm_client)
    return [{"text": text, "caption": None}]


async def _summarize_segment(
    *,
    segment_text: str,
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
) -> str | None:
    system_prompt = (
        "Summarize the given conversational episode in 1-2 concise sentences. "
        "Focus on the main topic or theme discussed."
    )
    try:
        client = llm_client or get_llm_client(
            step_context={"operation": "memorize", "step_id": "segment_summary"},
        )
        response = await client.chat(segment_text, system_prompt=system_prompt)
        return response.strip() if response else None
    except Exception:
        logger.exception("Failed to summarize segment")
        return None


async def _preprocess_video(
    *,
    local_path: str,
    template: str,
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
    parse_multimodal_response: Callable[[str, str, str], tuple[str | None, str | None]],
) -> list[dict[str, str | None]]:
    try:
        if not VideoFrameExtractor.is_ffmpeg_available():
            logger.warning("ffmpeg not available, cannot process video. Returning None.")
            return [{"text": None, "caption": None}]

        frame_path = VideoFrameExtractor.extract_middle_frame(local_path)

        try:
            client = llm_client or get_llm_client()
            processed = await client.vision(prompt=template, image_path=frame_path, system_prompt=None)
            description, caption = parse_multimodal_response(processed, "detailed_description", "caption")
            return [{"text": description, "caption": caption}]
        finally:
            try:
                pathlib.Path(frame_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to clean up frame %s: %s", frame_path, exc)

    except (OSError, RuntimeError) as exc:
        logger.error("Video preprocessing failed: %s", exc, exc_info=True)
        return [{"text": None, "caption": None}]


async def _preprocess_image(
    *,
    local_path: str,
    template: str,
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
    parse_multimodal_response: Callable[[str, str, str], tuple[str | None, str | None]],
) -> list[dict[str, str | None]]:
    client = llm_client or get_llm_client()
    processed = await client.vision(prompt=template, image_path=local_path, system_prompt=None)
    description, caption = parse_multimodal_response(processed, "detailed_description", "caption")
    return [{"text": description, "caption": caption}]


async def _preprocess_document(
    *,
    text: str,
    template: str,
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
    escape_prompt_value: Callable[[str], str],
    parse_multimodal_response: Callable[[str, str, str], tuple[str | None, str | None]],
) -> list[dict[str, str | None]]:
    prompt = template.format(document_text=escape_prompt_value(text))
    client = llm_client or get_llm_client()
    processed = await client.chat(prompt)
    processed_content, caption = parse_multimodal_response(processed, "processed_content", "caption")
    return [{"text": processed_content or text, "caption": caption}]


async def _preprocess_audio(
    *,
    text: str,
    template: str,
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
    escape_prompt_value: Callable[[str], str],
    parse_multimodal_response: Callable[[str, str, str], tuple[str | None, str | None]],
) -> list[dict[str, str | None]]:
    prompt = template.format(transcription=escape_prompt_value(text))
    client = llm_client or get_llm_client()
    processed = await client.chat(prompt)
    processed_content, caption = parse_multimodal_response(processed, "processed_content", "caption")
    return [{"text": processed_content or text, "caption": caption}]


def _estimate_text_tokens(text: str) -> int:
    words = len((text or "").split())
    if words < 1:
        return 0
    return int(words / 0.75)


def _compute_batch_max_items(total_message_count: int) -> int:
    if total_message_count >= 80:
        return 12
    if total_message_count >= 40:
        return 8
    return 6


def _message_is_primary_for_memorize(message: Mapping[str, Any]) -> bool:
    flag = message.get("memorize_chat")
    if isinstance(flag, bool):
        return flag
    return True


def _message_index_for_sort(message: Mapping[str, Any]) -> int:
    raw = message.get("_message_index")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _summary_row_lines(row: Mapping[str, Any]) -> list[str]:
    summary = str(row.get("summary") or "").strip()
    if not summary:
        return []
    if "\n" not in summary:
        return [summary]
    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    return lines


def _grouped_chat_timestamp(message: Mapping[str, Any]) -> Any:
    return message.get("received_at") or message.get("ts_ms") or message.get("created_at")


def _prepare_grouped_chat_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for message in sorted(messages, key=_message_index_for_sort):
        row = dict(message)
        conversation_id = str(row.get("source_conversation_id") or row.get("conversation_id") or "").strip()
        if conversation_id:
            row["conversation_id"] = conversation_id
        timestamp = _grouped_chat_timestamp(row)
        if timestamp and "received_at" not in row:
            row["received_at"] = timestamp
        prepared.append(row)
    return prepared


def _render_grouped_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    soul_name: str | None = None,
) -> str:
    return format_grouped_chat_history(
        _prepare_grouped_chat_messages(messages),
        time_label_resolver=format_relative_time_label,
        soul_name=soul_name,
    )


def _render_with_summary_rows(
    *,
    primary_messages: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    soul_name: str | None = None,
) -> str:
    prefix_lines: list[str] = []
    for row in summary_rows:
        prefix_lines.extend(_summary_row_lines(row))
    primary_rendered = _render_grouped_chat_messages(primary_messages, soul_name=soul_name)
    parts = ["\n".join(prefix_lines).strip(), primary_rendered]
    return "\n\n".join(part for part in parts if part).strip()


async def _summarize_background_messages(
    *,
    messages: Sequence[Mapping[str, Any]],
    llm_client: Any | None,
    summarize_segment: Callable[..., Awaitable[str | None]],
    soul_name: str | None = None,
) -> str | None:
    if not messages:
        return None
    rendered = format_chat_messages(
        sorted(messages, key=_message_index_for_sort),
        soul_name=soul_name,
        default_role="user",
    )
    if not rendered:
        return None
    summary = await summarize_segment(rendered, llm_client=llm_client)
    return str(summary or "").strip() or None


async def _summarize_background_rollup(
    *,
    prior_summary: str | None,
    messages: Sequence[Mapping[str, Any]],
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
    soul_name: str | None = None,
) -> str:
    if not messages:
        msg = "background rollup requires at least one message"
        raise ValueError(msg)
    rendered = format_chat_messages(
        sorted(messages, key=_message_index_for_sort),
        soul_name=soul_name,
        default_role="user",
    )
    if not rendered:
        msg = "background rollup rendered empty message payload"
        raise ValueError(msg)
    prior_block = str(prior_summary or "").strip()
    prompt_parts = [
        "Prior rolling summary (if present):",
        prior_block or "<none>",
        "",
        "New unsummarized tail:",
        rendered,
    ]
    prompt = "\n".join(prompt_parts).strip()
    system_prompt = (
        "You maintain a rolling summary for one background chat. "
        "Return one concise paragraph that merges prior summary + new tail. "
        "Preserve names, concrete facts, quoted phrases, and references. "
        "Drop pleasantries and filler. No bullets. No markdown."
    )
    client = llm_client or get_llm_client(
        step_context={"operation": "memorize", "step_id": "background_rollup"},
    )
    response = await client.chat(prompt, system_prompt=system_prompt)
    summary = str(response or "").strip()
    if not summary:
        msg = "background rollup returned empty summary"
        raise ValueError(msg)
    return summary


async def _summarize_background_groups_batched(
    *,
    grouped_messages: Mapping[str, Sequence[Mapping[str, Any]]],
    group_order: Sequence[str],
    llm_client: Any | None,
    get_llm_client: Callable[..., Any],
    extract_json_blob: Callable[[str], str],
    soul_name: str | None = None,
) -> dict[str, str]:
    batches: list[dict[str, Any]] = []
    for source_key in group_order:
        messages = grouped_messages.get(source_key) or []
        if not messages:
            continue
        source_label = str(messages[0].get("source_label") or source_key).strip() or source_key
        rendered = format_chat_messages(
            sorted(messages, key=_message_index_for_sort),
            soul_name=soul_name,
            default_role="user",
        )
        if not rendered:
            continue
        batches.append(
            {
                "source_key": source_key,
                "source_label": source_label,
                "messages": rendered,
            }
        )
    if not batches:
        return {}
    prompt = json.dumps({"groups": batches}, ensure_ascii=False, indent=2)
    system_prompt = (
        "Summarize each background group independently. "
        "Return strict JSON only: {\"summaries\":[{\"source_key\":\"...\",\"summary\":\"...\"}]}. "
        "Each summary must be 1-2 concise sentences preserving names, facts, references, and quoted phrases. "
        "Drop filler and pleasantries."
    )
    client = llm_client or get_llm_client(
        step_context={"operation": "memorize", "step_id": "background_batch_summary"},
    )
    raw = await client.chat(prompt, system_prompt=system_prompt)
    try:
        payload = json.loads(str(raw or ""))
    except json.JSONDecodeError:
        payload = json.loads(extract_json_blob(str(raw or "")))
    summaries = payload.get("summaries") if isinstance(payload, Mapping) else None
    if not isinstance(summaries, list):
        msg = "background batch summary returned invalid JSON schema"
        raise ValueError(msg)
    out: dict[str, str] = {}
    for row in summaries:
        if not isinstance(row, Mapping):
            continue
        source_key = str(row.get("source_key") or "").strip()
        summary = str(row.get("summary") or "").strip()
        if not source_key or not summary:
            continue
        out[source_key] = summary
    expected_keys = {str(batch.get("source_key") or "").strip() for batch in batches}
    missing_keys = sorted(key for key in expected_keys if key and key not in out)
    if missing_keys:
        msg = f"background batch summary missing source keys: {', '.join(missing_keys)}"
        raise ValueError(msg)
    return out


async def _render_episode_with_background_context(
    *,
    primary_messages: Sequence[Mapping[str, Any]],
    background_messages: Sequence[Mapping[str, Any]],
    llm_client: Any | None,
    summarize_background_groups_batched: Callable[..., Awaitable[dict[str, str]]],
    memorize_config: Any,
    soul_name: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    prim = sorted(primary_messages, key=_message_index_for_sort)
    bg = sorted(background_messages, key=_message_index_for_sort)
    if not prim and not bg:
        return "", []
    if not bg:
        return _render_grouped_chat_messages(prim, soul_name=soul_name), []

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    group_order: list[str] = []
    for msg in bg:
        source_key = str(msg.get("source_conversation_id") or msg.get("source_label") or "background").strip() or "background"
        if source_key not in grouped:
            grouped[source_key] = []
            group_order.append(source_key)
        grouped[source_key].append(msg)

    total_bg_tokens = sum(_estimate_text_tokens(str(msg.get("content") or "")) for msg in bg)
    raw_floor = int(getattr(memorize_config, "background_extra_messages_tokens", 100) or 100)
    summarize_background = total_bg_tokens >= max(0, raw_floor)

    batched_summaries: dict[str, str] = {}
    if summarize_background:
        batched_summaries = await summarize_background_groups_batched(
            grouped_messages=grouped,
            group_order=group_order,
            llm_client=llm_client,
            soul_name=soul_name,
        )

    summary_rows: list[dict[str, Any]] = []
    for source_key in group_order:
        group_msgs = grouped[source_key]
        if summarize_background:
            summary = str(batched_summaries.get(source_key) or "").strip()
            if not summary:
                msg = f"missing background summary for source '{source_key}'"
                raise ValueError(msg)
            summary_lines = [summary]
        else:
            rendered_group = _render_grouped_chat_messages(group_msgs, soul_name=soul_name)
            summary_lines = rendered_group.splitlines()
            if not summary_lines:
                continue
        source_label = str(group_msgs[0].get("source_label") or source_key).strip() or source_key
        first_idx = _message_index_for_sort(group_msgs[0])
        after_index = None
        for primary in prim:
            pidx = _message_index_for_sort(primary)
            if pidx <= first_idx:
                after_index = pidx
            else:
                break
        summary_rows.append(
            {
                "after_index": after_index,
                "summary": "\n".join(summary_lines),
                "source_label": source_label,
            }
        )

    return _render_with_summary_rows(
        primary_messages=prim,
        summary_rows=summary_rows,
        soul_name=soul_name,
    ), summary_rows


def _render_episode_with_summary_rows(
    *,
    primary_messages: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    soul_name: str | None = None,
) -> str:
    return _render_with_summary_rows(
        primary_messages=sorted(primary_messages, key=_message_index_for_sort),
        summary_rows=[dict(row) for row in summary_rows if isinstance(row, Mapping)],
        soul_name=soul_name,
    )


def _prepare_episode(
    *,
    modality: str,
    text: str | None,
    message_indices: Any,
    dedupe_message_indices: Callable[[Sequence[int | float | str]], list[int]],
    extract_message_indices: Callable[[str | None], list[int]],
) -> tuple[str | None, list[int]]:
    if modality != "conversation":
        return None, []
    if isinstance(message_indices, list) and message_indices:
        indices = dedupe_message_indices([
            value for value in message_indices if isinstance(value, (int, float, str))
        ])
        segment_text = None
        if isinstance(text, str) and text.strip():
            segment_text = format_conversation_for_preprocess(text)
            if not segment_text.strip():
                segment_text = text.strip()
        return segment_text, indices
    if not isinstance(text, str) or not text.strip():
        return None, []
    segment_text = format_conversation_for_preprocess(text)
    if not segment_text.strip():
        segment_text = text.strip()
    indices = extract_message_indices(segment_text)
    if not indices:
        indices = conversation_message_indices(text)
    return segment_text, indices


def _parse_multimodal_response(
    raw: str,
    content_tag: str,
    caption_tag: str,
    *,
    extract_tag_content: Callable[[str, str], str | None],
) -> tuple[str | None, str | None]:
    content = extract_tag_content(raw, content_tag)
    caption = extract_tag_content(raw, caption_tag)
    if not content:
        content = raw.strip()
    if not caption and content:
        first_sentence = content.split(".")[0]
        caption = first_sentence if len(first_sentence) <= 200 else first_sentence[:200]
    return content, caption


def _extract_tag_content(raw: str, tag: str) -> str | None:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(raw)
    if not match:
        return None
    content = match.group(1).strip()
    return content or None
