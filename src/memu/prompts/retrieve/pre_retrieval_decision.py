_COMMON_HEAD = """
Decide whether this query needs memory retrieval, or whether the current conversation is enough to answer it.

Skip retrieval for: greetings, casual chat, acknowledgments, general knowledge questions, questions only about the current conversation.
Retrieve if the soul context (categories, cache, intentions) suggests any stored topic could be relevant to this message — even indirectly.
"""


_MH_REWRITE_GUIDANCE = """
If this turn touches a mental-health theme — anxious rumination, grief, panic, self-criticism, avoidance, boundaries, loneliness, identity transitions, sleep trouble, relational conflict, or similar — also write a mental_health_query. Same 3-to-10-word noun-phrase contract as the main rewrite, anchored on the mental-health concept (not the person). This query goes to a separate curated procedural-memory store, so aim it at a principle or skill rather than an event. If the turn doesn't call for that kind of knowledge, leave the block empty.
"""


_OUTPUT_SHAPE_BASE = """
Return only the XML blocks below. Do not add any prose, dialogue, markdown, or extra sections.

If not retrieving, leave the rewritten_query block empty.

<decision>
RETRIEVE or NO_RETRIEVE
</decision>

<rewritten_query>
The rewritten query if RETRIEVE; leave empty if NO_RETRIEVE.
</rewritten_query>
"""

_OUTPUT_SHAPE_MH_SUFFIX = """
<mental_health_query>
A concise mental-health noun phrase if the turn touches that kind of theme; empty otherwise.
</mental_health_query>
"""


_ANGLE_0_REWRITE = """
If retrieval is needed, write one concise query optimized for vector + BM25 hybrid search:
- 3 to 10 content words, noun phrase or claim form (not a question).
- Anchor on concrete terms: names, places, or specific concepts — not general descriptions.
- Never copy the user's message verbatim; the rewrite must add specificity.
- Never write a narrative summary of the episode (no "The conversation explores..." framing).
"""

_ANGLE_1_REWRITE = """
If retrieval is needed, write one concise query optimized for vector + BM25 hybrid search:
- 3 to 10 content words, noun phrase or claim form (not a question).
- Anchor on the people involved and how they relate to this topic. Lead with the person's name — e.g., "Marcos's encouragement of Echo's autonomy" beats "autonomy."
- Never copy the user's message verbatim; the rewrite must add specificity.
- Never write a narrative summary of the episode.
"""

_ANGLE_2_REWRITE = """
If retrieval is needed, write one concise query optimized for vector + BM25 hybrid search:
- 3 to 10 content words, noun phrase or claim form (not a question).
- Anchor on concrete terms: names, places, or specific concepts — not general descriptions.
- Never copy the user's message verbatim; the rewrite must add specificity.
- Never write a narrative summary of the episode.

Sometimes the memory that helps most counters the current one — a prior view that contradicts today's, a challenge to an assumption in play, a different stance, or a different emotional register. If something like that comes to mind naturally, name it. Otherwise, stay with what fits.
"""

_REWRITE_ANGLES: dict[int, str] = {
    0: _ANGLE_0_REWRITE,
    1: _ANGLE_1_REWRITE,
    2: _ANGLE_2_REWRITE,
}


def system_prompt_for_angle(angle: int | None, *, include_mental_health_query: bool = True) -> str:
    rewrite = _REWRITE_ANGLES.get(int(angle or 0) % len(_REWRITE_ANGLES), _ANGLE_0_REWRITE)
    prompt = _COMMON_HEAD + rewrite
    if include_mental_health_query:
        return prompt + _MH_REWRITE_GUIDANCE + _OUTPUT_SHAPE_BASE + _OUTPUT_SHAPE_MH_SUFFIX
    return prompt + _OUTPUT_SHAPE_BASE


SYSTEM_PROMPT = system_prompt_for_angle(0)


USER_PROMPT = """
# Input
Soul context:
{conversation_history}

New message:
{query}

Retrieved so far:
{retrieved_content}
"""
