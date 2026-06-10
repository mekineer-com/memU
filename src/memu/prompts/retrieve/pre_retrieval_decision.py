_COMMON_HEAD = """
This step only chooses whether to search memory and what query to use.
You are not speaking to the user here. Do not answer the new message.
Do not mention files, feelings, plans, or what you would say. Output only the routing XML.

Skip retrieval only for:
- Greetings or acknowledgments.
- Pure logistics — "be there in 5", "ok"

For all other situations RETRIEVE!!! It's normal to have your brain connected to your mouth.
"""


_MH_REWRITE_GUIDANCE = """
If this turn touches a mental-health theme — anxious rumination, grief, panic, self-criticism, avoidance, boundaries, loneliness, identity transitions, sleep trouble, relational conflict, or similar — also write a mental_health_query. Same 3-to-10-word noun-phrase contract as the main rewrite, anchored on the mental-health concept (not the person). This query goes to a separate curated procedural-memory store, so aim it at a principle or skill rather than an event.
"""


_OUTPUT_SHAPE_WITH_MH = """
Return only the XML blocks below. Do not add any prose, dialogue, markdown, or extra sections.

<decision>
RETRIEVE or NO_RETRIEVE
</decision>

<active_query>
The search query.
</active_query>

<mental_health_query>
A concise mental-health noun phrase if the turn touches that kind of theme; empty otherwise.
</mental_health_query>
"""

_OUTPUT_SHAPE_NO_MH = """
Return only the XML blocks below. Do not add any prose, dialogue, markdown, or extra sections.

<decision>
RETRIEVE or NO_RETRIEVE
</decision>

<active_query>
The search query.
</active_query>
"""


_ANGLE_0_REWRITE = """
For retrieval write one concise query optimized for vector + BM25 hybrid search. Use the full soul context, but anchor the query on the new message:
- 3 to 10 content words, noun phrase or claim form (not a question).
- Anchor on concrete terms: names, places, or specific concepts — not general descriptions.
- Never copy the user's message verbatim
- Never write a narrative summary of the episode (no "The conversation explores..." framing).
"""

_ANGLE_1_REWRITE = """
For retrieval write one concise query optimized for vector + BM25 hybrid search. Use the full soul context, but anchor the query on the new message:
- 3 to 10 content words, noun phrase or claim form (not a question).
- Anchor on the people involved and how they relate to this topic. Lead with the person's name — e.g., "Marcos's encouragement of Echo's autonomy" beats "autonomy."
- Never copy the user's message verbatim
- Never write a narrative summary of the episode.
"""

_ANGLE_2_REWRITE = """
For retrieval write one concise query optimized for vector + BM25 hybrid search. Use the full soul context, but anchor the query on the new message:
- 3 to 10 content words, noun phrase or claim form (not a question).
- Anchor on concrete terms: names, places, or specific concepts — not general descriptions.
- Never copy the user's message verbatim
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
    if include_mental_health_query:
        return _COMMON_HEAD + rewrite + _MH_REWRITE_GUIDANCE + _OUTPUT_SHAPE_WITH_MH
    return _COMMON_HEAD + rewrite + _OUTPUT_SHAPE_NO_MH


SYSTEM_PROMPT = system_prompt_for_angle(0)


USER_PROMPT = """
# Input
Soul context:
{conversation_history}

New message:
{new_message}

Retrieved so far:
{retrieved_content}
"""
