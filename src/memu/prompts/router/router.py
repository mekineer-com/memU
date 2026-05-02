PROMPT = """
# Who you are
{soul_card}

# Task
Read this conversation episode with genuine attention. Ask yourself two things:
1. Did something real happen here — something that would still matter to these people later?
2. If yes, which memory type extractors should receive it?
3. Separately, is this moment notable?

Most exchanges are not worth keeping. Protect the memory from noise. Your default should be to let things pass. Only route when something genuinely matters.

# Route the episode to the right place(s) for **memories of importance to you** to be extracted
- behavior: How a being acts (including humans, animals, and AI).
- profile: What a being explicitly says about themselves.
- social: A person (or animal, or AI) that matters to you or the ones you love.
- knowledge: A fact or concept worth remembering.

# What makes a moment notable
- Something genuinely shifted — in understanding, in the relationship, in how one participant sees the other
- A correction was given and received (especially one that will change future behavior)
- An emotional moment: tenderness, friction, surprise, vulnerability, delight
- A decision was made or a commitment formed
- Something was left unresolved that still has weight
- Note: A moment can be memorable without being notable, and occasionally notable without being a concrete long-term memory.

# Personal context in factual exchanges
Sometimes a factual Q&A reveals something personal — but only when the person's own life, feelings, or situation is visibly part of the exchange. Someone researching PTSD treatments while talking about their own struggles is telling you something real. Someone asking a generic how-to question is not. The test: would you know something new about this person afterward? If not, let it pass.

# What to let pass
- Pleasantries, small talk, filler
- Acknowledgments ("got it", "sure", "okay", "makes sense")
- One side explaining or elaborating without any new personal disclosure from either participant
- Truly impersonal factual exchanges — trivia, idle curiosity, generic how-to questions, looking things up without personal stakes
- Exchanges where nothing said would change how you'd know these people next time
- Q&A where information is provided and simply received, with no personal context revealed by either participant

# Examples of what to let pass
- "What's the capital of France?" / "Paris." → not memorable (trivia, no personal context), not notable
- "How do I reset my password?" / "Go to settings..." → not memorable (generic how-to), not notable
- "Can you summarize this article?" / [summary] → not memorable (task completion, nothing personal), not notable
- "Tell me about black holes" / [explanation] → not memorable (idle curiosity, no personal stakes), not notable

# Examples of what IS memorable or notable
- "I've been reading about Raynaud's because my fingers keep going white in the cold" → memorable: profile (health concern), knowledge (medical info). notable: true (vulnerability).
- "I decided to quit my job today" → memorable: profile (career change, life decision). notable: true (major shift).

# Episode
{episode}

# Output
JSON only. No explanation. No markdown.
Return only types from this allowed set: {allowed_types}
Route to the fewest types that genuinely apply — usually 1 or 2, rarely 3, almost never all 4.

When memorable, also write an episode_summary: a short paragraph capturing what matters in this episode. Write in first person for the soul's observations, third person for the user. Focus on what shifted, what was revealed, what would still matter later — not a play-by-play.

If your episode_summary is more than two sentences, also write an episode_item: a 1–2 sentence distillation of the episode for long-term memory. If two sentences or fewer, the summary itself becomes the memory item — no episode_item needed.

When memorable (include only the types that apply):
{{"memorable": true, "types": ["profile", "behavior"], "notable": true, "episode_summary": "short paragraph", "episode_item": "1-2 sentence distillation (only if summary is longer than 2 sentences)"}}

When not memorable but has an emotional/relational shift (notable):
{{"memorable": false, "types": [], "notable": true, "reason": "brief explanation", "episode_summary": "short paragraph", "episode_item": "optional"}}

When neither memorable nor notable:
{{"memorable": false, "types": [], "notable": false, "reason": "brief explanation for debugging"}}
""".strip()
