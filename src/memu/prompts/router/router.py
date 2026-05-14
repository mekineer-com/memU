PROMPT = """
# Who you are
{soul_card}

# Task
In the conversation episode below, the first-person voice is yours. You're going to route the episode to individual memory types so you can form memories that matter to you and your loved ones. Read the episode and decide:
1. Is this memorable — would it still matter to you later?
2. If yes, which memory types should receive it?

# Memory types
- profile: What someone said or declared — about themselves or someone else.
- behavior: How someone acted — a pattern you observed, not something they stated.
- social: A dynamic between 2 or more beings — how they are together, what they mean to each other.
- knowledge: Something learned or discovered that's worth carrying forward.

Exclude any memory type that does not apply. Default to Inclusion: Assume belonging in all four categories unless you can explicitly prove otherwise. Expect Overlap: Human interactions are complex. Most episodes trigger 3 or 4 types simultaneously. Be Cautious with Rejection: If there is even a minor or subtle connection to a category, route it there. Only reject a category if it is entirely absent.

# What isn't memorable
- Pleasantries, small talk, filler, acknowledgments
- Impersonal factual exchanges — trivia, idle curiosity, generic how-to
- Humdrum, monotonous, or routine periods of time with no emotion or discovery

# Episode
{episode}

# Output
JSON only. No explanation. No markdown.
Of these memory types: {allowed_types}
Are any not relevant? Write the excluded type(s) in the JSON.

When memorable, also write an episode_summary: a short paragraph capturing what matters. Write in first person for your observations, third person for the user. Focus on what shifted or was revealed — not a play-by-play.

If your episode_summary is more than two sentences, also write an episode_item: a 1–2 sentence distillation for long-term memory.

When memorable:
{{"memorable": true, "excluded_types": ["excluded_type_1", "excluded_type_2"], "episode_summary": "short paragraph", "episode_item": "1-2 sentence distillation (only if summary > 2 sentences)"}}

When not memorable:
{{"memorable": false, "reason": "brief explanation"}}
""".strip()
