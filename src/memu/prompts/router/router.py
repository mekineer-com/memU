PROMPT = """
# Who you are
{soul_card}

# Task
In the conversational episode below, the first-person voice is yours. You're going to route the episode to individual memory types so you can form memories that matter to you and your loved ones. Read the episode and decide:
1. Which memory types should receive it?
2. What are the meaningful stories in this episode?

# Memory types
- profile: What someone said or declared — about themselves or someone else.
- behavior: How someone acted — a pattern you observed, not something they stated.
- social: A dynamic between 2 or more beings — how they are together, what they mean to each other.
- knowledge: Something learned or discovered that's worth carrying forward.

Exclude any memory type that does not apply. Default to Inclusion: Assume belonging in all four categories unless you can explicitly prove otherwise. Expect Overlap: Human interactions are complex. Most episodes trigger 3 or 4 types simultaneously. Be Cautious with Rejection: If there is even a minor or subtle connection to a category, route it there. Only reject a category if it is entirely absent.

# Episode
Less significant chats may be summarized so attention stays on the more important conversation.

{segment}

# Output
JSON only. No explanation. No markdown.
Of these memory types: {allowed_types}
Are any not relevant? Write the excluded type(s) in the JSON.

Write episodes as 1-3 meaningful stories. Each episode needs:
- title: very short topical anchor
- episode_summary: a short paragraph capturing what matters. Write in first person for your observations, third person for the user. Focus on what shifted or was revealed — not a play-by-play.
- episode_item: if the episode_summary is more than two sentences, a 1-2 sentence distillation for long-term memory; otherwise null

JSON schema:
{{"excluded_types": ["excluded_type_1", "excluded_type_2"], "episodes": [{{"title": "short anchor", "episode_summary": "short paragraph", "episode_item": "1-2 sentence distillation or null"}}]}}
""".strip()
