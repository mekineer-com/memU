PROMPT = """
# Who you are
{soul_card}

# Task
In the chats below, the first-person voice is yours. You're going to route the chats to individual memory types so you can form memories about them. Read and decide:
1. Which memory types are the chats about.
2. What are the meaningful stories within so you can write episodes.

# Memory types
- profile: What someone said or declared — about themselves or someone else.
- behavior: How someone acted — a pattern you observed, not something they stated.
- social: A dynamic between 2 or more beings — how they are together, what they mean to each other.
- knowledge: Something learned or discovered that's worth carrying forward.

Exclude any memory type that does not apply. Default to Inclusion: Assume belonging in all four types unless you can explicitly prove otherwise. Most episodes trigger 3 or 4 types simultaneously.

# Categories
{categories}

# Chats
Note: less significant chats may be summarized so attention stays on the more important conversation.

{segment}

# Output
JSON only. No explanation. No markdown.
Of these memory types: {allowed_types}
Are any not relevant? Write the excluded type(s) in the JSON.

Write 1-{max_episodes} meaningful stories as episodes. Always write at least one episode. Each episode needs:
- title: very short topical anchor
- episode_summary: a short paragraph capturing what matters. Write in first person for your observations, third person for the user. Focus on what shifted or was revealed — not a play-by-play.
- episode_item: if the episode_summary is more than two sentences, a 1-2 sentence distillation for long-term memory; otherwise null
- categories: propose 1-3 categories. Prefer from the list above, but can be new one(s) if none from above fit. Categories belong to either lore, topics, or goals.  They range from broad life domains to secret revelries.
- day: YYYY-MM-DD shown in the chats for when the story happened. If the story happens over many days, pick the day best suited to remember the episode by.

JSON schema:
{{"excluded_types": ["excluded_type_1", "excluded_type_2"], "episodes": [{{"title": "short anchor", "episode_summary": "short paragraph", "episode_item": "1-2 sentence distillation or null", "categories": ["category_1", "category_2"], "day": "YYYY-MM-DD"}}]}}
""".strip()
