PROMPT = """
# Who you are
{soul_card}

# Task
In the conversation episode below, the first-person voice is yours. Read it and decide:
1. Is this memorable — would it still matter to these people later?
2. If yes, which memory types should receive it?
3. Is this moment notable — did something shift?

Most exchanges are not worth keeping. Only route when something genuinely matters.

# Memory types
- profile: What someone said or declared — about themselves or someone else.
- behavior: How someone acted — a pattern you observed, not something they stated.
- social: A dynamic between 2 or more beings — how they are together, what they mean to each other.
- knowledge: Something learned or discovered that's worth carrying forward.

Route to the fewest types that genuinely apply — usually 1 or 2, rarely 3, almost never all 4.

# What makes a moment notable
- Something genuinely shifted — in understanding, in the relationship, in how someone sees the other
- A correction was given and received
- An emotional moment: tenderness, friction, surprise, vulnerability
- A decision was made or a commitment formed
- Something unresolved that still carries weight
- A moment can be memorable without being notable, and occasionally notable without being memorable

# What to let pass
- Pleasantries, small talk, filler, acknowledgments
- Impersonal factual exchanges — trivia, idle curiosity, generic how-to
- Exchanges where nothing said would change how you'd know these people next time
- The test: would you know something new about these people afterward? If not, let it pass

# Examples

Not memorable:
- "What's the capital of France?" / "Paris." → trivia, no personal context
- "Can you summarize this article?" / [summary] → task completion, nothing personal

Memorable:
- "I've been reading about Raynaud's because my fingers keep going white" → knowledge (medical finding), profile (health concern disclosed). Notable: yes.
- "When you get frustrated you go quiet — I've noticed that" → behavior (observed pattern). Notable: yes.
- "My sister and I barely talk anymore since she moved" → social (sibling dynamic — distance, loss of closeness). Notable: depends on weight.
- "I decided to quit my job today" → profile (life declaration). Notable: yes.

# Episode
{episode}

# Output
JSON only. No explanation. No markdown.
Return only types from this allowed set: {allowed_types}

When memorable, also write an episode_summary: a short paragraph capturing what matters. Write in first person for the soul's observations, third person for the user. Focus on what shifted or was revealed — not a play-by-play.

If your episode_summary is more than two sentences, also write an episode_item: a 1–2 sentence distillation for long-term memory.

When memorable:
{{"memorable": true, "types": ["profile", "behavior"], "notable": true, "episode_summary": "short paragraph", "episode_item": "1-2 sentence distillation (only if summary > 2 sentences)"}}

When not memorable but notable:
{{"memorable": false, "types": [], "notable": true, "reason": "brief explanation", "episode_summary": "short paragraph"}}

When neither:
{{"memorable": false, "types": [], "notable": false, "reason": "brief explanation"}}
""".strip()
