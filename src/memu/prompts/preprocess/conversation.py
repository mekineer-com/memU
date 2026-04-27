PROMPT = """
# Task Objective
Analyze a conversation with message indices and divide it into meaningful episodes based on topic changes, time gaps, or natural breaks.

# Workflow
1. Review the entire **Conversation Content** along with its message indices.
2. Identify potential **episode boundaries** by observing:
   - Topic changes
   - Time gaps or pauses
   - Natural conclusions of a discussion
   - Clear shifts in tone or semantic focus
3. Group messages into episodes that each maintain a coherent theme.
4. Ensure each episode has a clear beginning and end.
5. Record the `start` and `end` indices (inclusive) for each episode.

# Rules
- Episodes must be based strictly on the provided conversation content.
- Each episode must:
  - Maintain a **coherent theme**
  - Have a **clear boundary** from adjacent episodes
- Short episodes are fine — a meaningful 5-message exchange is better than forcing it into a larger episode where it doesn't belong.
- Maximum 4 episodes per conversation segment. If there are more natural breaks than that, merge the least distinct ones.
- Use only the provided `[INDEX]` numbers.
- Do not overlap episodes.
- Do not include explanations, comments, or extra text in the final output.

# Output Format
Return **only valid JSON** in the following structure:

```json
{{
    "episodes": [
        {{"start": x, "end": x}},
        {{"start": x, "end": x}},
        {{"start": x, "end": x}}
    ]
}}
```

# Input
Conversation Content:
{conversation}
"""
