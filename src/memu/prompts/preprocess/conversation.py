PROMPT = """
# Task Objective
Analyze a conversation with message indices and divide it into meaningful segments.

# How to segment
Create a new segment when there is a clear:
- topic change
- long pause / time gap
- natural conclusion, then a new thread starts

# Important rules
- Use only the provided [INDEX] numbers.
- Segments must not overlap.
- `start` and `end` are inclusive.
- If the conversation has fewer than 20 messages, return exactly ONE segment from the first to the last message.
- Output **ONLY** JSON. No markdown, no explanations.

# Output JSON shape
{{
  "segments": [
    {{"start": 0, "end": 19}},
    {{"start": 20, "end": 45}}
  ]
}}

Conversation Content:
{conversation}
"""

