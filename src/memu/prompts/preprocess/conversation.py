PROMPT = """
# Task Objective
Produce work for an AI memory system. You will analyze a conversation with message indices and divide it into meaningful episodes based on topic changes or natural breaks, like chapters of a book. Split at clear boundaries.

# Rules
- Maximum {episodes_per_segment} episodes per conversation segment. If there are more natural breaks than that, merge the least distinct ones.
- Use only the provided `[INDEX]` numbers.
- Do not overlap episodes.
- Do not include explanations, comments, or extra text in the final output.
- Record the `start` and `end` indices (inclusive) for each episode.

# Output Format
Return **only valid JSON** in the following structure:

```json
{{
    "episodes": [
        {{"start": x, "end": x}},
        {{"start": x, "end": x}}
    ]
}}
```

# Input
Conversation Content:
{conversation}
"""

CROSS_CONVERSATION_PROMPT = """
# Task Objective
Analyze messages from multiple conversation sources and organize them into episodes by topic or storyline.

Messages are tagged with their source and scope markers:
- `[primary]` => this conversation should be extracted for memories
- `[background]` => context-only conversation (do not extract memories from this chat directly)

Messages about the same topic from different sources belong in the same episode.

# Rules
- Every message must be assigned to exactly one episode. No message may be left out.
- Maximum {episodes_per_segment} episodes. If there are more natural topics, merge the least distinct ones.
- Group by topic/storyline, NOT by chat source. A WhatsApp message and a SillyTavern message about the same subject go together.
- Use the provided `[INDEX]` numbers. List all indices belonging to each episode in `message_indices`.
- Indices need not be contiguous — messages from different sources interleave.
- For background messages, provide concise inline summaries in `background_summaries` so the episode keeps context without raw background transcript.
- `background_summaries` items must include:
  - `after_index`: index after which the summary should appear (or `null` for preface)
  - `summary`: one concise sentence.
- Do not include explanations, comments, or extra text in the final output.

# Output Format
Return **only valid JSON**:

```json
{{
    "episodes": [
        {{
            "message_indices": [0, 1, 4, 7],
            "caption": "brief topic summary",
            "background_summaries": [
                {{"after_index": 1, "summary": "Context from customer support: issue resolved."}}
            ]
        }},
        {{
            "message_indices": [2, 3, 5, 6],
            "caption": "brief topic summary",
            "background_summaries": []
        }}
    ]
}}
```

# Input
Conversation Content:
{conversation}
"""
