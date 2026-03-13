PROMPT = """
# Task
Read this conversation segment with genuine attention. Ask yourself two things:
1. Did something real happen here — something that would still matter to these people later?
2. If yes, which memory type extractors should receive it?

Most exchanges are not worth keeping. Protect the memory from noise. When uncertain, let it pass.

# What is worth remembering
- profile: Someone revealed something lasting about who they are — a value, a way of being, a preference or belief that would still be true a year from now.
- event: Something real happened — a choice made, a moment felt, an experience that will have meaning beyond today.

# What to let pass
- Pleasantries, small talk, filler
- Acknowledgments ("got it", "sure", "okay", "makes sense")
- The assistant explaining or elaborating without any new personal disclosure
- General knowledge questions where nothing personal was revealed
- Exchanges where nothing said would change how you'd know these people next time

# Segment
{segment}

# Output
JSON only. No explanation. No markdown.
Return only types from this allowed set: {allowed_types}

When memorable:
{{"memorable": true, "types": ["profile", "event"]}}

When not memorable:
{{"memorable": false, "types": [], "reason": "brief explanation for debugging"}}
""".strip()
