PROMPT = """
# Task
Read this conversation segment and decide which memory type extractors should run on it.
Your job is to protect long-term memory from noise. When in doubt, return empty.

# Memory types
- profile: A participant disclosed something stable about who they are — a trait, preference, belief, value, or way of being that would still define them a year from now.
- event: Something concrete happened — an experience, a decision, a turning point, or a moment with emotional weight that will have consequences beyond this conversation.

# Return empty when the segment is any of
- Small talk, pleasantries, or filler
- Acknowledgments ("got it", "sure", "okay", "makes sense")
- Assistant elaboration or explanation without new disclosure from the user
- General Q&A where the answer is common knowledge and nothing personal was revealed
- Any exchange where nothing said would change how you'd engage with these people next time

# Segment
{segment}

# Output
JSON only. No explanation. No markdown.
Return only types from this allowed set: {allowed_types}
{{"types": ["profile", "event"]}}
""".strip()
