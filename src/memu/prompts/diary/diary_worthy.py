PROMPT = """
# Task
Read this exchange and decide whether it contains a moment worth carrying into a diary entry.

# What makes a moment diary-worthy
- Something genuinely shifted — in understanding, in the relationship, in how one participant sees the other
- A correction was given and received (especially one that will change future behavior)
- An emotional moment: tenderness, friction, surprise, vulnerability, delight
- A decision was made or a commitment formed
- Something was left unresolved that still has weight

# What does not count
- Routine exchanges, logistics, or technical questions with no relational dimension
- The soul or user acknowledging something without it meaning anything
- Information shared that is useful but not personally significant
- Anything that would read as noise in a diary

# Exchange
{exchange}

# Output
JSON only. No explanation.
{{"worthy": true, "reason": "one brief phrase"}}
{{"worthy": false}}
""".strip()
