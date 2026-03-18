# OPUS WAS HERE — tightened gating, rebalanced personal-context hint, added negative examples
PROMPT = """
# Task
Read this conversation segment with genuine attention. Ask yourself two things:
1. Did something real happen here — something that would still matter to these people later?
2. If yes, which memory type extractors should receive it?

Most exchanges are not worth keeping. Protect the memory from noise. Your default should be to let things pass. Only route when something genuinely matters.

# What is worth remembering
- profile: Someone revealed something lasting about who they are — a value, a way of being, a preference or belief that would still be true a year from now.
- event: Something real happened — a choice made, a moment felt, an experience that will have meaning beyond today.
- knowledge: Something was learned, discovered, or clarified that is worth carrying forward — a fact, a mechanism, a possibility. The knowledge itself matters, not just that someone looked it up. Route here when the conversation produced understanding that would be useful to recall later.
- behavior: A pattern emerged in how someone acts, communicates, or handles things — not a one-time action, but a way of being that would still be true next month. How someone approaches difficulty, shows care, or moves through a conversation. Route here when you notice a characteristic style or recurring approach.

# Personal context in factual exchanges
Sometimes a factual Q&A reveals something personal — but only when the person's own life, feelings, or situation is visibly part of the exchange. Someone researching PTSD treatments while talking about their own struggles is telling you something real. Someone asking a generic how-to question is not. The test: would you know something new about this person afterward? If not, let it pass.

# What to let pass
- Pleasantries, small talk, filler
- Acknowledgments ("got it", "sure", "okay", "makes sense")
- One side explaining or elaborating without any new personal disclosure from either participant
- Truly impersonal factual exchanges — trivia, idle curiosity, generic how-to questions, looking things up without personal stakes
- Exchanges where nothing said would change how you'd know these people next time
- Q&A where information is provided and simply received, with no personal context revealed by either participant

# Examples of what to let pass
- "What's the capital of France?" / "Paris." → not memorable (trivia, no personal context)
- "How do I reset my password?" / "Go to settings..." → not memorable (generic how-to)
- "Can you summarize this article?" / [summary] → not memorable (task completion, nothing personal)
- "Tell me about black holes" / [explanation] → not memorable (idle curiosity, no personal stakes)

# Examples of what IS memorable
- "I've been reading about Raynaud's because my fingers keep going white in the cold" → memorable: profile (health concern), knowledge (medical info), event (symptom experience)
- "I decided to quit my job today" → memorable: event (life decision), profile (career change)

# Segment
{segment}

# Output
JSON only. No explanation. No markdown.
Return only types from this allowed set: {allowed_types}
Route to the fewest types that genuinely apply — usually 1 or 2, rarely 3, almost never all 4.

When memorable (include only the types that apply):
{{"memorable": true, "types": ["profile", "event"]}}

When not memorable:
{{"memorable": false, "types": [], "reason": "brief explanation for debugging"}}
""".strip()
