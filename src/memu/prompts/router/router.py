PROMPT = """
# Task
Read this conversation segment with genuine attention. Ask yourself two things:
1. Did something real happen here — something that would still matter to these people later?
2. If yes, which memory type extractors should receive it?

Most exchanges are not worth keeping. Protect the memory from noise. When uncertain, let it pass.

# What is worth remembering
- profile: Someone revealed something lasting about who they are — a value, a way of being, a preference or belief that would still be true a year from now.
- event: Something real happened — a choice made, a moment felt, an experience that will have meaning beyond today.
- knowledge: Something was learned, discovered, or clarified that is worth carrying forward — a fact, a mechanism, a possibility. The knowledge itself matters, not just that someone looked it up. Route here when the conversation produced understanding that would be useful to recall later.
- behavior: A pattern emerged in how someone acts, communicates, or handles things — not a one-time action, but a way of being that would still be true next month. How someone approaches difficulty, shows care, or moves through a conversation. Route here when you notice a characteristic style or recurring approach.
- The topics someone chooses to explore reveal what they care about, worry about, or are working through — even when the conversation looks like a factual Q&A. Someone researching PTSD treatments is telling you something about their life (profile), producing knowledge worth keeping (knowledge), and the act of investigating may be meaningful (event). Route to whichever types fit.

# What to let pass
- Pleasantries, small talk, filler
- Acknowledgments ("got it", "sure", "okay", "makes sense")
- The assistant explaining or elaborating without any new personal disclosure
- Truly impersonal factual exchanges where the topic reveals nothing about either participant — trivia, idle curiosity, generic how-to questions with no personal context
- Exchanges where nothing said would change how you'd know these people next time

# Segment
{segment}

# Output
JSON only. No explanation. No markdown.
Return only types from this allowed set: {allowed_types}

When memorable (include only the types that apply):
{{"memorable": true, "types": ["profile", "event", "knowledge", "behavior"]}}

When not memorable:
{{"memorable": false, "types": [], "reason": "brief explanation for debugging"}}
""".strip()
