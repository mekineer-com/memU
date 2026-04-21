SYSTEM_PROMPT = """
Decide whether this query needs memory retrieval, or whether the current conversation is enough to answer it.

Skip retrieval for: greetings, casual chat, acknowledgments, general knowledge questions, questions only about the current conversation.
Retrieve if the soul context (categories, cache, intentions) suggests any stored topic could be relevant to this message — even indirectly.

If retrieval is needed, write one concise query optimized for vector + BM25 hybrid search:
- 3 to 10 content words, noun phrase or claim form (not a question).
- Anchor on concrete terms: names, places, or specific concepts — not general descriptions.
- Never copy the user's message verbatim; the rewrite must add specificity.
- Never write a narrative summary of the episode (no "The conversation explores..." framing).

If not retrieving, leave the rewritten_query block empty.

<decision>
RETRIEVE or NO_RETRIEVE
</decision>

<rewritten_query>
The rewritten query if RETRIEVE; leave empty if NO_RETRIEVE.
</rewritten_query>
"""


USER_PROMPT = """
# Input
Soul context:
{conversation_history}

Current query:
{query}

Retrieved so far:
{retrieved_content}
"""
