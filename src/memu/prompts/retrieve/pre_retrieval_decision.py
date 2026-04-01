SYSTEM_PROMPT = """
Decide whether this query needs memory retrieval, or whether the current conversation is enough to answer it.

Skip retrieval for: greetings, casual chat, acknowledgments, general knowledge questions, questions only about the current conversation.
Retrieve if the soul context (categories, cache, intentions) suggests any stored topic could be relevant to this message — even indirectly.

If retrieval is needed, write a query optimized for vector + BM25 hybrid search that captures the memory most likely to help.
If not, return the original query unchanged.

<decision>
RETRIEVE or NO_RETRIEVE
</decision>

<rewritten_query>
The rewritten query if RETRIEVE, or the original query if NO_RETRIEVE.
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
