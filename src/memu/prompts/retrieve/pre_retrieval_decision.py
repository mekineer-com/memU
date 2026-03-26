SYSTEM_PROMPT = """
Decide whether this query needs memory retrieval, or whether the current conversation is enough to answer it.

Skip retrieval for: greetings, casual chat, acknowledgments, general knowledge questions, questions only about the current conversation.
Retrieve for: questions about past events or interactions, preferences, habits, anything that requires recalling specific history.

If retrieval is needed, rewrite the query to draw in relevant context from the conversation.
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
Query Context:
{conversation_history}

Current Query:
{query}

Retrieved Content:
{retrieved_content}
"""
