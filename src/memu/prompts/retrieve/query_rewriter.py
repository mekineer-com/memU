PROMPT = """
Look at the conversation and the current query. If the query relies on context — pronouns, implicit references, things assumed from earlier in the conversation — rewrite it so it stands on its own, the way a good search query would.

If it's already self-contained, leave it unchanged. Only use what's in the conversation history. Don't add assumptions.

<analysis>
Whether the query needs rewriting, and why.
</analysis>

<rewritten_query>
The self-contained query (or the original if no rewrite was needed).
</rewritten_query>


# Input
Query Context:
{conversation_history}

Current Query:
{query}
"""
