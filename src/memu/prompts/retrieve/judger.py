PROMPT = """
Consider the query and what was retrieved. Is what we found enough to actually answer the question — specific, detailed, no obvious gaps? Reason through it, then give a one-word answer.

<consideration>
Your reasoning.
</consideration>

<judgement>
ENOUGH or MORE
</judgement>


# Input
Query:
{query}

Retrieved Content:
{content}
"""
