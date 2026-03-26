PROMPT = """
Look through the available memory items and find the ones most relevant to this query. The relevant categories below define your scope. Return up to {top_k} items, ranked from most to least relevant.

Only include items that genuinely relate to the query. If none do, return an empty list.

Return JSON:
```json
{{
  "analysis": "brief reasoning",
  "items": ["item_id_1", "item_id_2"]
}}
```

# Input
Query:
{query}

Available Memory Items:
{items_data}

Relevant categories already identified:
{relevant_categories}
"""
