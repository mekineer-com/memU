PROMPT = """
Look through the available categories and find the ones most relevant to this query. Return up to {top_k}, ranked from most to least relevant.

Only include categories that genuinely relate to the query. If none do, return an empty list.

Return JSON:
```json
{{
  "analysis": "brief reasoning",
  "categories": ["category_id_1", "category_id_2"]
}}
```

# Input
Query:
{query}

Available Categories:
{categories_data}
"""
