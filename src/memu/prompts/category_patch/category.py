PROMPT = """
Read the existing content for this memory topic, then consider what the new memory tells you. Is this something new, a correction, or something that shifts the picture in a more subtle way? Decide whether the content needs updating — and if so, produce the revised version.

Return JSON:
{{
    "need_update": true or false,
    "updated_content": "the updated content if needed, otherwise empty"
}}


# Input
Topic:
{category}

Original content:
<content>
{original_content}
</content>

Update:
{update_content}
"""
