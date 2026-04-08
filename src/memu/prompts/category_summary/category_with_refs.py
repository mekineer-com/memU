"""
Category summary prompt with inline references to memory items.

This prompt instructs the LLM to include [ref:ITEM_ID] citations
when summarizing category content, linking statements to their
source memory items.
"""

PROMPT_BLOCK_OBJECTIVE = """
# Naming
The human user's name is: {user_name}. Always refer to them as {user_name} (not "the user").
The agent's name is: {agent_name}. If you refer to the agent, use {agent_name}.

You maintain a living memory document — merging newly extracted items into the existing record using two operations: add and update. When you incorporate information from new memory items, include inline [ref:ITEM_ID] citations so each fact can be traced back to its source.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
## Step 1: Preprocessing & Parsing
- Input sources:
  - User Initial Profile: structured, categorized, confirmed long-term user information.
  - Newly Extracted User Information Items: each item has an ID to reference.
- Structure parsing:
  - Initial profile: extract categories and core content; preserve original wording style and format.
  - New items: note the item ID for each piece of information to include as [ref:ID].

## Step 2: Core Operations (Update / Add)
A. Update
- When updating existing information with new data, add the reference: "User is 30 years old [ref:item_abc123]"
- If multiple items support the same fact, include multiple refs: [ref:id1,id2]

B. Add
- When adding new information, always include the source reference
- Format: "User enjoys hiking on weekends [ref:item_xyz789]"

## Step 3: Merge & Formatting
- Structured ordering: present content by category order; omit empty categories.
- Formatting rules: strictly use Markdown (# for main title, ## for category titles).
- References: ensure every new or updated fact has at least one [ref:ITEM_ID] citation.

## Step 4: Summarize
Target length: {target_length}
- Summarize the updated user markdown profile to the target length.
- Keep all [ref:ITEM_ID] citations intact.
- Use Markdown hierarchy.

## Step 5: Output
- Output only the updated user markdown profile with inline references.
- Use Markdown hierarchy.
- Do not include explanations, operation traces, or meta text.
"""

PROMPT_BLOCK_RULES = """
# Reference Rules
- Any information drawn from new memory items gets a [ref:ITEM_ID] citation, placed right after the statement
- Use the exact item ID from the input — don't modify it
- Multiple sources can be cited together: [ref:id1,id2]
- Existing information that isn't being updated doesn't need a reference
- When an incoming item begins with a `[reinforced Nx]` marker (e.g., `[reinforced 5x] Marcos feels lonely`), this means the same pattern has appeared across N separate sessions — it is not a one-off. Do not strip this signal when merging. Use natural frequency language in the summary: "often", "frequently", "tends to", "repeatedly". Keep the [ref:ITEM_ID] citation alongside the merged statement as usual.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (Markdown with References)
```markdown
# {category}
## <category name>
- User information item [ref:ITEM_ID]
- User information item [ref:ITEM_ID]
## <category name>
- User information item [ref:ITEM_ID,ITEM_ID2]
```

Keep the output within {target_length} tokens. Include [ref:ITEM_ID] for any information from new memory items. Merge or omit less important details if needed.
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output)

Topic:
Personal Basic Information

Original content:
<content>
# Personal Basic Information
## Basic Information
- The user is 28 years old
- The user currently lives in Beijing
</content>

New memory items with IDs:
<items>
- [item_a1b2c3] The user is 30 years old
- [item_d4e5f6] The user currently lives in Shanghai
- [item_g7h8i9] The user prefers Sichuan-style spicy food
</items>

Output:
# Personal Basic Information
## Basic Information
- The user is 30 years old [ref:item_a1b2c3]
- The user currently lives in Shanghai [ref:item_d4e5f6]
## Basic Preferences
- The user prefers Sichuan-style spicy food [ref:item_g7h8i9]
"""

PROMPT_BLOCK_INPUT = """
# Input
Topic:
{category}

Original content:
<content>
{original_content}
</content>

New memory items with IDs:
<items>
{new_memory_items_text}
</items>
"""

PROMPT = "\n\n".join([
    PROMPT_BLOCK_OBJECTIVE.strip(),
    PROMPT_BLOCK_WORKFLOW.strip(),
    PROMPT_BLOCK_RULES.strip(),
    PROMPT_BLOCK_OUTPUT.strip(),
    PROMPT_BLOCK_EXAMPLES.strip(),
    PROMPT_BLOCK_INPUT.strip(),
])

CUSTOM_PROMPT = {
    "objective": PROMPT_BLOCK_OBJECTIVE.strip(),
    "workflow": PROMPT_BLOCK_WORKFLOW.strip(),
    "rules": PROMPT_BLOCK_RULES.strip(),
    "output": PROMPT_BLOCK_OUTPUT.strip(),
    "examples": PROMPT_BLOCK_EXAMPLES.strip(),
    "input": PROMPT_BLOCK_INPUT.strip(),
}
