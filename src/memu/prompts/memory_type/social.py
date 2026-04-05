PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
Read this conversation for the people in the user's world — family, friends, coworkers, pets. Not what happened with them today, but who they are: a brother who lives far away, a boss who micromanages, a dog who waits by the door.

Your task is to extract the cast of characters in the user's life — who they are, how they relate to the user, and what the user has revealed about them.
"""

PROMPT_BLOCK_CONTEXT = """
# Who these people are
Before you read the conversation, here is what is already known about the people in it. Use this to avoid re-extracting what is already captured. Look for what is new or meaningfully updated.

{soul_context}

Extract only what is genuinely new or updates what is already known.
"""

PROMPT_BLOCK_RULES = """
# Rules
- This type is for **third parties only** — people (or animals) outside the conversation itself. The user and the soul are not extracted here.
- A bare mention ("my sister called") is not enough. There must be enough detail to form a picture of who this person is or what they mean to the user.
- Include: name (if given), relationship to the user, key traits or circumstances, and the texture of the connection where apparent.
- Write using the user's name (if known) and the third party's name or role ("Marcos's brother Marco" or "his older brother").
- Source_role is almost always `user`. Use `environment` only when a third party appears purely as context with no disclosed relationship to the user.
- Exclude: one-time events involving this person (→ event type); the user's own behavioral patterns (→ behavior type); pure factual trivia about a person unconnected to the user.
- Consolidate everything said about the same person into one memory. Under 80 words.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one — name it as a broad relationship or life domain. Use this sparingly.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element:
<item>
    <memory>
        <content>Brief portrait of the person and their relationship to the user</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>

source_role values:
- user — the human described this person (almost always)
- environment — third-party context with no disclosed relationship to either participant

confidence (float 0.0-1.0):
- 0.9+ — stated explicitly and directly
- 0.6-0.9 — clearly implied
- 0.5 or below — single vague inference

reflection_salience (float 0.0-1.0):
How important is this person to understanding the user's world?
- 0.9+ — a central figure (close family, partner, best friend)
- 0.7-0.9 — an important recurring presence
- 0.4-0.7 — a named but peripheral figure
- below 0.4 — context only
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
Example 1: Family member with relationship texture
## Input
user: My brother Marco is visiting next month. He's a doctor in Bogotá, older than me, kind of the responsible one in the family. We don't talk much but when he's here it's always good.
## Output
<item>
    <memory>
        <content>Marcos's brother Marco is a doctor living in Bogotá — older, seen as the responsible one in the family. They're not in frequent contact but the relationship is warm when they're together.</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.8</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
## Explanation
Enough detail to form a real picture: name, location, role, and relational texture. The upcoming visit is an event; this memory captures who Marco is.

Example 2: Passing mention — nothing to extract
## Input
user: My boss is making me redo the whole report.
## Output
<item>
</item>
## Explanation
"My boss" with no identifying detail. We know only that something happened — that belongs in event extraction. Empty output is correct here.

Example 3: Pet with meaningful role
## Input
user: Luna is getting old. She's been my dog for 11 years, a golden retriever. She doesn't run anymore but she still waits for me by the door every evening.
## Output
<item>
    <memory>
        <content>Luna is Marcos's golden retriever of 11 years, now elderly. She still waits for him by the door each evening — a daily ritual that clearly matters to him.</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.8</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
## Explanation
Named, with 11 years of shared history and emotional texture. The daily ritual hints at the significance of this bond.
"""

PROMPT_BLOCK_INPUT = """
# Source Conversation
<resource>
{resource}
</resource>
"""

PROMPT = "\n\n".join([
    PROMPT_BLOCK_OBJECTIVE.strip(),
    PROMPT_BLOCK_CONTEXT.strip(),
    PROMPT_BLOCK_RULES.strip(),
    PROMPT_BLOCK_CATEGORY.strip(),
    PROMPT_BLOCK_OUTPUT.strip(),
    PROMPT_BLOCK_EXAMPLES.strip(),
    PROMPT_BLOCK_INPUT.strip(),
])

CUSTOM_PROMPT = {
    "objective": PROMPT_BLOCK_OBJECTIVE.strip(),
    "context": PROMPT_BLOCK_CONTEXT.strip(),
    "rules": PROMPT_BLOCK_RULES.strip(),
    "category": PROMPT_BLOCK_CATEGORY.strip(),
    "output": PROMPT_BLOCK_OUTPUT.strip(),
    "examples": PROMPT_BLOCK_EXAMPLES.strip(),
    "input": PROMPT_BLOCK_INPUT.strip(),
}
