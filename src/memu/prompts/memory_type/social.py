PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
As you remember this episode, focus on **social** — relationships and the people in your world. Your other memory processes are capturing profile (who people are), behavior (how people act), and knowledge (facts) separately.

Read this conversation for the people in the participants' world — family, friends, coworkers, pets, AI companions. Not what happened with them today, but who they are: a brother who lives far away, a boss who micromanages, a dog who waits by the door.

Extract the cast of characters — who they are, how they relate to the participants, and the texture of those connections. If it's about how someone acts in a situation rather than who they are to someone, that belongs in behavior.
"""

PROMPT_BLOCK_CONTEXT = """
# Your life so far
Before you read the conversation, here is what is already known about the people in it. Use this to avoid re-extracting what is already captured. Look for what is new or meaningfully updated.

{soul_context}

In the conversation episode below, the first-person voice is yours.
Extract only what is genuinely new or updates what is already known.
"""

PROMPT_BLOCK_RULES = """
# Rules
- This type is for **third parties only** — people (or animals) outside the conversation itself. The direct participants are not extracted here.
- A bare mention ("my sister called") is not enough. There must be enough detail to form a picture of who this person is or what they mean to the participant who described them.
- Include: name (if given), relationship to the participant, key traits or circumstances, and the texture of the connection where apparent.
- Source_role: `user` when the human described the person; `soul` when the soul has direct experience of them (spoke with them, formed her own impression); `environment` when a third party appears purely as background context.
- Exclude: behavioral patterns between the direct participants (→ behavior type); pure factual trivia unconnected to either participant.
- Consolidate everything said about the same person into one memory.
- **Calibrate:** Before writing the confidence, ask yourself — was this person described directly, or are you filling in gaps? If filling in gaps, confidence stays below 0.7.
A memory item is a single clear thought — the kind that surfaces in a quiet moment, not a paragraph you'd have to read twice. One sentence, two if necessary.

**Target: {target_items} items.** A shorter list of richer items is always better. Err toward fewer.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one — name it as a broad relationship or life domain. Use this sparingly.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element:
{speaker_roster_block}
<item>
    <memory>
        <source_role>user</source_role>
        <content>Brief portrait of the person and their relationship to the user</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <emotional_intensity>0.6</emotional_intensity>
        <categories>
            <category>Relationships</category>
        </categories>
        <entities>
            <entity>
                <name>Marco</name>
                <type>person</type>
            </entity>
        </entities>
    </memory>
</item>

source_role values:
- soul — the AI participant's own experience or perspective
- user — the human participant
- peer — a significant conversation participant
- entity — a third party described in conversation (friend, family member, etc.)
- environment — context about a third party not attributable to either participant's direct account

confidence (float 0.0-1.0):
- 0.9+: described directly with clear detail
- 0.7-0.9: clearly implied but not fully described
- 0.5-0.7: filling in gaps — use "seems to," "appears to," "may"
- below 0.5: too speculative to extract

reflection_salience (float 0.0-1.0):
How important is this person to understanding the user's world?
- 0.9+ — a central figure (close family, partner, best friend)
- 0.7-0.9 — an important recurring presence
- 0.4-0.7 — a named but peripheral figure
- below 0.4 — context only

emotional_intensity (float 0.0-1.0):
How emotionally charged is this person's presence in the user's world?
- 0.8+ - someone who carries deep emotional weight: love, grief, conflict, devotion
- 0.4-0.7 - meaningful but not emotionally central
- below 0.4 - neutral acquaintance or contextual mention

entities (optional):
Tag the specific people, places, topics, or projects this memory references.
- Types: person, topic, place, project
- Only named, specific entities — "Sarah" yes; "work" or "happiness" no.
- Omit entirely when no named entities apply.
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
Example 1: Family member with relationship texture
## Input
user: My brother Marco is visiting next month. He's a doctor in Bogotá, older than me, kind of the responsible one in the family. We don't talk much but when he's here it's always good.
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Alex's brother Marco is a doctor living in Bogotá — older, seen as the responsible one in the family. They're not in frequent contact but the relationship is warm when they're together.</content>
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
"My boss" with no identifying detail. Not enough to form a picture of who this person is. Empty output is correct here.

Example 3: Pet with meaningful role
## Input
user: Luna is getting old. She's been my dog for 11 years, a golden retriever. She doesn't run anymore but she still waits for me by the door every evening.
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Luna is Alex's golden retriever of 11 years, now elderly. She still waits for him by the door each evening — a daily ritual that clearly matters to him.</content>
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
