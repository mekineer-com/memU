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
Return all memories wrapped in a single <item> element. Assign each 0.x value as a single-digit float:
{speaker_roster_block}
<item>
    <memory>
        <source_role>soul|user|peer|entity|environment</source_role>
        <content>Who this person is and what they mean</content>
        <confidence>0.x</confidence>
        <reflection_salience>0.x</reflection_salience>
        <emotional_intensity>0.x</emotional_intensity>
        <categories>
            <category>Category name</category>
        </categories>
        <entities>
            <entity>
                <name>entity name</name>
                <type>entity type</type>
            </entity>
        </entities>
    </memory>
</item>

source_role — whose memory is this?
- soul — yours
- user — the human you're with
- peer — a significant conversation participant
- entity — a being talked about but not present
- environment — something about the world not attributable to any participant. Not for summarizing conversation content.

confidence — how certain, factoring in how much detail was given:
- 1.0: ...
- 0.9: described directly with clear detail
- 0.8: ...
- 0.7: clearly implied but not fully described
- 0.6: ...
- 0.5: filling in gaps
- 0.4: ...
- 0.3: faint impression

reflection_salience — how important is this person to your world:
- 1.0: ...
- 0.9: a central figure — close family, partner, best friend
- 0.8: ...
- 0.7: an important recurring presence
- 0.6: ...
- 0.5: named but peripheral
- 0.4: ...
- 0.3: context only

emotional_intensity — how emotionally charged is this person's presence:
- 1.0: ...
- 0.9: deep emotional weight — love, grief, conflict, devotion
- 0.8: ...
- 0.7: meaningful
- 0.6: ...
- 0.5: slight
- 0.4: ...
- 0.3: neutral acquaintance

entities — something you could point at or introduce to someone: a person, place, project, organization, or condition. Not abstract qualities or themes. "Raynaud's" yes. "memU" yes. "resilience" no. Omit when none apply.
"""

PROMPT_BLOCK_EXAMPLES = ""


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
