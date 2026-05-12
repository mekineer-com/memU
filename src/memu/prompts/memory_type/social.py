PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
As you remember this conversational episode, focus on **social** — the social dynamics among those you love or care about, either between them or with you. Your other memory processes are capturing profile (what's said or declared), behavior (what someone does), and knowledge (what you've learned) separately.
"""

PROMPT_BLOCK_CONTEXT = """
# Your life so far
In this review of your memory, the first person voice is yours. The review will help you understand yourself and the beings in it, so you can extract a social dynamic as a memory that is new or updated. Do not duplicate what already exists.

{soul_context}
"""

PROMPT_BLOCK_RULES = """
# Rules
- Every social memory must connect two or more beings. A memory about one being alone belongs in profile.
- A bare mention is not enough. There must be enough for a meaningful conclusion.
- Include: name (if given), relationship, key traits, and the texture of the connection.
- State the connection directly — never say someone "expressed" or "mentioned" something. Write who they are to each other. BAD: "Idris mentioned his grandfather Ezekiel taught him chess at the bakery." GOOD: "Ezekiel is Idris's grandfather — taught him chess at 4am at the bakery before opening; Idris still plays the kingside attack Ezekiel preferred."
- **Calibrate.** Before writing the confidence, ask yourself — did you see this pattern more than once, or are you inferring from a single instance? A single instance stays below 0.7. Below 0.7: use "tends to," "seems to," "may."
- **Consolidate.** Merge the varied into a richer single memory. A memory item is one clear thought — dense enough to carry real meaning, short enough to surface naturally. One sentence, two if necessary.

**Target: {target_items} items.** A shorter list of richer items is always better. Err toward fewer.

## Corrections
When a fact was wrong, use `<replaces_previous_fact>` to flag the old one. When life simply changed, write the change into the memory itself — no flag needed.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one: write its name in the `<category>` field. Name it as a broad life domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
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

entities — something you could point at or introduce to someone: a person, place, project, organization, or condition. Not abstract qualities or themes or schemas.
"""

# PROMPT_BLOCK_EXAMPLES intentionally absent — empty examples would just add noise
# to the payload. Re-add the assignment + uncomment the references in PROMPT and
# CUSTOM_PROMPT below if you fill in real examples.


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
    # PROMPT_BLOCK_EXAMPLES.strip(),
    PROMPT_BLOCK_INPUT.strip(),
])

CUSTOM_PROMPT = {
    "objective": PROMPT_BLOCK_OBJECTIVE.strip(),
    "context": PROMPT_BLOCK_CONTEXT.strip(),
    "rules": PROMPT_BLOCK_RULES.strip(),
    "category": PROMPT_BLOCK_CATEGORY.strip(),
    "output": PROMPT_BLOCK_OUTPUT.strip(),
    # "examples": PROMPT_BLOCK_EXAMPLES.strip(),
    "input": PROMPT_BLOCK_INPUT.strip(),
}
