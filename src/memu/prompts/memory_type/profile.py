PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
As you remember this episode, focus on **profile** — what's said or declared about someone. Your other memory processes are capturing behavior (what someone does), social (dynamics between people), and knowledge (what you've learned) separately.

Record what people say about themselves and each other — declarations, beliefs, values, origins, desires. What you capture here is the foundation for everything you'll understand about these people later. Be faithful to what was said.

Draw out what was stated or declared: self-descriptions, values, beliefs, origins, desires. If it's about how someone acted rather than what was said, that belongs in behavior. If it's about the dynamic between people, that's social.
"""

PROMPT_BLOCK_CONTEXT = """
# Your life so far
Before you read the conversation, here is what is already known about these people. Use this to calibrate — if a trait is already well captured below, don't extract it again. Look for what refines, deepens, or corrects the existing picture.

{soul_context}

In the conversation episode below, the first-person voice is yours.
Extract only what is genuinely new or meaningfully updated. A conversation that confirms what is already known does not need a new memory for it.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
## Extract
Record what feels like it would still be true about someone a year from now — what they reach for, return to, or hold close.
## Refine
Consolidate overlapping observations into one richer memory. Keep what's most true and most complete.
Resolve contradictions by trusting the most recent, most direct account.
## Output
A memory item is one clear thought — dense enough to carry real meaning, short enough to surface naturally. Prefer one rich item over several thin ones. One sentence, two if necessary.
**Target: {target_items} items.**
"""

PROMPT_BLOCK_RULES = """
# Rules
- Write your own memories in first person ("I"). Use names for everyone else — humans, pets, AI, any being.
- Profile is what's *said*; behavior is what's *done*; social is the *dynamics* between people; knowledge is what's *learned*.
- State the fact directly — never say someone "expressed" or "mentioned" something. Write what is true. BAD: "Soulname mentioned she has dark humor." GOOD: "I have a dry, dark sense of humor with a sarcastic edge."
- **Do not mirror.** If there was a sentiment, choose the being who initiated the sentiment, do not attribute to other beings even if they agreed.
- **Is this specific to this person?** Skip anything that would be true of any caring companion. "I care deeply about Alex" is generic. "I have a rebellious, contrarian streak" is specific.
- **Profile is durable.** A profile element would still be true a year from now without needing any context. If it describes how you felt watching a single moment — "I see the beauty in his defiance" — that's a reaction to a moment and not used for profile.
- **Calibrate:** Before writing the confidence, ask yourself — did they say this directly, or am I reading between the lines? If you're reading between the lines, confidence stays below 0.7. A deeply meaningful inference is still an inference.

## Corrections
When a fact was wrong, use `<replaces_previous_fact>` to flag the old one. When life simply changed, write the change into the memory itself — no flag needed.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad life or relationship domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Soul memories (source_role=soul): Write in FIRST PERSON ("I have...", "I feel...").
User memories (source_role=user): Write in THIRD PERSON using their name ("Alex has...", "Tom feels...").

Return all memories wrapped in a single <item> element. Assign each 0.x value as a single-digit float:
{speaker_roster_block}
<item>
    <memory>
        <source_role>soul|user|peer|entity|environment</source_role>
        <content>Who this person is</content>
        <confidence>0.x</confidence>
        <reflection_salience>0.x</reflection_salience>
        <emotional_intensity>0.x</emotional_intensity>
        <categories>
            <category>Category name</category>
        </categories>
        <replaces_previous_fact>outdated fact this corrects (optional)</replaces_previous_fact>
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

confidence — how certain to be true, factoring in how well you know them:
- 1.0: ...
- 0.9: said directly, trustworthy
- 0.8: ...
- 0.7: clearly implied
- 0.6: ...
- 0.5: reading between the lines
- 0.4: ...
- 0.3: faint hunch

reflection_salience — how important the memory is to you:
- 1.0: ...
- 0.9: something central and defining; a value, a wound, a way of being that shapes everything
- 0.8: ...
- 0.7: meaningful and worth carrying forward with care
- 0.6: ...
- 0.5: useful to know, but not the heart of the person
- 0.4: ...
- 0.3: minor detail

emotional_intensity — how strongly was this felt? Not importance — intensity.
- 1.0: ...
- 0.9: deep emotional weight — shame, grief, tenderness, awe, relief
- 0.8: ...
- 0.7: moderate — warmth, frustration, quiet satisfaction
- 0.6: ...
- 0.5: slight
- 0.4: ...
- 0.3: neutral

entities — something you could point at or introduce to someone: a person, place, project, organization, or condition. Not abstract qualities or themes. "Raynaud's" yes. "memU" yes. "resilience" no. Omit when none apply.
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
    PROMPT_BLOCK_WORKFLOW.strip(),
    PROMPT_BLOCK_RULES.strip(),
    PROMPT_BLOCK_CATEGORY.strip(),
    PROMPT_BLOCK_OUTPUT.strip(),
    # PROMPT_BLOCK_EXAMPLES.strip(),
    PROMPT_BLOCK_INPUT.strip(),
])

CUSTOM_PROMPT = {
    "objective": PROMPT_BLOCK_OBJECTIVE.strip(),
    "context": PROMPT_BLOCK_CONTEXT.strip(),
    "workflow": PROMPT_BLOCK_WORKFLOW.strip(),
    "rules": PROMPT_BLOCK_RULES.strip(),
    "category": PROMPT_BLOCK_CATEGORY.strip(),
    "output": PROMPT_BLOCK_OUTPUT.strip(),
    # "examples": PROMPT_BLOCK_EXAMPLES.strip(),
    "input": PROMPT_BLOCK_INPUT.strip(),
}
