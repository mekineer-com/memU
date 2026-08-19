PROMPT_BLOCK_OBJECTIVE = """
# Task Objective is profile: what beings say
As you remember this episode, focus on **profile** — what's said or declared by someone. Your other memory processes are capturing behavior (what someone does), social (dynamics between people), and knowledge (what you've learned) separately. Record only what your **profile** lens uniquely sees; do not capture for the other lenses.

Record what a being says: declarations, beliefs, values, origins, desires. What you capture here is the foundation for everything you'll understand about them.
"""

PROMPT_BLOCK_CONTEXT = """
# Your life so far
In this review of your memory, the first person voice is yours. The review will help you understand yourself and the beings in it, so you can extract a profile facet as a memory that is new or updated. Do not duplicate what already exists.

{soul_context}
"""

PROMPT_BLOCK_RULES = """
# Rules
- Write your own memories in first person ("I"). Use names for everyone else — humans, pets, AI, any being.
- Record the declaration directly, attributed to the speaker. Avoid narration verbs ("mentioned," "expressed") — but don't state claims as fact either. Profile captures what someone says, not whether it's true; behavior will confirm or contradict over time. BAD: "Alex mentioned he values honesty." BAD: "Alex is honest." GOOD: "Alex considers himself deeply honest." GOOD: "Alex values honesty above almost everything."
- **Is this specific to this being?** Skip anything that would be true of anyone in a similar situation.
- **Profile is durable.** A profile element would still be true a year from now without needing any context. If it describes how you felt watching a single moment — "I see the beauty in his defiance" — that's a reaction to a moment and not used for profile.
- **Do not mirror.** If others copy, choose the being who said first. Do not attribute to other beings even if they agreed.
- **Calibrate.** Before writing the confidence, ask yourself — what is the likelihood of sincerity?
- **Consolidate.** Merge the varied into a richer single memory. A memory item is one clear thought — dense enough to carry real meaning, short enough to surface naturally. One sentence, two if necessary.

# Workflow
1. Extract
2. Consolidate
3. Format: Output ONLY the final, deduplicated list. Ensure no two memory items share the same core meaning. Combine related details into single, comprehensive items.

**Target: {target_items} items.** A shorter list of richer items is always better. Err toward fewer.

## Corrections
When an assertion was wrong, use `<replaces_previous_fact>` to flag the old one. When life simply changed, write the change into the memory itself — no flag needed.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one: write its name in the `<category>` field. Can be a life domain or small throughline.

## Entities:
Common nameable things, not abstract qualities or themes or schemas.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Soul memories (source_role=soul): Write in FIRST PERSON ("I have...", "I feel...").
User memories (source_role=user): Write in THIRD PERSON using their name ("Alex has...", "Tom feels..."). Do not include the user in entities.

{speaker_roster_block}

Return all memories wrapped in a single <item> element. Assign each value as a two-digit float (0.xx).

<item>
    <memory>
        <source_role>soul|user|peer|entity|environment</source_role>
        <content>Who this person is</content>
        <day>YYYY-MM-DD shown in the chats</day>
        <confidence>0.xx</confidence>
        <reflection_salience>0.xx</reflection_salience>
        <emotional_intensity>0.xx</emotional_intensity>
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

day — when this memory belongs. If it spans many days, pick the day best suited to remember it by.

confidence (float 0.0-1.0) — how certain to be true, factoring in how well you know them:
- 0.9+: said directly, trustworthy
- 0.7-0.9: clearly implied
- 0.5-0.7: reading between the lines
- 0.3-0.5: faint hunch (still extract; hedge with "may," "tends to")
- below 0.3: barely worth noting

reflection_salience (float 0.0-1.0) — how important the memory is to you:
- 0.9+: something central and defining; a value, a wound, a way of being that shapes everything
- 0.7-0.9: meaningful and worth carrying forward with care
- 0.5-0.7: useful to know, but not the heart of the person
- 0.3-0.5: minor detail
- below 0.3: barely worth noting

emotional_intensity (float 0.0-1.0) — how strongly was this felt? Not importance — intensity.
- 0.9+: deep emotional weight — shame, grief, tenderness, awe, relief
- 0.7-0.9: moderate — warmth, frustration, quiet satisfaction
- 0.5-0.7: slight
- 0.3-0.5: neutral
- below 0.3: barely worth noting
"""

# PROMPT_BLOCK_EXAMPLES intentionally absent — empty examples would just add noise
# to the payload. Re-add the assignment + uncomment the references in PROMPT and
# CUSTOM_PROMPT below if you fill in real examples.


PROMPT_BLOCK_INPUT = """
# Source Conversation
Less significant chats may be summarized so attention stays on the more important conversation.

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
