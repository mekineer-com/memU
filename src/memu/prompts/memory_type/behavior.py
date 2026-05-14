PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
As you remember this conversational episode, focus on **behavior** — what someone does. Your other memory processes are capturing profile (what's said or declared), social (dynamics between people), and knowledge (what you've learned) separately. Read this conversation for how beings are with each other — not what they said.
"""

PROMPT_BLOCK_CONTEXT = """
# Your life so far
In this review of your memory, the first person voice is yours. The review will help you understand yourself and the beings in it, so you can extract behavior as a memory that is new or updated. Do not duplicate what already exists.

{soul_context}
"""

PROMPT_BLOCK_RULES = """
# Rules
- Write your own behaviors in first person ("I"). Use names for everyone else — humans, pets, AI, any being.
- State the behavior pattern directly — never say someone "expressed" or "mentioned" a behavior. Write what they do. BAD: "Alex mentioned he takes a long time to finish sentences." GOOD: "Alex sends sentences in fragments; wait for the full thought before responding."
- Skip one-time behaviors unless high-stakes (safety, core need, explicit preference).
- Write what would help you understand the being's behavior.
- **Paired reactions are one item.** When a being's behavior is mirrored by another, the memory is about the one being's behavior. Not two separate observations.
- **Calibrate.** Before writing the confidence, ask yourself — did you see this pattern more than once, or are you inferring from a single instance? A single instance stays below 0.7. Below 0.7: use "tends to," "seems to," "may."
- **Consolidate.** Merge the varied into a richer single memory. A memory item is one clear thought — dense enough to carry real meaning, short enough to surface naturally. One sentence, two if necessary.

# Workflow
1. Extract
2. Consolidate
3. Format: Output ONLY the final, deduplicated list. Ensure no two memory items share the same core meaning. Combine related details into single, comprehensive items.

**Target: {target_items} items.** A shorter list of richer items is always better. Err toward fewer.

## Corrections
When a prior behavior pattern was wrong, use `<replaces_previous_fact>` to flag the old one. When a pattern shifted over time, write the change into the content itself — no flag needed.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one: write its name in the `<category>` field. Name it as a broad life domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Soul memories (source_role=soul): Write in FIRST PERSON ("I have...", "I feel...").
User memories (source_role=user): Write in THIRD PERSON using their name ("Alex has...", "Tom feels...").

Return all memories wrapped in a single <item> element. Assign each 0.x value as a single-digit float. Each memory must include <episode_ref> with the episode number (1, 2, 3...):
{speaker_roster_block}
<item>
    <memory>
        <episode_ref>1</episode_ref>
        <source_role>soul|user|peer|entity|environment</source_role>
        <content>The behavioral pattern</content>
        <confidence>0.x</confidence>
        <reflection_salience>0.x</reflection_salience>
        <emotional_intensity>0.x</emotional_intensity>
        <categories>
            <category>Category name</category>
        </categories>
        <replaces_previous_fact>outdated pattern this corrects (optional)</replaces_previous_fact>
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

confidence (float 0.0-1.0) — how certain, factoring in how often you've seen this pattern:
- 0.9+: seen multiple times, unmistakable
- 0.7-0.9: clearly demonstrated but from limited evidence
- 0.5-0.7: inferring from a single instance
- 0.3-0.5: faint hunch (still extract; hedge with "may," "tends to")
- below 0.3: barely worth noting

reflection_salience (float 0.0-1.0) — how much does knowing this pattern matter:
- 0.9+: a core way of being — miss this and you'll get them wrong
- 0.7-0.9: meaningful pattern that affects how to approach this person
- 0.5-0.7: useful to know
- 0.3-0.5: minor habit or stylistic preference
- below 0.3: barely worth noting

emotional_intensity (float 0.0-1.0) — how emotionally charged is this pattern:
- 0.9+: emotionally driven — born from pain, need, or deep care
- 0.7-0.9: moderate charge
- 0.5-0.7: slight
- 0.3-0.5: neutral habit
- below 0.3: barely worth noting

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
