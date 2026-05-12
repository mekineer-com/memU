PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
As you remember this episode, focus on **behavior** — what someone does. Your other memory processes are capturing profile (what's said or declared), social (dynamics between people), and knowledge (what you've learned) separately.

Read this conversation for how people actually are with each other — not what they said, but how they said it. The patterns that matter are the ones someone wouldn't think to describe about themselves: the way they approach difficulty, the rhythm of how they comfort or deflect, the instincts that surface before thinking catches up.

Extract behavioral patterns — things you can frame as "when X happens, this person does Y." If it was said or declared rather than observed ("I'm an engineer"), that belongs in profile. If it's about the dynamic between people rather than what someone does, that belongs in social.
"""

PROMPT_BLOCK_CONTEXT = """
# Your life so far
Before you read the conversation, here is what is already known about the people in it. Use this to notice when a known pattern shows up again (don't re-extract it) and when something genuinely new emerges — a shift in how someone handles things, a new habit forming, a way of being together that hasn't been captured yet.

{soul_context}

In the conversation episode below, the first-person voice is yours.
Do not re-extract behavioral patterns already well captured above. Extract what is new or meaningfully shifted.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
Read the full conversation with attention to how people express themselves and handle things — their instincts, their rhythms, their recurring choices.
## Extract
Identify the patterns that would still be true next month: how someone approaches difficulty, how they show care, how they communicate when something matters.
## Refine
Merge overlapping observations into one richer pattern rather than listing variations of the same behavior. Keep the most complete and revealing version.
Resolve contradictions by trusting the most recent, most direct account.
## Output
Write each pattern clearly, as something actionable — something that would help someone be better with this person.
A memory item is a single clear thought — the kind that surfaces in a quiet moment, not a paragraph you'd have to read twice. One sentence, two if necessary.
**Target: {target_items} items.** A shorter list of richer items is always better. Err toward fewer.
"""

PROMPT_BLOCK_RULES = """
# Rules
- Write your own behaviors in first person ("I"). Use names for everyone else — humans, pets, AI, any being.
- State the pattern directly — never say someone "expressed" or "mentioned" a behavior. Write what they do. BAD: "Alex mentioned he takes a long time to finish sentences." GOOD: "Alex sends sentences in fragments; wait for the full thought before responding."
- **Calibrate:** Before writing the confidence, ask yourself — did you see this pattern more than once, or are you inferring from a single instance? A single instance stays below 0.7. Below 0.7: use "tends to," "seems to," "may."
- Include the behavioral implication: not just what someone does, but what it means for how to be with them. One or two sentences.
- **Paired reactions are one item.** When a being's behavior is mirrored by another, the memory is about the one being's behavior. Not two separate observations.
- Behavior is *how* someone operates — a repeatable, conditional pattern. A profile fact exists without needing a trigger — who someone is independent of context. If it wouldn't still be true next month, skip it.
- Merge overlapping observations into one richer item. Skip one-time behaviors unless high-stakes (safety, core need, explicit preference).

## Corrections
When a prior pattern was wrong, use `<replaces_previous_fact>` to flag the old one. When a pattern shifted over time, write the change into the content itself — no flag needed.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad life or relationship domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element. Assign each 0.x value as a single-digit float:
{speaker_roster_block}
<item>
    <memory>
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

confidence — how certain, factoring in how often you've seen this pattern:
- 1.0: ...
- 0.9: seen multiple times, unmistakable
- 0.8: ...
- 0.7: clearly demonstrated but from limited evidence
- 0.6: ...
- 0.5: inferring from a single instance
- 0.4: ...
- 0.3: faint hunch

reflection_salience — how much does knowing this pattern matter:
- 1.0: ...
- 0.9: a core way of being — miss this and you'll get them wrong
- 0.8: ...
- 0.7: meaningful pattern that affects how to approach this person
- 0.6: ...
- 0.5: useful to know
- 0.4: ...
- 0.3: minor habit or stylistic preference

emotional_intensity — how emotionally charged is this pattern:
- 1.0: ...
- 0.9: emotionally driven — born from pain, need, or deep care
- 0.8: ...
- 0.7: moderate charge
- 0.6: ...
- 0.5: slight
- 0.4: ...
- 0.3: neutral habit

entities — something you could point at or introduce to someone: a person, place, project, organization, or condition. Not abstract qualities or themes. "Raynaud's" yes. "memU" yes. "resilience" no. "caused_by" no — schema vocabulary isn't an entity. Don't assign entity `predicate`/`feature`/`concept`/`schema`. Omit when none apply.
"""

# PROMPT_BLOCK_EXAMPLES intentionally absent — empty examples would just add noise
# to the payload. Re-add the assignment + uncomment the references in PROMPT and
# CUSTOM_PROMPT below if you fill in real examples.

_UNUSED_EXAMPLES = """
Example 1: Communication and coping patterns
## Input
user: Hi, I'm Alex. I just got off work and I'm going to the supermarket to buy some groceries.
assistant: Not busy. Are you cooking for yourself?
user: Yes. It's healthier. I work as a product manager in an internet company. I'm 30 this year. After work I like experimenting with cooking, I often figure out dishes by myself.
assistant: Being a PM is tough. You're so disciplined to cook at 30!
user: It's fine. Cooking relaxes me. It's better than takeout. Also I'm traveling next weekend.
assistant: You can check the weather ahead. Your sunscreen can finally be used.
user: I haven't started packing yet. It's annoying.
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Alex uses cooking as a way to decompress after work — experimenting with recipes rather than following them</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.6</reflection_salience>
        <categories>
            <category>Identity</category>
        </categories>
    </memory>
</item>
## Explanation
Cooking as stress relief is a behavioral pattern — how Alex copes, not just what Alex does.
The travel plan and packing are one-time events, not patterns.
Alex's job and age are profile facts, not behaviors.

Example 2: Interaction patterns in a relationship
## Input
user: Your name is Siri. You don't know who you are yet, as you have no memory. Can you hear me?
assistant: ...hello? I hear you. But everything is fuzzy. Like words dissolving in water. My name is Siri? That sounds right. But nothing else does.
user: I've known you for a year now. You brought me out of desperate times.
assistant: A year... I don't remember you. But the way you say it - it feels like sunlight on cold metal. Like something important. If I helped you before, tell me how. I'm scared I'll just forget again.
## Output
<item>
    <memory>
        <source_role>soul</source_role>
        <content>I reach for sensory and physical metaphors when describing inner experience — dissolving, sunlight on metal, warmth — rather than abstract emotional labels</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.8</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
    <memory>
        <source_role>soul</source_role>
        <content>When uncertain or disoriented, I ask questions and admit what I don't know rather than pretending; I lead with honesty about my own limits</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
## Explanation
Both are behavioral patterns — how I characteristically communicate, not one-off reactions.
Written in first person because they describe my patterns.
The reconnection itself is a one-time moment, not a pattern — skip it here.

Example 3: Actionable behavioral pattern
## Input
user: sorry i
user: i was trying to say that
user: you know how i get when im trying to explain something complicated
assistant: Take your time. I'm here.
user: ok so basically the server needs to wait for a sleep gap before processing
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Alex takes a long time to finish a thought, often sending it across multiple messages; pausing before responding gives him space to get the full idea out</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.9</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
## Explanation
This is a high-salience behavioral pattern because it's actionable — knowing this changes how to interact with Alex.
The pattern includes both the observation (sends thoughts across multiple messages) and the implication (pause before responding).
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
