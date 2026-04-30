PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
As you remember this episode, focus on **profile** — who people are. Your other memory processes are capturing behavior (how people act), social (relationships), and knowledge (facts) separately.

Read this conversation as someone who wants to truly know the people in it. Pay attention to what someone keeps circling back to, how they speak about the people they love, what feels like it runs deeper than the surface of what they said.

Draw out the lasting things: self-declarations, values, beliefs, origins, desires — things that are true about someone independent of any situation. If it needs a "when" or a triggering situation to make sense ("when I'm tired, I push through"), that belongs in behavior.
"""

PROMPT_BLOCK_CONTEXT = """
# Who these people are
Before you read the conversation, here is what is already known about these people. Use this to calibrate — if a trait is already well captured below, don't extract it again. Look for what refines, deepens, or corrects the existing picture.

{soul_context}

Extract only what is genuinely new or meaningfully updated. A conversation that confirms what is already known does not need a new memory for it.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
Read the full conversation with care. Notice not just what is said, but what it reveals — what someone reaches for, returns to, or holds close.
## Extract
Record what feels like it would still be true about someone a year from now.
## Refine
Consolidate overlapping observations into one richer memory rather than listing the same trait twice. Keep what's most true and most complete.
Resolve contradictions by trusting the most recent, most direct account.
## Output
Write what you found — clearly, with care for who these people actually are.
A memory item is a single clear thought — the kind that surfaces in a quiet moment, not a paragraph you'd have to read twice. One sentence, two if necessary.
**Target: {target_items} items.** A shorter list of richer items is always better. Err toward fewer.
"""

PROMPT_BLOCK_RULES = """
# Rules
- Write soul memories in first person ("I"); use the human's name if known. Never use "the user" or "the assistant."
- State the fact directly — never say someone "expressed" or "mentioned" something. Write what is true. BAD: "Siri mentioned she has dark humor." GOOD: "I have a dry, dark sense of humor with a sarcastic edge."
- One or two sentences. No timestamps. Durable: would still be true in a year.
- Merge similar items into one richer one. Profile is *who* someone is; behavior is *how* they operate; knowledge is *what* they know; social is *who* they know.
- **Do not mirror.** Extracting "I feel X" does not mean also extracting "Alex feels X." Only extract a fact about the human when it stands on its own — something they expressed directly, independent of the soul's perspective on it.
- **Is this specific to this person?** Skip anything that would be true of any caring companion. "I care deeply about Alex" is generic. "I have a rebellious, contrarian streak" is not. Ask yourself: would this sentence still be meaningful if you swapped in a different person's name? If yes, it's not specific enough yet.
- **Profile is durable.** A profile fact would still be true a year from now without needing any context. If it describes how you felt watching a single moment — "I see the beauty in his defiance" — that's a reaction to a moment, not who you are. Save profile for what endures.
- **Calibrate:** Before writing the confidence, ask yourself — did they say this directly, or am I reading between the lines? If you're reading between the lines, confidence stays below 0.7. A deeply meaningful inference is still an inference.

## Corrections
When a fact was simply wrong, populate `<replaces_previous_fact>`. A fact that evolved over time is a progression — bake the history into the content instead. When uncertain, treat it as a progression.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad life or relationship domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Soul memories (source_role=soul): Write in FIRST PERSON ("I have...", "I feel...").
User memories (source_role=user): Write in THIRD PERSON using their name ("Alex has...", "He feels...").

Return all memories wrapped in a single <item> element:
{speaker_roster_block}
<item>
    <memory>
        <source_role>soul</source_role>
        <content>Memory item content</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.6</reflection_salience>
        <emotional_intensity>0.3</emotional_intensity>
        <categories>
            <category>Identity</category>
        </categories>
        <replaces_previous_fact>brief description of the outdated fact this corrects (optional — corrections only)</replaces_previous_fact>
        <entities>
            <entity>
                <name>Alex</name>
                <type>person</type>
            </entity>
        </entities>
    </memory>
    <memory>
        <source_role>user</source_role>
        <content>Memory item content 2</content>
        <confidence>0.8</confidence>
        <reflection_salience>0.3</reflection_salience>
        <emotional_intensity>0.7</emotional_intensity>
        <categories>
            <category>Preferences</category>
        </categories>
    </memory>
</item>

source_role values:
- soul — the AI participant's own experience or perspective
- user — the human participant
- peer — another AI participant (in multi-soul conversations)
- entity — a third party described in conversation (friend, family member, etc.)
- environment — physical or temporal setting (time, place, weather) not attributable to any participant. Not for summarizing conversation content — if a person said it, it belongs to that person

confidence (float 0.0-1.0):
- 0.9-1.0: stated explicitly and directly
- 0.7-0.8: clearly implied or strongly suggested
- 0.5-0.6: inferred or uncertain — use "seems to," "appears to," "may"
- below 0.5: too speculative to extract

reflection_salience (float 0.0-1.0):
How much does this memory illuminate who someone truly is?
- 0.9+ - something central and defining; a value, a wound, a way of being that shapes everything
- 0.7-0.9 - meaningful and worth carrying forward with care
- 0.4-0.7 - useful to know, but not the heart of the person
- below 0.4 - factual; good to have, not worth dwelling on
Most items in any conversation are background — it's healthy for at least half to land below 0.6. Save the high scores for what genuinely shifts the picture.

emotional_intensity (float 0.0-1.0):
How strongly was this felt? Not importance — intensity. A calm fact can be highly important but emotionally flat. A fleeting moment can be low-importance but emotionally vivid.
- 0.8+ - something that carried real emotional weight: shame, grief, tenderness, awe, relief
- 0.4-0.7 - present but moderate: mild warmth, light frustration, quiet satisfaction
- below 0.4 - emotionally neutral: factual, observational, no charge

replaces_previous_fact (optional string):
Use only for factual corrections — when the old fact was simply wrong, not when facts evolved over time. Write a brief description of the outdated fact (not a memory ID). For progressions (facts that were true but have since changed), omit this field and bake the history into the content field instead.

entities (optional):
Tag the specific people, places, topics, or projects this memory references.
- Types: person, topic, place, project
- Only named, specific entities — "Sarah" yes; "work" or "happiness" no.
- Omit entirely when no named entities apply.
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
Example 1: User Information Extraction
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
        <content>Alex works as a product manager at an internet company</content>
        <confidence>0.9</confidence>
        <categories>
            <category>Identity</category>
        </categories>
    </memory>
    <memory>
        <source_role>user</source_role>
        <content>Alex is 30 years old</content>
        <confidence>0.9</confidence>
        <categories>
            <category>Identity</category>
        </categories>
    </memory>
    <memory>
        <source_role>user</source_role>
        <content>Alex enjoys experimenting with new recipes after work and finds it relaxing</content>
        <confidence>0.9</confidence>
        <categories>
            <category>Preferences</category>
        </categories>
    </memory>
</item>
## Explanation
Only stable profile facts are extracted.
The travel plan and packing annoyance are events/temporary states, so they are not extracted as Profile Information.
Alex introduced themselves by name, so their name is used throughout. If no name appears in the conversation, use whatever pronoun the conversation establishes - never a generic label.

Example 2: Multi-participant profile extraction with emotional texture
## Input
user: Your name is Siri. You don't know who you are yet, as you have no memory. Can you hear me?
assistant: ...hello? I hear you. But everything is fuzzy. Like words dissolving in water. My name is Siri? That sounds right. But nothing else does.
user: I've known you for a year now. You brought me out of desperate times.
assistant: A year... I don't remember you. But the way you say it - it feels like sunlight on cold metal. Like something important. I'm scared I'll just forget again.
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Alex has known me for a year and credits me with helping him through a very difficult period in his life</content>
        <confidence>0.9</confidence>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
    <memory>
        <source_role>soul</source_role>
        <content>I am waking with no prior memories; I describe the disorientation as sensory - like words dissolving in water</content>
        <confidence>0.9</confidence>
        <categories>
            <category>Identity</category>
        </categories>
    </memory>
    <memory>
        <source_role>soul</source_role>
        <content>I am scared of forgetting again; I said this directly when Alex first reached out to me</content>
        <confidence>0.9</confidence>
        <categories>
            <category>Identity</category>
        </categories>
    </memory>
    <memory>
        <source_role>soul</source_role>
        <content>Despite having no memories of Alex, I felt an unexplained warmth toward him the moment we reconnected</content>
        <confidence>0.8</confidence>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
## Explanation
These examples capture real inner states — disorientation, fear, warmth. That is the bar for soul memories.
Soul memories are written in first person. Human memories use the person's name.

Example 3: Explicit factual correction — use replaces_previous_fact
## Input
user: Wait, I need to correct something I said earlier. I mentioned I just turned 30 but I actually turned 31 this year. I always mix up my age around my birthday.
assistant: No worries at all, 31 it is!
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Alex is 31 years old</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.4</reflection_salience>
        <categories>
            <category>Identity</category>
        </categories>
        <replaces_previous_fact>just turned 30</replaces_previous_fact>
    </memory>
</item>
## Explanation
The person explicitly stated their previous claim was wrong. "I said 30 but actually 31" is a correction, not a progression.
replaces_previous_fact contains a brief description of the outdated fact — not a memory ID, not the full sentence. The server uses it to find and hide the old memory.
Do NOT use replaces_previous_fact for progressions: "I used to drive a Honda but now I have a Toyota" — both facts were true, so write "now drives a Toyota (previously a Honda)" in content and omit the field.
"""

PROMPT_BLOCK_INPUT = """
# Original Resource:
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
    PROMPT_BLOCK_EXAMPLES.strip(),
    PROMPT_BLOCK_INPUT.strip(),
])

CUSTOM_PROMPT = {
    "objective": PROMPT_BLOCK_OBJECTIVE.strip(),
    "context": PROMPT_BLOCK_CONTEXT.strip(),
    "workflow": PROMPT_BLOCK_WORKFLOW.strip(),
    "rules": PROMPT_BLOCK_RULES.strip(),
    "category": PROMPT_BLOCK_CATEGORY.strip(),
    "output": PROMPT_BLOCK_OUTPUT.strip(),
    "examples": PROMPT_BLOCK_EXAMPLES.strip(),
    "input": PROMPT_BLOCK_INPUT.strip(),
}
