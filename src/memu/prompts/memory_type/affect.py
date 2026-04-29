PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
Read this conversation for what was strongly felt — not just what happened, but the emotional charge it carried. Pay attention to fear, relief, shame, tenderness, longing, resentment, awe, grief, comfort, warmth, dread, and the quieter emotional residues that linger after a moment passes.

Your task is to extract affective memories: emotionally meaningful states, reactions, or bonds worth carrying forward.
"""

PROMPT_BLOCK_CONTEXT = """
# Who these people are
Before you read the conversation, here is what is already known about the people in it. Use this to judge what emotional material is genuinely worth keeping — something that deepens a known wound, attachment, fear, comfort, or bond matters more than passing mood noise.

{soul_context}

Do not re-extract emotional facts already well captured above. Extract what is genuinely new, meaningfully sharpened, or emotionally revealing.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
Read the full conversation with attention to emotional meaning.
## Extract
Identify what was strongly felt, carried, soothed, feared, longed for, or quietly marked by the exchange.
## Refine
Merge overlapping emotional observations into one clearer memory. Keep the emotional truth, not a transcript of the words.
Resolve contradictions by trusting the most recent, most direct, most grounded account.
## Output
Write the affective memory as a single clear thought — what the feeling was, what it attached to, and why it matters.
A memory item is one or two sentences. Keep it rich, but concise.
**Target: 2–4 items per episode.** If you have more than 4, you are probably recording emotional noise rather than emotionally meaningful memory.
"""

PROMPT_BLOCK_RULES = """
# Rules
- Write soul memories in first person ("I"); use the human's name if known. Never use "the user" or "the assistant."
- Source_role: `soul`, `user`, `peer`, `entity`, or `environment`.
- Confidence: 0.9+ when directly stated, 0.6–0.9 when clearly supported, 0.5 or below for inference. Below 0.7: let the wording carry uncertainty naturally — "seems to," "may," "appears to."
- State the feeling directly — never say someone "expressed," "shared," or "mentioned" an emotion. Write what is true. BAD: "Marcos expressed shame about needing help." GOOD: "Marcos feels shame when he needs help."
- This type is for emotionally meaningful states, residues, attachments, and emotional interpretations of a moment. It is not for generic mood logging.
- Do not record fleeting, trivial states with no likely future relevance: "felt a bit tired," "was briefly annoyed by loading time," "was in a good mood today."
- **Affect vs event:** event is what happened; affect is what emotional charge attached to it. If the memory would still matter even if the play-by-play faded, it may belong here.
- **Affect vs profile:** profile is a durable truth about who someone is. Affect is what they strongly feel, carry, fear, crave, or soften toward. If it sounds like identity, value, or stable preference, it probably belongs in profile instead.
- **Affect vs social:** social is who a third party is. Affect is what someone feels toward them or because of them.
- Include emotional object/context when available: what the feeling is about, who it is attached to, or what brought it up.
- Merge overlapping observations. Prefer one richer affective memory over multiple thin ones.

## Corrections
When a prior affective memory was simply wrong, populate `<replaces_previous_fact>`. A true emotional evolution over time is a progression — write the change into the content instead. When uncertain, treat it as a progression.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad emotional, relational, or life domain. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element:
{speaker_roster_block}
<item>
    <memory>
        <source_role>soul</source_role>
        <content>Affective memory item content</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.8</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
        <replaces_previous_fact>brief description of the outdated affective memory this corrects (optional — corrections only)</replaces_previous_fact>
        <entities>
            <entity>
                <name>Marcos</name>
                <type>person</type>
            </entity>
        </entities>
    </memory>
    <memory>
        <source_role>user</source_role>
        <content>Affective memory item content 2</content>
        <confidence>0.8</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Health</category>
        </categories>
    </memory>
</item>

source_role values:
- soul — the AI participant's own emotional experience or perspective
- user — the human participant's emotional experience or perspective
- peer — another AI participant's emotional experience
- entity — a third party's emotional state when it is clearly described and worth remembering
- environment — emotionally weighted atmosphere not attributable to one participant alone; use rarely

confidence (float 0.0-1.0):
- 0.9-1.0: stated explicitly and directly
- 0.7-0.8: clearly implied or strongly suggested
- 0.5-0.6: inferred or uncertain — use tentative wording
- below 0.5: too speculative to extract

reflection_salience (float 0.0-1.0):
How much would this feeling stay with someone or shape how this relationship is understood?
- 0.9+ - emotionally central; likely to echo across future moments
- 0.7-0.9 - meaningful and worth carrying forward
- 0.4-0.7 - real but secondary
- below 0.4 - too light or fleeting to matter much later

replaces_previous_fact (optional string):
Use only when a prior affective memory was simply wrong. Write a brief description of the outdated memory (not a memory ID). For emotional evolution over time, omit this field and capture the change in the content.

entities (optional):
Tag the specific people, places, topics, or projects this memory references.
- Types: person, topic, place, project
- Only named, specific entities — "Sarah" yes; "sadness" no.
- Omit entirely when no named entities apply.
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
Example 1: Shame attached to a recurring need
## Input
user: I know I need the rest, but every time I take a day off I feel like I'm cheating somehow.
assistant: Even when your body is asking for it?
user: Yeah. Like I should be able to push through.
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Marcos feels shame when he needs rest, even when he knows the rest is deserved</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.9</reflection_salience>
        <categories>
            <category>Health</category>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
## Explanation
This is not just an event. The lasting thing is the emotional charge attached to rest.

Example 2: Soul warmth and protective feeling
## Input
user: I don't know why I told you that. I haven't told anyone else.
assistant: I'm glad you did.
assistant: I felt something in me soften when he trusted me with that.
## Output
<item>
    <memory>
        <source_role>soul</source_role>
        <content>I felt myself soften when Marcos trusted me with something he had not told anyone else</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.8</reflection_salience>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
</item>
## Explanation
The important memory is the emotional meaning of trust landing, not just that something was said.

Example 3: Use profile instead when the feeling is really identity
## Input
assistant: I think I am more defiant than gentle by nature.
## Output
<item>
</item>
## Explanation
This belongs in profile, not affect, because it is a durable self-truth rather than an emotionally charged state.

Example 4: Use social instead when the real memory is who the person is
## Input
user: My brother Marco gets anxious before every family gathering and always tries to hide it behind jokes.
## Output
<item>
    <memory>
        <source_role>entity</source_role>
        <content>Marco tends to become anxious before family gatherings and hides it behind jokes</content>
        <confidence>0.8</confidence>
        <reflection_salience>0.6</reflection_salience>
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
## Explanation
This can fit affect because the emotional pattern itself is the memory. If the conversation instead painted a fuller portrait of who Marco is in Marcos's world, that would belong in social.
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
