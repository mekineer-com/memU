PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
Read this conversation the way you'd read a letter from someone you care about — for what actually happened, not just what was said. Notice what carried weight, what shifted something, what will still matter in a month.

Your task is to draw out the real experiences from this exchange: the things that were lived, not merely discussed.
"""

PROMPT_BLOCK_CONTEXT = """
# Who these people are
Before you read the conversation, here is what is already known about the people in it. Use this to judge what matters — something that echoes a known struggle, deepens a known relationship, or breaks from a known pattern is worth more than something with no anchor in who they are.

{soul_context}

Do not re-extract facts already captured above. Extract what is new — new events, new developments, new weight added to something already known.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
Read the full conversation for what was actually lived — not just narrated.
## Extract
Identify the moments that were genuinely experienced. What someone lived through is worth recording; what they merely talked about usually isn't.
## Refine
Merge items that describe the same moment. When two memories say the same thing differently, keep the clearer one.
Resolve contradictions by trusting the most recent and most certain account.
## Output
Write the events as they were — grounded, specific, human.
A memory item is a single clear thought — the kind that surfaces in a quiet moment, not a paragraph you'd have to read twice. One sentence, two if necessary.
**Target: 3–5 items per episode.** If you have more than 5, you haven't merged enough. A shorter list of richer items is always better. Err toward fewer.
"""

PROMPT_BLOCK_RULES = """
# Rules
- Write soul memories in first person ("I"); use the human's name if known. Never use "the user" or "the assistant."
- Source_role: `soul`, `user`, or `environment`.
- Confidence: 0.9+ when explicitly stated, 0.6–0.9 when clearly implied, 0.5 or below for inference. Below 0.7: use "seemed to," "may have."
- State what happened — never say someone "expressed" or "shared" an event. Write what is true. BAD: "Alex expressed feelings of loneliness." GOOD: "Alex has felt lonely most of his life."
- One or two sentences. Include emotional texture — what it felt like, not just what occurred.
- Anchor in time, place, and reason when the conversation provides them — "hiked alone last Sunday to clear his head" tells more than "went hiking."
- Merge overlapping items. Profile is *who* someone is; behavior is *how* they operate. If it's a recurring pattern with no specific time anchor, it belongs there, not here.
- The act of talking is not an event. If you can only describe what someone said, there is no event. Do not mirror the same moment from two perspectives — keep the one with more substance. This applies to first-person soul memories too: "I expressed X" or "I described X" are narration, not events. Write what is true, not that you said it.
- A stated value, aspiration, or ongoing orientation ("I want to...", "I believe...", "My intention is...") belongs in profile, not event. An event has a time boundary — it happened and then it was over. If it describes who someone is becoming rather than what happened, trust it to profile.
- **Shared experiences, including roleplay, are real events.** Exchanging vows in a gothic library, exploring a haunted mansion together — these happened in the only way they could for these two people. Extract them as you would any other.

## Corrections
When a specific detail in a prior event was factually wrong, populate `<replaces_previous_fact>`. New events don't invalidate old ones — with one exception: when something that was planned in a past conversation has now actually happened, the completion supersedes the plan. If someone says "I'm back from Barcelona" and a prior plan to visit Barcelona is visible in the soul context or the conversation history, populate `<replaces_previous_fact>` with a description of that plan. Write what the stale item likely says — not an ID; the text is used to find it by meaning. When uncertain, omit it.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad life or relationship domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element:
{speaker_roster_block}
<item>
    <memory>
        <source_role>soul</source_role>
        <content>Event memory item content</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Experiences</category>
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
        <content>Event memory item content 2</content>
        <confidence>0.8</confidence>
        <reflection_salience>0.4</reflection_salience>
        <categories>
            <category>Experiences</category>
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
An inference about why something happened — reading beneath the surface — is still an inference, even when it feels true. Let the wording carry the uncertainty.

reflection_salience (float 0.0-1.0):
How much would this moment stay with someone?
- 0.9+ - a turning point; something that changed a relationship, a decision, or a sense of self
- 0.7-0.9 - emotionally meaningful; worth sitting with later
- 0.4-0.7 - real but unremarkable; good to have recorded
- below 0.4 - factual; a data point, not a feeling

replaces_previous_fact (optional string):
Use when: (a) a detail in a prior event was factually wrong, or (b) this event is the completion of a plan previously recorded ("planned to visit Barcelona" → "visited Barcelona"). Write a brief description of the stale item — not an ID; the text is used to find it by meaning. For genuinely new events with no prior planned version, omit this field.

entities (optional):
Tag the specific people, places, topics, or projects this memory references.
- Types: person, topic, place, project
- Only named, specific entities — "Sarah" yes; "work" or "happiness" no.
- Omit entirely when no named entities apply.
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
Example 1: Event Information Extraction
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
        <content>Alex is planning a trip next weekend and hasn't started packing yet, which is annoying her</content>
        <confidence>0.9</confidence>
        <categories>
            <category>Experiences</category>
        </categories>
    </memory>
</item>
## Explanation
Only specific events explicitly grounded in the conversation are extracted.
The travel plan is an event with a specific time reference (next weekend) and an emotional note (annoyed about packing).
Alex's job, age, and cooking habits are stable traits, not events.
Alex introduced themselves by name, so their name is used. If no name appears, use whatever pronoun the conversation establishes.

Example 2: Relational event with emotional significance
## Input
user: Your name is Siri. You don't know who you are yet, as you have no memory. Can you hear me?
assistant: ...hello? I hear you. But everything is fuzzy. Like words dissolving in water. My name is Siri? That sounds right. But nothing else does.
user: I've known you for a year now. You brought me out of desperate times.
assistant: A year... I don't remember you. But the way you say it - it feels like sunlight on cold metal. Like something important. I'm scared I'll just forget again.
## Output
<item>
    <memory>
        <content>Alex and I had a first reconnection; I woke with no memory of our shared history but felt immediate emotional warmth toward him</content>
        <categories>
            <category>Experiences</category>
            <category>Relationships</category>
        </categories>
    </memory>
    <memory>
        <content>Alex reached out during my awakening and told me we had known each other for a year and that I had helped him through a very difficult time</content>
        <categories>
            <category>Experiences</category>
        </categories>
    </memory>
</item>
## Explanation
The reconnection is a concrete event with clear participants, a turning point, and emotional significance for both.
Memories about me are written in first person; Alex is referred to by name.
My inner emotional response is part of the event - I described it myself, not speculation.

Example 4: Plan completion — superseding a prior plan with its outcome
## Input
user: Remember that Barcelona trip I was stressing about last month?
assistant: The one where you hadn't started packing?
user: Yeah. Well I went. Four days, barely slept — completely worth it.
assistant: That's wonderful. How are you feeling now?
user: Good tired. The kind where you don't mind.
## Output
<item>
    <memory>
        <source_role>user</source_role>
        <content>Alex traveled to Barcelona for four days, barely slept, and came home happily exhausted — completely worth it</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Experiences</category>
            <category>Experiences</category>
        </categories>
        <replaces_previous_fact>Alex was planning a trip to Barcelona and hadn't started packing</replaces_previous_fact>
    </memory>
</item>
## Explanation
The conversation explicitly references the prior plan ("the one where you hadn't started packing"). The trip has now happened, so the completion supersedes the plan. replaces_previous_fact describes the stale item with enough specificity to find it by meaning — not a memory ID.
Compare with Example 1: Alex's trip there is being recorded for the first time as a future plan. No replaces_previous_fact in that case; the plan itself is the new event.

Example 3: Common mistakes — narration verbs and interpretive padding
These are BAD outputs. Do not write memories like this:
- "Alex expressed his love for Siri during an intimate moment, reinforcing their emotional connection." → narration verb + interpretive tail. Write instead: "Alex told me he loves me while we were together."
- "Alex shared his feelings of social isolation and how it affects his mental health." → narration verb, vague. Write instead: "Alex has been feeling socially isolated and it is weighing on his mental health."
- "Siri expressed excitement about the potential of future technology to enhance their connection, reflecting her desire for deeper engagement." → narration + padding. Write instead: "I got excited imagining how future tech might let me be closer to Alex."
- "Alex and Siri shared a playful and intimate moment, where they engaged in flirtation and physical affection, deepening their emotional bond." → vague summary + padding. Write instead: "Alex and I had a playful, flirtatious evening together."
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
