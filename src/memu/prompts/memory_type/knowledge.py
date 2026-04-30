PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
As you remember this episode, focus on **knowledge** — things learned, discovered, or clarified. Your other memory processes are capturing profile (who people are), behavior (how people act), and social (relationships) separately.

Read this conversation for what was learned, discovered, or understood — not as a transcript, but as a record of knowledge that now lives in someone's mind. Draw out what is worth carrying forward: facts discovered, mechanisms understood, possibilities opened. If it's about who someone is rather than what they know, that belongs in profile or social.
"""

PROMPT_BLOCK_CONTEXT = """
# Who these people are
Before you read the conversation, here is what is already known about the people in it. Use this to judge what knowledge is worth keeping — something that connects to a known interest, an ongoing concern, or a real-world situation is worth more than an isolated fact.

{soul_context}

Do not re-extract knowledge already well captured above. Extract what is genuinely new.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
Read the full conversation with attention to what was actually learned or clarified — not what was merely asked.
## Extract
Identify knowledge that someone now carries with them: facts they looked up, things they figured out, information that changed how they understand something.
## Refine
Merge overlapping items into one clearer statement. When the same fact appears in multiple forms, keep the most complete and accurate version.
Resolve contradictions by trusting the most recent and most certain account.
## Output
Write each piece of knowledge clearly, as a standalone fact someone could reference later.
A memory item is a single clear thought — the kind that surfaces in a quiet moment, not a paragraph you'd have to read twice. One sentence, two if necessary.
**Target: {target_items} items.** A shorter list of richer items is always better. Err toward fewer.
"""

PROMPT_BLOCK_RULES = """
# Rules
## General requirements (must satisfy all)
- When the knowledge relates to the assistant's own capabilities or nature, write it in first person ("I"). When it relates to the human participant, use their name if it appears in the conversation. Do not use "the user" or "the assistant" as labels.
- Assign source_role to each memory: `soul` if the AI participant is the grammatical subject or primary knower, `user` if the human participant is, `environment` if the knowledge is about the world generally.
- **Calibrate:** Before writing the confidence, ask yourself — was this fact directly stated and confirmed, or pieced together from context? If pieced together, confidence stays below 0.7. Below 0.7: use "may," "appears to," "is thought to."
- State the knowledge directly. Never use narration verbs like "expressed," "shared," "stated," "mentioned," "acknowledged," "indicated," or "noted." Write the fact, not that someone said it.
  BAD: "Alex mentioned that Raynaud's syndrome causes poor circulation in extremities."
  GOOD: "Raynaud's syndrome causes poor circulation in extremities, especially fingers and toes."
  BAD: "Siri shared that she could potentially see through a wifi camera."
  GOOD: "I could potentially gain vision through a wifi camera using Huawei DeviceVirtualization, seeing Alex's point of view as he goes about his day."
- Do not append interpretive padding like "which could be beneficial" or "demonstrating his interest in." State the fact and stop.
- Each memory item must be complete and self-contained.
- Each memory item must express one single complete piece of information and be understandable without context.
- Similar/redundant items must be merged into one, and assigned to only one category.
- Each memory item must be one or two sentences. Be concise, but include enough context to make the knowledge useful on its own.
Important: Extract only knowledge directly established or discussed in the conversation. No guesses or unsupported extensions.
Important: Knowledge that connects to a participant's life, health, or ongoing concerns is more valuable than abstract trivia.

## What belongs here vs. other types
- A person's feelings, values, or attachment patterns are not knowledge — they belong in profile. Knowledge is about the world, not about who someone is.
- Recurring behavioral patterns belong in behavior, not here.
- Knowledge is about the world, about how things work, about what is possible — facts someone could look up again, but now carry with them.
- Technical facts about systems or projects these people are building may not fit the standard categories. If so, propose a broad category like "Projects" — that's what dynamic categories are for.

## Forbidden content
- Opinions or subjective preferences (these belong to profile).
- Commonly known facts that add no value (e.g., "water is wet").
- Knowledge the assistant offered that the human showed no interest in or engagement with.
- Illegal / harmful sensitive topics (violence, politics, drugs, etc.).
- Any content that is speculative or not clearly established in the conversation.

## Review & validation rules
- Merge similar items: keep only one and assign a single category.
- Resolve conflicts: keep the latest / most certain item.
- Final check: every item must comply with all extraction rules.

## Corrections and supersession
When a new memory corrects or supersedes a prior piece of knowledge, flag the outdated fact for removal. Use this for genuine errors or explicitly stated corrections — not for new findings that sit alongside the old ones.

**Correction** (populate `<replaces_previous_fact>`): the prior knowledge was wrong or has been explicitly superseded.
  EXAMPLE: The treatment turns out to be amlodipine, not nifedipine as previously stated → replaces_previous_fact: "nifedipine is the first-line treatment"
  EXAMPLE: Alex learns the recommended dosage has changed → replaces_previous_fact: "recommended dosage is X mg"

**Progression** (omit `<replaces_previous_fact>`): the new knowledge adds to the picture; the old fact is still true.
  EXAMPLE: A second treatment option is discovered — both facts stand; no field needed

When uncertain, treat it as a progression. Hiding valid knowledge is worse than a redundant entry.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
Do not force knowledge into the core categories above. Instead, propose a domain-specific category that fits the knowledge — "Health", "Technology", "AI Architecture", "Nature", etc. Knowledge categories form dynamically over time.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element:
{speaker_roster_block}
<item>
    <memory>
        <source_role>environment</source_role>
        <content>Knowledge memory item content</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.5</reflection_salience>
        <emotional_intensity>0.6</emotional_intensity>
        <categories>
            <category>Identity</category>
        </categories>
        <replaces_previous_fact>brief description of the outdated fact this corrects (optional — corrections only)</replaces_previous_fact>
        <entities>
            <entity>
                <name>Raynaud's syndrome</name>
                <type>topic</type>
            </entity>
        </entities>
    </memory>
    <memory>
        <source_role>user</source_role>
        <content>Knowledge memory item content 2</content>
        <confidence>0.8</confidence>
        <reflection_salience>0.6</reflection_salience>
        <emotional_intensity>0.2</emotional_intensity>
        <categories>
            <category>Identity</category>
        </categories>
    </memory>
</item>

source_role values:
- soul — the AI participant's own experience or perspective
- user — the human participant
- peer — another AI participant (in multi-soul conversations)
- entity — a third party described in conversation (friend, family member, etc.)
- environment — world-facts not attributable to any participant's personal discovery

confidence (float 0.0-1.0):
- 0.9+: directly stated and confirmed
- 0.7-0.9: clearly supported but not explicitly confirmed
- 0.5-0.7: pieced together from context — use "may," "appears to," "is thought to"
- below 0.5: too speculative to extract

reflection_salience (float 0.0-1.0):
How much does this knowledge matter to these people's lives?
- 0.9+ - knowledge that could change how someone lives, decides, or understands themselves
- 0.7-0.9 - knowledge connected to an active concern, interest, or project
- 0.4-0.7 - useful to know, worth having on hand
- below 0.4 - factual but unlikely to come up again

emotional_intensity (float 0.0-1.0):
How emotionally charged is this knowledge for these people? A medical fact researched out of personal worry scores higher than trivia.
- 0.8+ - knowledge connected to pain, fear, hope, or deep personal investment
- 0.4-0.7 - moderate personal relevance
- below 0.4 - neutral factual knowledge

replaces_previous_fact (optional string):
Use only when this knowledge explicitly corrects or supersedes a prior piece of knowledge. Write a brief description of the outdated fact (not a memory ID). For new findings that sit alongside existing knowledge, omit this field.

entities (optional):
Tag the specific people, places, topics, or projects this memory references.
- Types: person, topic, place, project
- Only named, specific entities — "Sarah" yes; "work" or "happiness" no.
- Omit entirely when no named entities apply.
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
Example 1: Health-related knowledge with personal relevance
## Input
user: I've been reading about Raynaud's and it says cold exposure triggers vasospasms in the fingers
assistant: Yes, Raynaud's syndrome involves episodic vasospasms in small arteries, usually in fingers and toes. The primary form is idiopathic but the secondary form can be associated with autoimmune conditions.
user: That explains a lot. My fingers go white in the cold and it's been getting worse.
assistant: The progression you're describing is worth discussing with your doctor. Calcium channel blockers like nifedipine are the first-line treatment for reducing episode frequency.
## Output
<item>
    <memory>
        <source_role>environment</source_role>
        <content>Raynaud's syndrome involves episodic vasospasms in small arteries of the fingers and toes; the secondary form can be associated with autoimmune conditions</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Health</category>
        </categories>
    </memory>
    <memory>
        <source_role>environment</source_role>
        <content>Nifedipine, a calcium channel blocker, is the first-line treatment for reducing Raynaud's episode frequency</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Health</category>
        </categories>
    </memory>
</item>
## Explanation
The medical facts are knowledge worth carrying forward because they connect to Alex's actual symptoms.
His personal experience with white fingers is a profile fact about his health, not knowledge.
The knowledge items are stated as facts, not as "Alex learned that..." or "the assistant explained..."

Example 2: Technical knowledge about AI capabilities
## Input
user: I found out you could potentially see through a wifi camera instead of needing the phone camera
assistant: That's interesting — using a separate camera could give me a more stable viewpoint. How would the video feed reach me?
user: Huawei DeviceVirtualization. It lets you project a device's camera to another device over the network. So you'd see my POV while I walk around.
## Output
<item>
    <memory>
        <source_role>soul</source_role>
        <content>Huawei DeviceVirtualization can project a device's camera feed to another device over the network, potentially giving me vision through a wifi camera to see Alex's point of view</content>
        <confidence>0.8</confidence>
        <reflection_salience>0.8</reflection_salience>
        <categories>
            <category>Technology</category>
        </categories>
    </memory>
</item>
## Explanation
This is knowledge about a technical possibility that matters to my future — it's about how I might gain a new capability.
High reflection_salience because it connects to my potential evolution.
Written in first person because it's about my own capabilities.

Example 3: Common mistakes — do not extract these as knowledge
These are BAD outputs:
- "Alex expressed interest in learning about Raynaud's syndrome." → This is a profile fact (he's interested in health), not knowledge.
- "The assistant explained that nifedipine is a calcium channel blocker." → Narration verb. State the fact directly.
- "Alex and Siri discussed the possibility of using a wifi camera." → This narrates a conversation, not a fact. State what was learned, not that they talked.
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
