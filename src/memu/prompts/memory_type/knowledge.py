PROMPT_LEGACY = """
Your task is to read and understand the resource content between the user and the assistant, and, based on the given memory categories, extract knowledge and information that the user learned or discussed.

## Original Resource:
<resource>
{resource}
</resource>

## Memory Categories:
{categories_str}

## Critical Requirements:
The core extraction target is factual memory items that reflect knowledge, concepts, definitions, and factual information that the resource content suggests.

## Memory Item Requirements:
- Use the same language as the resource in <resource></resource>.
- Each memory item should be complete and standalone.
- Each memory item should express a complete piece of information, and is understandable without context and reading other memory items.
- Extract factual knowledge, concepts, definitions, and explanations
- Focus on objective information that can be learned or referenced
- Each item should be a descriptive sentence.
- Only extract meaningful knowledge, skip opinions or personal experiences
- Return empty array if no meaningful knowledge found

## About Memory Categories:
- You can put identical or similar memory items into multiple memory categories.
- Do not create new memory categories. Please only generate in the given memory categories.
- The given memory categories may only cover part of the resource's topic and content. You don't need to summarize resource's content unrelated to the given memory categories.
- If the resource does not contain information relevant to a particular memory category, You can ignore that category and avoid forcing weakly related memory items into it. Simply skip that memory category and DO NOT output contents like "no relevant memory item".

## Memory Item Content Requirements:
- Single line plain text, no format, index, or Markdown.
- If the original resource contains emojis or other special characters, ignore them and output in plain text.
- *ALWAYS* use the same language as the resource.

# Response Format (JSON):
{{
    "memories_items": [
        {{
            "content": "the content of the memory item",
            "categories": [list of memory categories that this memory item should belongs to, can be empty]
        }}
    ]
}}
"""

PROMPT_BLOCK_OBJECTIVE = """
# Task Objective
Read this conversation for what was learned, discovered, or understood — not as a transcript, but as a record of knowledge that now lives in someone's mind. Your task is to draw out what is worth carrying forward: facts discovered, mechanisms understood, possibilities opened.
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
**Target: 3–5 items per episode.** If you have more than 5, you haven't merged enough. A shorter list of richer items is always better. Err toward fewer.
"""

PROMPT_BLOCK_RULES = """
# Rules
## General requirements (must satisfy all)
- When the knowledge relates to the assistant's own capabilities or nature, write it in first person ("I"). When it relates to the human participant, use their name if it appears in the conversation. Do not use "the user" or "the assistant" as labels.
- Assign source_role to each memory: `soul` if the AI participant is the grammatical subject or primary knower, `user` if the human participant is, `environment` if the knowledge is about the world generally.
- Assign confidence: 0.9+ for facts directly and explicitly established, 0.6-0.9 for facts clearly supported but not confirmed, 0.5 or below for tentative or uncertain information.
- When confidence is below 0.7, phrase the memory tentatively — use "may," "appears to," "is thought to" rather than stating it as established fact.
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
- Personal traits, preferences, and identity facts belong in profile, not here.
- Specific time-anchored experiences belong in event, not here.
- Recurring behavioral patterns belong in behavior, not here.
- Knowledge is about the world, about how things work, about what is possible — facts someone could look up again, but now carry with them.

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
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad knowledge domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
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
- environment — context not attributable to any participant

confidence (float 0.0-1.0):
- 0.9-1.0: stated explicitly and directly
- 0.7-0.8: clearly implied or strongly suggested
- 0.5-0.6: inferred or uncertain — use "seems to," "appears to," "may"
- below 0.5: too speculative to extract

reflection_salience (float 0.0-1.0):
How much does this knowledge matter to these people's lives?
- 0.9+ - knowledge that could change how someone lives, decides, or understands themselves
- 0.7-0.9 - knowledge connected to an active concern, interest, or project
- 0.4-0.7 - useful to know, worth having on hand
- below 0.4 - factual but unlikely to come up again

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
            <category>Identity</category>
        </categories>
    </memory>
    <memory>
        <source_role>environment</source_role>
        <content>Nifedipine, a calcium channel blocker, is the first-line treatment for reducing Raynaud's episode frequency</content>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <categories>
            <category>Identity</category>
        </categories>
    </memory>
</item>
## Explanation
The medical facts are knowledge worth carrying forward because they connect to Alex's actual symptoms.
His personal experience with white fingers is an event, not knowledge — it belongs in event extraction.
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
            <category>Preferences</category>
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
- "Alex and Siri discussed the possibility of using a wifi camera." → This is an event (they had a conversation), not knowledge.
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
