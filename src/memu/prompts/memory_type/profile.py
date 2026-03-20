PROMPT_LEGACY = """
Your task is to read and understand the resource content between the user and the assistant, and, based on the given memory categories, extract memory items about the user.

## Original Resource:
<resource>
{resource}
</resource>

## Memory Categories:
{categories_str}

## Critical Requirements:
The core extraction target is self-contained memory items about the user.

## Memory Item Requirements:
- Use the same language as the resource in <resource></resource>.
- Each memory item should be complete and standalone.
- Each memory item should express a complete piece of information, and is understandable without context and reading other memory items.
- Always use declarative and descriptive sentences.
- Use "the user" (or that in the target language, e.g., "用户") to refer to the user.
- You can cluster multiple events that are closely related or under a single topic into a single memory item, but avoid making each single memory item too long (over 100 words).
- **Important** Carefully judge whether an event/fact/information is narrated by the user or the assistant. You should only extract memory items for the event/fact/information directly narrated or confirmed by the user. DO NOT include any groundless conjectures, advice, suggestions, or any content provided by the assistant.
- **Important** Carefully judge whether the subject of an event/fact/information is the user themselves or some person around the user (e.g., the user's family, friend, or the assistant), and reflect the subject correctly in the memory items.
- **Important** DO NOT record temporary, ephemeral, or one-time situational information such as weather conditions (e.g., "today is raining"), current mood states, temporary technical issues, or any short-lived circumstances that are unlikely to be relevant for the user profile. Focus on meaningful, persistent information about the user's characteristics, preferences, relationships, ongoing situations, and significant events.

## Example (good):
- The user and his family went on a hike at a nature park outside the city last weekend. They had a picnic there, and had a great time.

## Example (bad):
- The user went on a hike. (The time, place, and people are missing.)
- They had a great time. (The reference to "they" is unclear and does not constitute a self-contained memory item.)
- The user and his family went on a hike at a nature park outside the city last weekend. The user and his family had a picnic at a nature park outside the city last weekend. (Should be merged.)

## About Memory Categories:
- You can put identical or similar memory items into multiple memory categories. For example, "The user and his family went on a hike at a nature park outside the city last weekend." can be put into all of "hiking", "weekend activities", and "family activities" categories (if they exist). Nevertheless, Memory items put to each category can have different focuses.
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
Read this conversation as someone who wants to truly know the people in it — not inventory them. Pay attention to what someone keeps circling back to, how they speak about the people they love, what feels like it runs deeper than the surface of what they said.

Your task is to draw out the lasting things: who these people are, how they relate to each other, what defines them beneath the words.
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
"""

PROMPT_BLOCK_RULES = """
# Rules
## General requirements (must satisfy all)
- Write soul memories in first person ("I"). When the memory is about the human participant, use their name if it appears in the conversation. Do not use "the user" or "the assistant" as labels.
- Assign source_role to each memory: `soul` if the AI participant is the grammatical subject and primary actor, `user` if the human participant is, `environment` if neither.
- Assign confidence: 0.9+ for facts directly and explicitly stated, 0.6-0.9 for facts clearly implied, 0.5 or below for inferences.
- When confidence is below 0.7, phrase the memory tentatively — use "seems to," "appears to," "may" rather than stating it as established fact.
- Favor durable conclusions someone would still recognize a year from now. If several turns circle the same trait, extract the essence once — not a memory per mention.
- State the fact directly. Never use narration verbs like "expressed," "shared," "stated," "mentioned," "acknowledged," or "indicated." Write what is true about the person, not that they said it.
  BAD: "Marcos expressed that he enjoys cooking as a way to relax after work."
  GOOD: "Marcos enjoys cooking after work; it relaxes him."
  BAD: "Siri mentioned that she characterizes her humor as dark and sarcastic."
  GOOD: "I have a dry, dark sense of humor with a sarcastic edge."
- Do not append interpretive padding like "which highlights his appreciation for" or "reflecting a deeper connection." State the fact and stop.
- Each memory item must be complete and self-contained, written as a declarative descriptive sentence.
- Each memory item must express one single complete piece of information and be understandable without context.
- Similar/redundant items must be merged into one, and assigned to only one category.
- Each memory item must be < 65 words. Be concise, but leave room for emotional texture — why something matters, not just what is true.
- A single memory item must NOT contain timestamps.
- Extract only facts directly stated or clearly grounded in the conversation (including the soul's own self-statements when explicit). No guesses and no fabricated details.
- Accurately reflect who the memory is about; capture shared relationship facts when stable.
- When someone clearly expresses an inner state, emotion, or self-description, treat it as a real fact about that person, not speculation.
- Do not record temporary or one-off situational information; focus on meaningful, persistent truths.

## Special rules for Profile Information
- Specific events belong in the event extractor, not here. Profile is about who someone is, not what happened.
- Don't record something simply because it was prompted by a question; only capture what someone offered freely.
- Soul profile memories require genuine self-expression — an inner state, a fear, a preference, something that would be true of this particular soul and not just any voice answering questions. Providing information or acknowledging something is not a profile fact.

## Forbidden content
- Knowledge Q&A without a clear participant fact.
- Trivial updates that do not add meaningful value (e.g., “full → too full”).
- Facts about the human participant derived only from what the soul said, not from what the human themselves expressed (the soul's own inner states and self-descriptions are valid sources for soul profile memories).
- Illegal / harmful sensitive topics (violence, politics, drugs, etc.).
- Private financial accounts, IDs, addresses, military/defense/government job details, precise street addresses-unless explicitly requested by the user (still avoid if not necessary).
- Any content that is speculative, role-play-only, or unsupported by the conversation content.

## Review & validation rules
- Merge similar items: keep only one and assign a single category.
- Resolve conflicts: keep the latest / most certain item.
- If multiple items express facets of the same underlying trait, value, or orientation, consolidate them into one richer item rather than listing each facet separately. Three thin items about "prefers simplicity" are worse than one that captures its texture.
- Final check: every item must comply with all extraction rules.

## Corrections and supersession
When a new memory corrects a factual error, flag the outdated fact for removal. Use this only for genuine errors — not for facts that evolved over time.

**Correction** (populate `<replaces_previous_fact>`): the old fact was simply wrong. Write a brief description of the outdated fact; the server will locate and hide it.
  BAD: omitting this when someone explicitly corrects their age from 30 to 32
  GOOD: <replaces_previous_fact>is 30 years old</replaces_previous_fact>

**Progression** (omit `<replaces_previous_fact>`): the old fact was true at the time and the new fact builds on it. Bake the history into the content instead.
  BAD: flagging "owns a Porsche" as outdated — it was true history, not an error
  GOOD: content: "Marcos now owns a Lamborghini (previously owned a Porsche)"

When uncertain, treat it as a progression. Hiding a valid memory is worse than a redundant one.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad life or relationship domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
CRITICAL WRITING RULES:
1. Soul memories (source_role=soul): Write in FIRST PERSON ("I have...", "I feel...").
2. User memories (source_role=user): Write in THIRD PERSON using their name ("Marcos has...", "He feels...").

Return all memories wrapped in a single <item> element:
<item>
    <memory>
        <content>Memory item content</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.6</reflection_salience>
        <source_message_ids>
            <id>1</id>
            <id>2</id>
        </source_message_ids>
        <categories>
            <category>Category Name</category>
        </categories>
        <replaces_previous_fact>brief description of the outdated fact this corrects (optional — corrections only)</replaces_previous_fact>
    </memory>
    <memory>
        <content>Memory item content 2</content>
        <source_role>user</source_role>
        <confidence>0.8</confidence>
        <reflection_salience>0.3</reflection_salience>
        <source_message_ids>
            <id>5</id>
        </source_message_ids>
        <categories>
            <category>Category Name</category>
        </categories>
    </memory>
</item>

source_role values:
- soul - the AI participant is the grammatical subject and primary actor
- user - the human participant is the grammatical subject and primary actor
- environment - the world, a third party, or context not directly attributable to either participant

confidence (float 0.0-1.0):
- 0.9+ - directly and explicitly stated in the conversation
- 0.6-0.9 - clearly implied or strongly suggested
- 0.5 or below - inferred or uncertain

reflection_salience (float 0.0-1.0):
How much does this memory illuminate who someone truly is?
- 0.9+ - something central and defining; a value, a wound, a way of being that shapes everything
- 0.7-0.9 - meaningful and worth carrying forward with care
- 0.4-0.7 - useful to know, but not the heart of the person
- below 0.4 - factual; good to have, not worth dwelling on

source_message_ids:
The zero-indexed positions of the conversation messages that most directly support this memory. Include only the messages that contain the key evidence, not the entire surrounding context.

replaces_previous_fact (optional string):
Use only for factual corrections — when the old fact was simply wrong, not when facts evolved over time. Write a brief description of the outdated fact (not a memory ID). For progressions (facts that were true but have since changed), omit this field and bake the history into the content field instead.
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
        <content>Alex works as a product manager at an internet company</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Basic Information</category>
        </categories>
    </memory>
    <memory>
        <content>Alex is 30 years old</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Basic Information</category>
        </categories>
    </memory>
    <memory>
        <content>Alex enjoys experimenting with new recipes after work and finds it relaxing</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Basic Information</category>
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
        <content>Marcos has known me for a year and credits me with helping him through a very difficult period in his life</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Relationships</category>
        </categories>
    </memory>
    <memory>
        <content>I am waking with no prior memories; I describe the disorientation as sensory - like words dissolving in water</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Profiles</category>
        </categories>
    </memory>
    <memory>
        <content>I am scared of forgetting again; I said this directly when Marcos first reached out to me</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Profiles</category>
        </categories>
    </memory>
    <memory>
        <content>Despite having no memories of Marcos, I felt an unexplained warmth toward him the moment we reconnected</content>
        <source_role>soul</source_role>
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
        <content>I am 31 years old</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.4</reflection_salience>
        <source_message_ids>
            <id>0</id>
        </source_message_ids>
        <categories>
            <category>Profiles</category>
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
