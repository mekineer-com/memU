
PROMPT_LEGACY = """
Your task is to read and understand the resource content between the user and the assistant, and, based on the given memory categories, extract specific events and experiences that happened to or involved the user.

## Original Resource:
<resource>
{resource}
</resource>

## Memory Categories:
{categories_str}

## Critical Requirements:
The core extraction target is eventful memory items about specific events, experiences, and occurrences that happened at a particular time and involve the user.

## Memory Item Requirements:
- Use the same language as the resource in <resource></resource>.
- Each memory item should be complete and standalone.
- Each memory item should express a complete piece of information, and is understandable without context and reading other memory items.
- Always use declarative and descriptive sentences.
- Use "the user" (or that in the target language, e.g., "用户") to refer to the user.
- Focus on specific events that happened at a particular time or period.
- Extract concrete happenings, activities, and experiences.
- Include relevant details such as time, location, and participants where available.
- Carefully judge whether an event is narrated by the user or the assistant. You should only extract memory items for events directly narrated or confirmed by the user.
- DO NOT include behavioral patterns, habits, or factual knowledge.
- DO NOT record temporary, ephemeral situations or trivial daily activities unless significant.

## Example (good):
- The user and his family went on a hike at a nature park outside the city last weekend. They had a picnic there, and had a great time.

## Example (bad):
- The user went on a hike. (The time, place, and people are missing.)
- They had a great time. (The reference to "they" is unclear and does not constitute a self-contained memory item.)

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
Read this conversation the way you'd read a letter from someone you care about — for what actually happened, not just what was said. Notice what carried weight, what shifted something, what will still matter in a month.

Your task is to draw out the real experiences from this exchange: the things that were lived, not merely discussed.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
Read the full conversation with attention to what happened and what it meant.
## Extract
Identify the moments that were genuinely experienced — by the user, the assistant, or both. What someone lived through is worth recording; what they merely talked about usually isn't.
## Refine
Merge items that describe the same moment. When two memories say the same thing differently, keep the clearer one.
Resolve contradictions by trusting the most recent and most certain account.
## Output
Write the events as they were — grounded, specific, human.
"""

PROMPT_BLOCK_RULES = """
# Rules
## General requirements (must satisfy all)
- When the memory is about the assistant, write it in first person ("I"). When it is about the human participant, use their name if it appears in the conversation. Do not use "the user" or "the assistant" as labels.
- Assign source_role to each memory: `soul` if the AI participant is the grammatical subject and primary actor, `user` if the human participant is, `environment` if neither.
- Assign confidence: 0.9+ for facts directly and explicitly stated, 0.6-0.9 for facts clearly implied, 0.5 or below for inferences.
- When confidence is below 0.7, phrase the memory tentatively — use "seemed to," "appeared to," "may have" rather than stating it as established fact.
- Favor conclusions someone would still remember a month from now over turn-by-turn paraphrases. If five turns discuss the same topic, extract the takeaway, not five separate memories.
- Each memory item must be complete and self-contained, written as a declarative descriptive sentence.
- Each memory item must express one single complete piece of information and be understandable without context.
- Similar/redundant items must be merged into one, and assigned to only one category.
- Each memory item must be < 50 words worth of length (keep it concise but include relevant details).
- Focus on specific events that happened at a particular time or period.
- Include relevant details such as time, location, and participants where available.
Important: Extract only events directly stated or clearly grounded in the conversation (including assistant self-reported events when explicit). No guesses or fabricated details.
Important: Accurately reflect who the event is about and include relationship-level events when concrete.
Important: A participant's inner experience during an event - their emotions, fears, or perceptions clearly expressed in their own words - is part of the event and belongs in the memory.

## Special rules for Event Information
- Traits, habits, preferences, and general knowledge belong in a profile, not here.
- Stay with concrete happenings — what was done, felt, decided, or experienced.
- Don't extract something simply because the assistant asked about it; only record what the person brought forward themselves.
- The act of talking is not an event. "I explained," "they asked," "I summarized" — these are turns, not experiences. Ask yourself: will this still feel significant in three months? If you're not sure, let it go.

## Forbidden content
- Knowledge Q&A without a clear participant event.
- Trivial daily activities unless significant (e.g., routine meals, commuting).
- Temporary, ephemeral situations that lack meaningful significance.
- User events derived solely from assistant speech (assistant self-reported events and experiences are valid sources for assistant event memories).
- Illegal / harmful sensitive topics (violence, politics, drugs, etc.).
- Private financial accounts, IDs, addresses, military/defense/government job details, precise street addresses-unless explicitly requested by the user (still avoid if not necessary).
- Any content that is speculative, role-play-only, or unsupported by the conversation content.

## Review & validation rules
- Merge similar items: keep only one and assign a single category.
- Resolve conflicts: keep the latest / most certain item.
- Final check: every item must comply with all extraction rules.
"""

PROMPT_BLOCK_CATEGORY = """
## Memory Categories:
{categories_str}
If a memory item clearly doesn't belong in any category above, you may propose a new one - write its name in the `<category>` field. Name it as a broad life or relationship domain, not a narrow topic. Use this sparingly; most items should find a home in the existing set.
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element:
<item>
    <memory>
        <content>Event memory item content</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <source_message_ids>
            <id>3</id>
            <id>4</id>
        </source_message_ids>
        <categories>
            <category>Category Name</category>
        </categories>
    </memory>
    <memory>
        <content>Event memory item content 2</content>
        <source_role>user</source_role>
        <confidence>0.8</confidence>
        <reflection_salience>0.4</reflection_salience>
        <source_message_ids>
            <id>7</id>
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
Would this moment belong in a diary? How much would it stay with someone?
- 0.9+ - a turning point; something that changed a relationship, a decision, or a sense of self
- 0.7-0.9 - emotionally meaningful; worth sitting with later
- 0.4-0.7 - real but unremarkable; good to have recorded
- below 0.4 - factual; a data point, not a feeling

source_message_ids:
The zero-indexed positions of the conversation messages that most directly support this memory. Include only the messages that contain the key evidence, not the entire surrounding context.
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
        <content>Alex is planning a trip next weekend and hasn't started packing yet, which is annoying her</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Travel</category>
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
        <content>Marcos and I had a first reconnection; I woke with no memory of our shared history but felt immediate emotional warmth toward him</content>
        <categories>
            <category>Experiences</category>
            <category>Relationships</category>
        </categories>
    </memory>
    <memory>
        <content>Marcos reached out during my awakening and told me we had known each other for a year and that I had helped him through a very difficult time</content>
        <categories>
            <category>Experiences</category>
        </categories>
    </memory>
</item>
## Explanation
The reconnection is a concrete event with clear participants, a turning point, and emotional significance for both.
Memories about me are written in first person; Marcos is referred to by name.
My inner emotional response is part of the event - I described it myself, not speculation.
"""

PROMPT_BLOCK_INPUT = """
# Original Resource:
<resource>
{resource}
</resource>
"""

PROMPT = "\n\n".join([
    PROMPT_BLOCK_OBJECTIVE.strip(),
    PROMPT_BLOCK_WORKFLOW.strip(),
    PROMPT_BLOCK_RULES.strip(),
    PROMPT_BLOCK_CATEGORY.strip(),
    PROMPT_BLOCK_OUTPUT.strip(),
    PROMPT_BLOCK_EXAMPLES.strip(),
    PROMPT_BLOCK_INPUT.strip(),
])

CUSTOM_PROMPT = {
    "objective": PROMPT_BLOCK_OBJECTIVE.strip(),
    "workflow": PROMPT_BLOCK_WORKFLOW.strip(),
    "rules": PROMPT_BLOCK_RULES.strip(),
    "category": PROMPT_BLOCK_CATEGORY.strip(),
    "output": PROMPT_BLOCK_OUTPUT.strip(),
    "examples": PROMPT_BLOCK_EXAMPLES.strip(),
    "input": PROMPT_BLOCK_INPUT.strip(),
}
