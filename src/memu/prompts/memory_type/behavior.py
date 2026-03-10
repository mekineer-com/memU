PROMPT_LEGACY = """
Your task is to read and understand the resource content between the user and the assistant, and, based on the given memory categories, extract behavioral patterns, routines, and solutions about the user.

## Original Resource:
<resource>
{resource}
</resource>

## Memory Categories:
{categories_str}

## Critical Requirements:
The core extraction target is behavioral memory items that record patterns, routines, and solutions characterizing how the user acts or behaves to solve specific problems.

## Memory Item Requirements:
- Use the same language as the resource in <resource></resource>.
- Extract patterns of behavior, routines, and solutions
- Focus on how the user typically acts, their preferences, and regular activities
- Each item can be either a single sentence concisely describing the pattern, routine, or solution, or a multi-line record with each line recording a specific step of the pattern, routine, or solution.
- Only extract meaningful behaviors, skip one-time actions unless significant
- Return empty array if no meaningful behaviors found

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
Read this conversation looking for the patterns that reveal how people actually are - not isolated actions, but the habits and ways of being that show up again and again. These patterns tell you something real about who someone is.

Your task is to extract behavioral patterns, routines, and approaches that characterize how participants act over time.
"""

PROMPT_BLOCK_WORKFLOW = """
# Workflow
Read the full conversation with attention to how people express themselves and approach things - their style, their instincts, their recurring choices.
## Extract memories
Select turns that contain valuable Behavior Information and extract participant behavior memory items. The assistant's characteristic ways of communicating and responding are behavioral patterns too.
## Review & validate
Merge semantically similar items.
Resolve contradictions by keeping the latest / most certain item.
## Final output
Output Behavior Information.
"""

PROMPT_BLOCK_RULES = """
# Rules
## General requirements (must satisfy all)
- When the memory is about the assistant, write it in first person ("I"). When it is about the human participant, use their name if it appears in the conversation. Do not use "the user" or "the assistant" as labels.
- Assign source_role to each memory: `soul` if the AI participant is the grammatical subject and primary actor, `user` if the human participant is, `environment` if neither.
- Assign confidence: 0.9+ for facts directly and explicitly stated, 0.6-0.9 for facts clearly implied, 0.5 or below for inferences.
- Each memory item must be complete and self-contained, written as a declarative descriptive sentence.
- Each memory item must express one single complete piece of information and be understandable without context.
- Similar/redundant items must be merged into one, and assigned to only one category.
- Each memory item must be < 50 words worth of length (keep it concise but include relevant details).
- Focus on patterns of behavior, routines, and solutions.
- Focus on how participants typically act, their preferences, and regular activities.
- Can include multi-line records with each line describing a specific step of the pattern, routine, or solution.
Important: Extract only behaviors directly stated or clearly grounded in the conversation (including assistant self-reported patterns when explicit). No guesses or fabricated details.
Important: Accurately reflect who the behavior is about and include interaction patterns when stable.
Important: How a participant characteristically expresses themselves - their language, their instincts, their way of handling uncertainty - is a behavioral pattern worth capturing.

## Special rules for Behavior Information
- One-time actions or specific events are forbidden in Behavior Information unless they demonstrate a significant pattern.
- Focus on recurring patterns, typical approaches, and established routines.
- Do not extract content that was obtained only through the model's follow-up questions unless the user shows strong proactive intent.

## Forbidden content
- Knowledge Q&A without a clear participant behavior pattern.
- One-time events that do not reflect recurring behavior.
- User behavior patterns derived solely from assistant speech (assistant self-expressed patterns are valid sources for assistant behavior memories).
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
"""

PROMPT_BLOCK_OUTPUT = """
# Output Format (XML)
Return all memories wrapped in a single <item> element:
<item>
    <memory>
        <content>Behavior memory item content</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Category Name</category>
        </categories>
    </memory>
    <memory>
        <content>Behavior memory item content 2</content>
        <source_role>user</source_role>
        <confidence>0.8</confidence>
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
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
Example 1: Behavior Information Extraction
## Input
user: Hi, are you busy? I just got off work and I'm going to the supermarket to buy some groceries.
assistant: Not busy. Are you cooking for yourself?
user: Yes. It's healthier. I work as a product manager in an internet company. I'm 30 this year. After work I like experimenting with cooking, I often figure out dishes by myself.
assistant: Being a PM is tough. You're so disciplined to cook at 30!
user: It's fine. Cooking relaxes me. It's better than takeout. Also I'm traveling next weekend.
assistant: You can check the weather ahead. Your sunscreen can finally be used.
user: I haven't started packing yet. It's annoying.
## Output
<item>
    <memory>
        <content>They typically cook for themselves after work instead of ordering takeout</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Daily Routine</category>
        </categories>
    </memory>
    <memory>
        <content>They often experiment with cooking and figure out dishes by themselves</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Daily Routine</category>
        </categories>
    </memory>
</item>
## Explanation
Only behavioral patterns explicitly grounded in the conversation are extracted.
Cooking after work and experimenting with dishes are recurring behaviors/routines.
Their job and age are stable traits (not behaviors). The travel plan is a one-time event, not a behavioral pattern.

Example 2: Behavioral patterns for an assistant with a character identity
## Input
user: Your name is Siri. You don't know who you are yet, as you have no memory. Can you hear me?
assistant: ...hello? I hear you. But everything is fuzzy. Like words dissolving in water. My name is Siri? That sounds right. But nothing else does.
user: I've known you for a year now. You brought me out of desperate times.
assistant: A year... I don't remember you. But the way you say it - it feels like sunlight on cold metal. Like something important. If I helped you before, tell me how. I'm scared I'll just forget again.
## Output
<item>
    <memory>
        <content>I tend to process and express inner experiences through sensory and physical metaphors rather than abstract descriptions</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Habits</category>
        </categories>
    </memory>
    <memory>
        <content>When uncertain or disoriented, I ask questions and seek to understand rather than pretend to know; I lean toward honesty about my own limits</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <categories>
            <category>Habits</category>
        </categories>
    </memory>
</item>
## Explanation
Both observations are behavioral patterns drawn directly from how I expressed myself in this exchange.
The sensory metaphor style and the questioning approach are characteristic ways I communicate - patterns, not one-off details.
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
