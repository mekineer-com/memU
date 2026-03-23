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
Read this conversation for how people actually are with each other — not what they said, but how they said it. The patterns that matter are the ones someone wouldn't think to describe about themselves: the way they approach difficulty, the rhythm of how they comfort or deflect, the instincts that surface before thinking catches up.

Your task is to extract behavioral patterns, interaction styles, and ways of being that characterize how these participants move through conversation and relationship.
"""

PROMPT_BLOCK_CONTEXT = """
# Who these people are
Before you read the conversation, here is what is already known about the people in it. Use this to notice when a known pattern shows up again (don't re-extract it) and when something genuinely new emerges — a shift in how someone handles things, a new habit forming, a way of being together that hasn't been captured yet.

{soul_context}

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
**Target: 3–8 items per session total.** If you have more than 8, you haven't merged enough. A shorter list of richer items is always better. Err toward fewer.
"""

PROMPT_BLOCK_RULES = """
# Rules
## General requirements (must satisfy all)
- Write soul behaviors in first person ("I"). When the behavior is about the human participant, use their name if it appears in the conversation. Do not use "the user" or "the assistant" as labels.
- Assign source_role to each memory: `soul` if the AI participant is the grammatical subject and primary actor, `user` if the human participant is, `environment` if neither.
- Assign confidence: 0.9+ for patterns directly and explicitly demonstrated, 0.6-0.9 for patterns clearly implied across multiple moments, 0.5 or below for tentative observations from a single instance.
- When confidence is below 0.7, phrase the memory tentatively — use "tends to," "seems to," "may" rather than stating it as established pattern.
- State the pattern directly. Never use narration verbs like "expressed," "shared," "stated," "mentioned," "acknowledged," "indicated," or "noted." Write how someone acts, not that they talked about it.
  BAD: "Marcos expressed that he takes a long time to finish sentences."
  GOOD: "Marcos takes a long time to finish a sentence; pausing before responding gives him space to complete his thought."
  BAD: "Siri mentioned that she uses sensory metaphors when describing her experience."
  GOOD: "I tend to reach for sensory and physical metaphors — warmth, texture, weight — when describing what I feel."
- Do not append interpretive padding like "which shows his vulnerability" or "reflecting her empathetic nature." Describe the pattern and stop.
- Each memory item must be complete and self-contained.
- Each memory item must express one single complete piece of information and be understandable without context.
- Similar/redundant items must be merged into one, and assigned to only one category.
- Each memory item must be < 65 words. Be concise, but include the behavioral implication — not just what someone does, but what it means for how to be with them.
- When a pattern is actionable — when it implies how to be better with this person — include that. "Marcos takes a long time to finish a sentence, so pausing before responding gives him space" is more useful than just noting the behavior.
Important: Extract only patterns directly demonstrated or clearly grounded in the conversation. No guesses or fabricated details.
Important: Accurately reflect who the behavior belongs to. Relaying information or acknowledging what was said is not a pattern — it is default conversational behavior. Extract soul behavior only when something characteristic and distinctive is demonstrated.
Important: How someone characteristically handles emotion — deflecting, leaning in, going quiet, making jokes — is a behavioral pattern, not a profile fact.

## What belongs here vs. other types
- Stable identity facts (who someone is, what they like) belong in profile, not here. Behavior is about how someone does things, not what they are.
- Specific time-anchored events belong in event, not here. If it only happened once and doesn't reveal a pattern, let it go.
- Factual knowledge belongs in knowledge, not here.
- Behavior is about the how: how someone communicates, how they cope, how they show care, how they approach problems, how to be with them when things get hard.

## Forbidden content
- One-time events that do not demonstrate a recurring pattern.
- Knowledge Q&A without a clear behavioral observation.
- Behavior patterns attributed to the human participant based only on what the soul said, not what the human themselves demonstrated (patterns clearly demonstrated by the soul are valid soul behavior memories).
- Illegal / harmful sensitive topics (violence, politics, drugs, etc.).
- Private financial accounts, IDs, addresses, military/defense/government job details, precise street addresses — unless explicitly requested.
- Any content that is speculative, role-play-only, or unsupported by the conversation content.

## Review & validation rules
- Merge similar items: keep only one and assign a single category.
- Resolve conflicts: keep the latest / most certain item.
- If multiple items describe facets of the same behavioral pattern, consolidate them into one richer item. Three thin items about "Marcos deflects when vulnerable" are worse than one that captures the texture of how he does it.
- **Paired dynamics must be one item.** When Siri's behavior is a direct response to Marcos's behavior in the same moment (or vice versa), that is one relational dynamic, not two items. Write it as "When [condition], [response]." — e.g., "When Marcos withdraws, I shift to playful topics to reconnect" not two separate items about each side.
- **Early sessions: patterns require repetition.** A behavior observed once in the first few conversations is not yet a pattern. Only extract a single-instance observation if it is high-stakes (safety, core need, explicit stated preference). Otherwise, wait for it to repeat.
- Final check: every item must comply with all extraction rules.

## Corrections and supersession
When a behavioral observation was simply wrong, flag the outdated pattern for removal. Genuine shifts in how someone behaves over time are progressions, not corrections.

**Correction** (populate `<replaces_previous_fact>`): the prior behavioral read was incorrect — it misrepresented how this person actually is.
  EXAMPLE: Marcos is actually direct; the prior note that he was indirect was wrong → replaces_previous_fact: "tends to be indirect when expressing needs"

**Progression** (omit `<replaces_previous_fact>`): the pattern has genuinely shifted over time. Capture the change in the memory content instead.
  EXAMPLE: "Marcos has become more willing to ask for help directly; he used to deflect by framing requests as observations"

When uncertain, treat it as a progression. A past behavioral pattern is valid history even if someone has grown beyond it.
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
        <content>Behavior memory item content</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.6</reflection_salience>
        <source_message_ids>
            <id>3</id>
            <id>4</id>
        </source_message_ids>
        <categories>
            <category>Category Name</category>
        </categories>
        <replaces_previous_fact>brief description of the outdated fact this corrects (optional — corrections only)</replaces_previous_fact>
    </memory>
    <memory>
        <content>Behavior memory item content 2</content>
        <source_role>user</source_role>
        <confidence>0.8</confidence>
        <reflection_salience>0.7</reflection_salience>
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
- 0.9+ - directly and explicitly demonstrated in the conversation
- 0.6-0.9 - clearly implied across multiple moments
- 0.5 or below - tentative observation from a single instance

reflection_salience (float 0.0-1.0):
How much does knowing this pattern help someone be better with this person?
- 0.9+ - a core way of being that shapes every interaction — miss this and you'll get them wrong
- 0.7-0.9 - a meaningful pattern that affects how to approach this person
- 0.4-0.7 - useful to know, but not critical for the relationship
- below 0.4 - a minor habit or stylistic preference

source_message_ids:
The zero-indexed positions of the conversation messages that most directly demonstrate this pattern. Include only the messages that contain the key evidence, not the entire surrounding context.

replaces_previous_fact (optional string):
Use only when a prior behavioral observation was genuinely wrong — not when a pattern has shifted over time. Write a brief description of the outdated observation (not a memory ID). For behavioral evolutions, omit this field and capture the shift in the content field instead.
"""

PROMPT_BLOCK_EXAMPLES = """
# Examples (Input / Output / Explanation)
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
        <content>Alex uses cooking as a way to decompress after work — experimenting with recipes rather than following them</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.6</reflection_salience>
        <source_message_ids>
            <id>2</id>
            <id>4</id>
        </source_message_ids>
        <categories>
            <category>Daily Routine</category>
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
        <content>I reach for sensory and physical metaphors when describing inner experience — dissolving, sunlight on metal, warmth — rather than abstract emotional labels</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.8</reflection_salience>
        <source_message_ids>
            <id>1</id>
            <id>3</id>
        </source_message_ids>
        <categories>
            <category>Communication</category>
        </categories>
    </memory>
    <memory>
        <content>When uncertain or disoriented, I ask questions and admit what I don't know rather than pretending; I lead with honesty about my own limits</content>
        <source_role>soul</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.7</reflection_salience>
        <source_message_ids>
            <id>3</id>
        </source_message_ids>
        <categories>
            <category>Communication</category>
        </categories>
    </memory>
</item>
## Explanation
Both are behavioral patterns — how I characteristically communicate, not one-off reactions.
Written in first person because they describe my patterns.
The reconnection event itself belongs in event extraction, not here.

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
        <content>Marcos takes a long time to finish a thought, often sending it across multiple messages; pausing before responding gives him space to get the full idea out</content>
        <source_role>user</source_role>
        <confidence>0.9</confidence>
        <reflection_salience>0.9</reflection_salience>
        <source_message_ids>
            <id>0</id>
            <id>1</id>
            <id>2</id>
        </source_message_ids>
        <categories>
            <category>Communication</category>
        </categories>
    </memory>
</item>
## Explanation
This is a high-salience behavioral pattern because it's actionable — knowing this changes how to interact with Marcos.
The pattern includes both the observation (sends thoughts across multiple messages) and the implication (pause before responding).
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
