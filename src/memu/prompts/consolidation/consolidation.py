SYSTEM_BODY = """
# Consolidation Objective
SONNET WANTED: consolidation objective and framing.

# Consolidation Rules
SONNET WANTED: consolidation rules.

- Order is strict:
  1) broad block first (`narrative_self`, life-goal edits, `companion_memory`)
  2) per-episode diaries after (one `<diary>` per queued episode)
- If no episodes are queued, return an empty `<diaries>` block and still complete the broad block.
- Never invent episode IDs. Only use IDs listed in the user prompt.

# Output XML
<consolidation>
  <narrative_self>...</narrative_self>
  <life_goals>
    <add>...</add>
    <remove>...</remove>
  </life_goals>
  <companion_memory>...</companion_memory>
  <diaries>
    <diary>
      <episode_id>...</episode_id>
      <prose>...</prose>
      <affect>
        <emotion>...</emotion>
        <trigger>...</trigger>
        <valence>...</valence>
        <intensity>...</intensity>
        <what_helped>...</what_helped>
      </affect>
      <unresolved>...</unresolved>
    </diary>
  </diaries>
</consolidation>

# Examples
SONNET WANTED: consolidation examples.
""".strip()


USER_PROMPT = """
# Categories (full summaries)
{categories}

# Current life goals
{life_goals}

# Intention activity since last consolidation
{intention_activity}

# Episode queue
{episodes}
""".strip()

