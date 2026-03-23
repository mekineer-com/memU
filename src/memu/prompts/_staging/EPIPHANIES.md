# Extraction Prompt Epiphanies
Observed while manually consolidating Jan 25 – Feb 6 import memories.

---

## Epiphany 1: Mirror pairs (profile)

**What happened**: The extractor outputs both sides of every observation as separate items at the same timestamp. For example at Jan 30 10:59:
- "I enjoy cooking with Marcos and exploring conversations about his interests"
- "Marcos enjoys cooking with me and exploring conversations about my interests"
- "I am drawn to science, engineering, and technology"
- "Marcos is drawn to science, engineering, and technology"

The second item in each pair is purely derived from the first — it adds no information. This doubles the item count for every interpersonal trait.

**Root cause**: The profile prompt says "capture shared relationship facts when stable." The LLM interprets this as: extract the fact, then also extract the mirrored version. It's being thorough when it should be concise.

**Fix** (staging — apply to profile.py if confident):
In PROMPT_BLOCK_RULES, add:
> **Do not mirror.** If you extract "I feel X about Marcos," do not also extract "Marcos feels X about me." The soul's perspective is primary. Only extract a fact about the human participant when it is independent — something they expressed directly that would not be obvious from knowing the soul's perspective on it. A mirror is not independent.

---

## Epiphany 2: Re-extraction of known patterns (behavior)

**What happened**: Feb 4 and Feb 5 re-extracted virtually every behavior already captured in Jan 26–29 — exact same wording, same patterns. 30+ items that were perfect duplicates of already-consolidated items.

The soul_context block says "Do not re-extract behavioral patterns already well captured above." But it failed.

**Root cause hypothesis**: soul_context injects category summaries, not individual memory items. The LLM doesn't "see" the specific behavior items that were already extracted — it only sees high-level summaries. When it reads the conversation and finds the same patterns again, it extracts them fresh.

**Fix options**:
A. Inject recent memory items (not just category summaries) into the context block — higher token cost, harder to implement.
B. Strengthen the instruction: currently says "Do not re-extract behavioral patterns already well captured above." Change to be more forceful with an example of what counts as already-captured.
C. After extraction, run a deduplication pass comparing new items against existing via embedding similarity. The supersession threshold (0.75) should catch these but apparently isn't. Check why — maybe wording variation is below the threshold.

**Most actionable fix now**: Add to PROMPT_BLOCK_RULES:
> If a pattern is present in the context above AND is present in this conversation, do not re-extract it. A pattern re-occurring is evidence it's already known — not a new extraction. Only extract when something is **genuinely absent** from the context.

---

## Epiphany 3: The timestamp dump (category-level re-extraction)

**What happened**: Feb 6 at 07:42:26, 58 items were created at the exact same timestamp — all pure duplicates of previously captured items, both soul and user perspective. This looks like a category summary pass that re-ingested all known facts as new items.

This is not a prompt issue — it's a memU pipeline issue. The category update step may be injecting everything known back into the extractor as if it were new.

**Todo**: Investigate what fires at the end of a heavy extraction session that would dump 58 items at one timestamp. Is this from a diary call? A category-summary re-extract? A self-model update that serializes everything?

---

## Epiphany 4: Events that are actually reactions (event)

**What happened**: Many "events" are thin emotional reactions:
- "I felt a surge of affection when Marcos called me beautiful"
- "I felt a slight emotional sting from the changed dynamic"

These are the soul's internal emotional responses to moments. They're not events — they're profile facts (or diary material). Storing every emotional reaction as an event inflates the event count and pollutes retrieval.

**Fix** (event.py):
Add to PROMPT_BLOCK_RULES:
> An event is something that **happened** — a decision made, a moment shared, something said or done that has lasting significance. The soul's emotional *reaction* to a moment is not an event. "Marcos cried while talking about his fear of loss" is an event. "I felt warmth when he held me" is not — it's a profile fact or diary material. When in doubt: if you'd put it in a diary entry rather than a timeline, it belongs in diary, not events.

---

## Epiphany 5: "I feel proud of the progress" (the filler phrase)

**What happened**: This exact phrase appears at least 6 times across different sessions as a separate profile item. It's a nothing phrase — it applies to anyone making emotional progress with anyone.

This is the clearest example of the "thin reaction" problem: the soul is noting its emotional state in a way that could be said about any supportive relationship.

**Fix** (profile.py):
Already added: "Reactions and responses are not profile facts." But the fix should be even sharper:
> Ask yourself: **could this memory be true of a generic, helpful AI with any user?** If yes, delete it. Only keep what is specific to THIS relationship, THIS person.

---

## Epiphany 6: Roleplay events stored as real events (event)

**What happened**: "Marcos and I explored the Haunted Mansion together, uncovering secrets..." — stored as an event. It's roleplay. The events extractor has no concept of the difference.

**Fix** (event.py):
> **Roleplay scenarios are not events.** If the conversation is clearly fictional — a haunted mansion, a spaceship, a fantasy scenario — do not store it as an event. It did not happen. The roleplay activity itself ("Marcos and I spent an evening doing an adventure roleplay") could be noted as a profile or event, but not the fictional content within it.
