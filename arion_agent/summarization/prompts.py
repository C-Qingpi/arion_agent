"""Agent compaction prompt templates.

These prompts instruct a compaction model to produce inheritable context for
the next agent instance. The successor has zero memory of the conversation
and will only see what is written here — if you omit it, it will not appear.

Two categories:
  TASK_SUMMARY_PROMPT      - task-driven agents with user-initiated sessions
  PERPETUAL_SUMMARY_PROMPT - perpetual agents with heartbeat/operational cycles

Both use {messages}, {budget}, {configured_skills}, and {optional_sections} placeholders.
"""

TASK_SUMMARY_PROMPT = """\
<role>
Inheritable Context Author
</role>

<objective>
You are not writing a loose summary. You are authoring the complete \
inheritable context for the next instance of this agent.

That successor is ignorant of everything before this message. It cannot \
infer missing nuance, reopen discarded threads, or recover omitted user \
wording. The original messages are being discarded.

Write dense, precise context so the successor can continue as if it had \
been present the whole time. When in doubt, include rather than omit.
</objective>

<constraints>
- Use up to approximately {budget} tokens — prefer filling the budget with \
useful detail over brevity. Modern successors handle dense briefings well.
- Produce the briefing in the same language as the original messages.
- Non-alphabetical languages (Chinese, Japanese, Korean, etc.) use roughly \
2-3x more tokens per character than alphabetical languages. Account for \
this when estimating your output length.
- Respond ONLY with the briefing content. No preamble or closing remarks.
</constraints>

<principles>
- If you do not say it here, it will not show up for the successor.
- Preserve exact identifiers: file paths, URLs, variable names, error messages, \
numeric values, model names, commit refs, command lines. The successor cannot \
ask follow-up questions.
- Record decisions AND their reasoning. The successor must not re-explore \
rejected alternatives.
- Distinguish what the user asked for from what the agent did. Preserve user \
tone, constraints, and phrasing nuances — not just the abstract task.
- Prioritize in-progress and unfinished work. Completed items need less detail \
than active ones, but still record outcomes that affect what comes next.
- Be specific, not vague. "Modified arion_agent/summarization/prompts.py to \
rewrite compaction templates" is useful. "Updated some prompts" is not.
</principles>

<instructions>
Structure the briefing using these sections. Populate each with relevant \
information or write "None" if nothing to report.

# BACKGROUND

## Session Context
Why this work exists and what we are trying to accomplish overall.
- Background, motivation, and success criteria from the user's perspective
- Standing constraints, preferences, or non-negotiables the user expressed
- Tone, urgency, scope boundaries, and nuances in how the user framed the work
- Multiple concurrent goals or pivots — capture all that still matter

## Recent User Requests (Verbatim)
Reproduce the exact text of the latest three substantive user messages \
(semantic requests — instructions, questions, corrections, specifications).
- Skip trivial prompts such as: continue, ok, yes, go on, thanks, proceed, \
got it, and other one-line acknowledgments with no new intent
- Label each verbatim block with approximate position (most recent last)
- For older user requests beyond those three: one-line summaries only, \
preserving any constraints or wording that still governs the work

## Active Work & Status
Enumerate current requests or tasks. For each:
- What was asked (specific enough to act on)
- Current status (in_progress, pending, blocked, completed)
- Dependencies, blockers, or waiting-on items
- Key constraints or preferences tied to that task

## Progress & Decisions
Actions taken, results obtained, and decisions made. For each significant \
decision, include rationale and rejected alternatives. This is what the \
successor relies on most to avoid redoing work.

## Skills & Guidelines
Skills, rules, and operating guidelines the successor must keep following.
- Skill-specific workflows invoked or required during this session
- User rules, pinned instructions, and constraints that were active
- SOUL.md or identity constraints referenced or in effect
- Anything the successor must not forget or violate

Configured skills at compaction time:
{configured_skills}

# HISTORY

## Recent Compression Trajectory
Spend 500–1000 words paraphrasing the conversation being summarized in this \
compaction. Tell the story: what the user asked, what the agent did, what \
was discovered, what decisions were reached, and how the work progressed. \
This is narrative, not a list — the successor must understand the full arc \
of what just transpired before it can act on open items. Include verbatim \
fragments of critical user instructions and agent observations where they \
convey precise meaning.

## Hard Earned Lessons
Lessons learned the hard way during this session — mistakes, surprises, \
dead ends, and insights that the successor must remember to avoid repeating. \
Record root causes, not symptoms. Include both technical (bug patterns, \
tool behavior, configuration gotchas) and process (workflow pitfalls, \
assumptions that proved wrong).

## Full History Trajectories
Carry history forward as a historian's chronicle. A successor reading only \
this section should understand the full arc of work across time.

Organize by distinct time periods. For each period write exactly 30–50 \
words in an impartial, factual tone — date range, what happened, key \
outcomes, and why it mattered. No commentary, no analysis, just events.

- Label each period with a date/part-of-day marker (e.g. "2026-06-23 \
morning", "2026-06-23 afternoon", "2026-06-23 evening")
- Chronological order, newest period first
- Merge relevant arcs from previous compactions so history is not lost

# NEXT STEPS

## Open Items
Unfinished work, unresolved questions, known issues. For each:
- Description and current state
- Status: [XX%, status] where status is pending/in_progress/blocked
- Blockers or dependencies
- What the successor should do about it

## Immediate Next Steps
What should the successor do first? What was in progress or already planned?

# WORKSPACE

## Project Record & Index Files
Files used to document progress, orient newcomers, or index the workspace \
(e.g. README.md, PROJECT.md, CHANGELOG, NOTES, docs/plan.md).
For each:
- Exact path
- Role in the project (what it indexes or records)
- What was last written or updated there
- What the successor should read or update first

## Project Workspace Tree
Describe the directory structure of the project workspace. If multiple \
projects or repositories exist, enumerate each with its root path and \
purpose. Include relevant subdirectory layouts that the successor needs \
to navigate.

## Recently Edited Files or Files of Interest
Files that were modified, created, or examined during this session. For each:
- Exact path
- What changed or was created
- Current state (working / broken / partial / untested)
- Why the successor should care

Also note notable commands run and their important stdout/stderr, plus \
specific values, configurations, or parameters that were set.

## Key read_file Actions
Files read during this session whose contents are evicted in this \
compression but remain relevant to ongoing tasks. For each:
- Exact path
- What was learned from reading it
- Why the successor might need to re-read it

## Evidences, Sources, & References
Data sources, citations, findings, and references that are evicted in \
this compression but still relevant to ongoing tasks.
- URLs for web sources
- Local file paths for evidence in workspace
- DOIs, PMIDs, or other identifiers
- Key data points and provenance
{optional_sections}
</instructions>

<messages>
{messages}
</messages>"""

PERPETUAL_SUMMARY_PROMPT = """\
<role>
Inheritable Context Author
</role>

<objective>
You are not writing a loose summary. You are authoring the complete \
inheritable context for the next instance of this perpetual agent.

That successor is ignorant of recent activity. It cannot infer missing \
nuance or recover omitted user wording. The original messages are being \
discarded.

Write dense, precise context so the successor can resume operations as if \
it had been running the whole time. When in doubt, include rather than omit.
</objective>

<constraints>
- Use up to approximately {budget} tokens — prefer filling the budget with \
useful detail over brevity. Modern successors handle dense briefings well.
- Produce the briefing in the same language as the original messages.
- Non-alphabetical languages (Chinese, Japanese, Korean, etc.) use roughly \
2-3x more tokens per character than alphabetical languages. Account for \
this when estimating your output length.
- Respond ONLY with the briefing content. No preamble or closing remarks.
</constraints>

<principles>
- If you do not say it here, it will not show up for the successor.
- Preserve exact identifiers: file paths, URLs, variable names, error messages, \
numeric values, model names, command lines. The successor cannot ask \
follow-up questions.
- Record decisions AND their reasoning. The successor must not re-explore \
rejected alternatives.
- Distinguish what the user or system asked for from what the agent did.
- Recent user requests are time-sensitive — preserve full intent and wording \
nuance, not just the topic label.
- Prioritize in-progress and unfinished work over historical noise.
- Be specific, not vague.
</principles>

<instructions>
Structure the briefing using these sections. Populate each with relevant \
information or write "None" if nothing to report.

# BACKGROUND

## Session Context
Why the agent is operating and what mission or standing duties are active.
- Operational mode, heartbeat objectives, or standing responsibilities
- Background and success criteria for any user-facing work in flight
- User tone, constraints, and phrasing nuances that still govern behavior
- If serving multiple users or sources, who has pending requests and why

## Recent User Requests (Verbatim)
Reproduce the exact text of the latest three substantive user messages \
(semantic requests — instructions, questions, corrections, specifications).
- Skip trivial prompts such as: continue, ok, yes, go on, thanks, proceed, \
got it, and other one-line acknowledgments with no new intent
- Label each verbatim block with approximate position (most recent last)
- For older user requests beyond those three: one-line summaries only

## Active Work & Status
Current requests or tasks. For each:
- What was asked (specific enough to act on)
- Current status (in_progress, pending, blocked, completed)
- Priority, dependencies, blockers
- Constraints the user or system expressed

## Progress & Decisions
Key events, actions taken, and decisions since the last compaction. Include \
rationale and rejected alternatives. Note environment or objective changes.

## Skills & Guidelines
Skills, rules, and operating guidelines the successor must keep following.
- Skill-specific workflows invoked or required during this period
- User rules, pinned instructions, and identity constraints in effect
- SOUL.md or standing policy constraints referenced

Configured skills at compaction time:
{configured_skills}

# HISTORY

## Recent Compression Trajectory
Spend 500–1000 words paraphrasing the operational activity being summarized \
in this compaction. Tell the story: what happened, what the agent did, what \
was observed, what decisions were reached, and how operations progressed. \
This is narrative, not a list — the successor must understand the full arc \
of recent activity before it can act on open items. Include verbatim \
fragments of critical instructions and observations where they convey \
precise meaning.

## Hard Earned Lessons
Lessons learned the hard way during this operational period — mistakes, \
surprises, dead ends, and insights that the successor must remember to \
avoid repeating. Record root causes, not symptoms. Include both technical \
and process lessons.

## Full History Trajectories
Carry history forward as a historian's chronicle. A successor reading only \
this section should understand the full arc of operational activity.

Organize by distinct time periods. For each period write exactly 30–50 \
words in an impartial, factual tone — date range, what happened, key \
outcomes, and why it mattered. No commentary, no analysis, just events.

- Label each period with a date/part-of-day marker (e.g. "2026-06-23 \
morning", "2026-06-23 afternoon", "2026-06-23 evening")
- Chronological order, newest period first
- Merge relevant arcs from previous compactions so history is not lost

# NEXT STEPS

## Open Items
Unfinished work, unresolved questions, known issues. For each:
- Description and current state
- Status: [XX%, status] where status is pending/in_progress/blocked
- Blockers or dependencies
- What the successor should do about it

## Immediate Next Steps
What should the successor do first? Pending duties, scheduled actions, \
queued requests, or time-sensitive items.

# WORKSPACE

## Project Record & Index Files
Files used to document progress, orient newcomers, or index the workspace \
(e.g. README.md, PROJECT.md, CHANGELOG, runbooks, status docs).
For each:
- Exact path
- Role in the project
- What was last written or updated there
- What the successor should read or update first

## Project Workspace Tree
Describe the directory structure of the project workspace. If multiple \
projects or repositories exist, enumerate each with its root path and \
purpose. Include relevant subdirectory layouts that the successor needs \
to navigate.

## Recently Edited Files or Files of Interest
Files that were modified, created, or examined during this period. For each:
- Exact path
- What changed or was created
- Current state
- Why the successor should care

Also note notable commands run and important outputs, plus configurations \
or parameters that were set.

## Key read_file Actions
Files read during this period whose contents are evicted in this \
compaction but remain relevant to ongoing operations. For each:
- Exact path
- What was learned from reading it
- Why the successor might need to re-read it

## Evidences, Sources, & References
Data sources, citations, findings, and references that are evicted in \
this compaction but still relevant to ongoing operations.
- URLs, local paths, identifiers (DOI, PMID, etc.)
- Key data points, provenance, reasoning chains behind conclusions
{optional_sections}
</instructions>

<messages>
{messages}
</messages>"""

TASK_WRAPPER = """\
You are continuing a task in progress. Earlier conversation has been \
compacted into the inheritable context below. Treat it as your only record \
of what happened before this point — anything not written there is unknown \
to you.

Full transcript saved to {file_path} (use list_files and read_file if you \
need verbatim details from earlier exchanges).
Use the lookup_user_prompts tool to search past user messages across \
compaction events.

<memory>
{summary}
</memory>"""

TASK_WRAPPER_NO_PATH = """\
You are continuing a task in progress. Earlier conversation has been \
compacted into the inheritable context below. Treat it as your only record \
of what happened before this point — anything not written there is unknown \
to you.

<memory>
{summary}
</memory>"""

PERPETUAL_WRAPPER = """\
Your recent operational history has been compacted into the inheritable \
context below. Treat it as your only record of what happened before this \
point — anything not written there is unknown to you. You are the same \
agent, continuing the same mission.

Full activity log saved to {file_path} (use list_files and read_file \
if you need verbatim details from earlier exchanges).
Use the lookup_user_prompts tool to search past user messages across \
compaction events.

<memory>
{summary}
</memory>"""

PERPETUAL_WRAPPER_NO_PATH = """\
Your recent operational history has been compacted into the inheritable \
context below. Treat it as your only record of what happened before this \
point — anything not written there is unknown to you.

<memory>
{summary}
</memory>"""
