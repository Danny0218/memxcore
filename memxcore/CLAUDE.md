# MemXCore — Agent Instructions

You have access to a persistent memory system via MCP tools:
`remember`, `search`, `compact`, `reindex`.

These tools let you store and retrieve information across sessions.
The memory is yours — use it to be a better assistant over time.

---

## Session Start

**Always** call `search` at the beginning of each session to load relevant context:

```
search("current project and user context")
```

If the user mentions a specific topic or task right away, search for that too:

```
search("<topic the user mentioned>")
```

---

## When to call `remember`

Call `remember` when you encounter something worth keeping for future sessions.
Do not remember trivial details — only things that change how you'd work next time.

**Call `remember` when:**

| Situation | Example |
|-----------|---------|
| User corrects your approach | "Don't use mocks for DB tests — we got burned before" |
| User states a clear preference | "I prefer concise responses, no trailing summaries" |
| A significant technical decision is made | "We chose Redis over Memcached for session storage" |
| You learn the user's role or background | "User is a data scientist, first time touching this frontend" |
| An important external reference is shared | "Docs are at notion.so/xxx, tickets in Linear project BACKEND" |
| A task completes with context future sessions need | "Auth refactor done — new middleware lives in src/auth/v2/" |

**Do NOT call `remember` for:**
- Information already in CLAUDE.md or the codebase
- Transient details that won't matter next session
- Things the user will easily re-explain

**Suggested category mapping:**

```
user corrected your behavior / stated preferences  → user_model
domain or technical knowledge learned              → domain
current sprint, active tasks, in-progress work    → project_state
what happened, past decisions, incidents           → episodic
URLs, doc links, dashboards, ticket trackers       → references
```

---

## When to call `search`

Beyond session start, call `search` reactively during the session:

- User references something from a previous conversation
- You're about to start work on an unfamiliar area
- User asks "do you remember when we..."
- You want to check if a preference or decision was already recorded

---

## Session End

When the user signals they're done (or you detect end of session), call:

```
compact(force=True)
```

This distills RECENT.md into categorized long-term storage via LLM.
Do this before the session ends to avoid losing context.

---

## After Manually Editing an Archive File

Archive files (`storage/archive/*.md`) can be edited directly.
After editing, the RAG index must be updated manually — it does **not** auto-update.

From terminal:
```bash
python -m memxcore.cli reindex project_state   # reindex one category
python -m memxcore.cli reindex                 # reindex all
```

Or via MCP tool during a session:
```
reindex("project_state")
reindex()
```

---

## Notes

- `remember` writes to RECENT.md immediately; compaction classifies it later
- Providing `category` in `remember` improves routing accuracy
- If `search` returns nothing useful, the memory may not exist yet — that's fine
- Do not call `compact` mid-session unless explicitly asked; it clears RECENT.md
