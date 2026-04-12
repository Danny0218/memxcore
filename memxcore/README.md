# MemXCore

**English** | [中文](README.zh-CN.md)

Persistent memory system for AI coding assistants. Stores and retrieves memories across sessions via MCP (Model Context Protocol), with semantic search powered by ChromaDB + sentence-transformers.

Works with **Claude Code**, **Cursor**, **Gemini CLI**, and any MCP-compatible tool.

---

## Architecture

```
RECENT.md          <- L0: raw append log (WAL). Every remember() lands here.
journal/<date>.md  <- L0.5: permanent raw archive. Compact saves here before clearing RECENT.md.
archive/<cat>.md   <- L1: distilled long-term memory, per category, with YAML front matter
USER.md            <- L2: permanent facts (remember(permanent=True)), never compacted
chroma/            <- RAG vector index (rebuilt from archive/ files, lossy-safe)
index.json         <- Keyword search index (tags + summaries from archive/ front matter)
```

**Search tiers (in order):**
1. **RAG semantic search** (ChromaDB + BM25, RRF fusion) — most accurate, requires `chromadb` + `sentence-transformers`
2. **LLM relevance judgment** — Claude reads all facts and picks relevant ones, requires API key
3. **Keyword fallback** — always available, scans archive files directly

**Write flow:**
1. `remember(text)` -> append to RECENT.md
2. When RECENT.md exceeds token threshold -> raw content archived to `journal/<date>.md` (permanent, lossless)
3. LLM distills into structured facts, preserving original event timestamps (`occurred_at`)
4. Each fact -> written to `archive/<category>.md` + upserted into ChromaDB (dual-write)

---

## Quick Start

```bash
pip install memxcore          # core
pip install 'memxcore[rag,bm25]'   # + hybrid search (recommended)
pip install 'memxcore[all]'   # everything

# Or from source:
git clone https://github.com/Danny0218/memxcore.git
cd memxcore && pip install '.[rag,bm25]'
```

**Check your setup:**
```bash
memxcore doctor
# or: python -m memxcore.cli doctor
```

**Requirements:** Python 3.11+

**LLM API key** (any provider supported via [litellm](https://docs.litellm.ai/)):
```bash
# Pick one:
export ANTHROPIC_API_KEY=sk-ant-...     # Anthropic
export OPENAI_API_KEY=sk-...            # OpenAI
export GEMINI_API_KEY=...               # Google Gemini
# Or use Ollama for fully local/offline: no key needed, just set model in config.yaml
```

Without any API key, compaction falls back to basic mode (truncation instead of LLM distillation).

> **Important: Compaction is not automatic.**
> MemXCore does not detect session boundaries. If compaction is never triggered,
> memories accumulate in RECENT.md and are not distilled into long-term storage.
>
> **Recommended setup** (pick one):
> - Use the provided **Stop hook** (`memxcore.hooks.auto_remember`) — extracts facts after each response and triggers compaction automatically.
> - Call `compact(force=True)` via the MCP tool at the end of each session.
> - Run `memxcore compact` from the command line as a cron job or manual step.
>
> See the [Integration](#integration-claude-code) section for hook setup examples.

**Workspace resolution:**

By default, MemXCore resolves the workspace from the module's install location (suitable for development / editable installs). For pip-installed usage, set `MEMXCORE_WORKSPACE` to your project root:

```bash
export MEMXCORE_WORKSPACE=/path/to/your/project
```

Or in your MCP server config:
```json
{
  "env": { "MEMXCORE_WORKSPACE": "/path/to/your/project" }
}
```

---

## Configuration (`config.yaml`)

Located at `memxcore/config.yaml`. All fields are optional — sensible defaults apply.

```yaml
compaction:
  strategy: llm             # 'llm' = LLM distills to structured facts (via litellm)
                            # 'basic' = truncate first 50 lines to general.md
  threshold_tokens: 2000    # Trigger compaction when RECENT.md exceeds this
  min_entries: 3            # Don't compact if fewer than N entries in RECENT.md
  check_interval: 5         # Check token count every N remember() calls (reduces I/O)
  stale_minutes: 10         # Auto-compact on search() if RECENT.md older than this

llm:
  # model: anthropic/claude-sonnet   # uncomment & change to your provider
  #   Examples: openai/gpt-4o, gemini/gemini-2.5-flash, ollama/llama3
  #   If omitted, auto-detects from available API key
  # api_key_env: OPENAI_API_KEY     # env var name (auto-detects if omitted)
  # base_url:                       # Optional: custom API endpoint (gateway/proxy)

rag:
  embedding_model: paraphrase-multilingual-MiniLM-L12-v2  # multilingual (Chinese↔English cross-lingual)
  top_k: 10                           # Max results from semantic search
  rrf_k: 60                           # Reciprocal Rank Fusion constant

watch: false    # Auto-reindex archive/*.md on file change (off by default)

memory:
  categories:
    - id: user_model
      description: "User identity, preferences, communication style, feedback"
    - id: domain
      description: "Transferable domain/technical knowledge — valid across projects"
    - id: project_state
      description: "Active tasks, in-progress decisions — decays within weeks"
    - id: episodic
      description: "Time-bound events and past decisions — has TTL"
    - id: references
      description: "URLs, doc links, dashboards, ticket trackers, Slack channels"
```

---

## Integration

### One-command setup (recommended)

```bash
memxcore setup                              # interactive: asks where to store memories
memxcore setup --workspace ~/my-project     # non-interactive
memxcore setup --dry-run                    # preview without making changes
```

Setup will ask you to choose a workspace directory for storing memories (default: `~/.memxcore`). This path is written into all tool configurations.

Automatically detects and configures:

| Tool | What it does |
|------|-------------|
| **Claude Code** | Registers MCP server, installs hooks (auto-remember + auto-compact), appends agent rules to `~/.claude/CLAUDE.md` |
| **Cursor** | Writes MCP config to `~/.cursor/mcp.json`, copies rules to `~/.cursor/rules/memxcore.mdc` |
| **Windsurf** | Writes MCP config to `~/.codeium/windsurf/mcp_config.json` |
| **Codex (OpenAI)** | Writes MCP config to `~/.codex/config.toml` |
| **Gemini CLI** | Writes MCP config to `~/.gemini/settings.json` |

After setup, restart your tool and verify the MCP connection.

### Manual setup

If you prefer manual configuration, the MCP server config for any tool is:

```json
{
  "mcpServers": {
    "memxcore": {
      "command": "/path/to/your/.venv/bin/python",
      "args": ["-m", "memxcore.mcp_server"],
      "env": { "MEMXCORE_WORKSPACE": "/path/to/your/workspace" }
    }
  }
}
```

---

## MCP Tools Reference

| Tool | Args | Description |
|------|------|-------------|
| `remember` | `text`, `category?`, `permanent?`, `tenant_id?` | Store a memory |
| `search` | `query`, `max_results?`, `after?`, `before?`, `tenant_id?` | Retrieve memories (semantic + keyword), with optional time range |
| `compact` | `force?`, `tenant_id?` | Distill RECENT.md into categorized archives (also archives to journal) |
| `flush` | `tenant_id?` | Move RECENT.md to journal (lossless, no LLM) |
| `reindex` | `category?`, `tenant_id?` | Rebuild RAG + BM25 index after manual edits |
| `purge_journal` | `keep_days?`, `tenant_id?` | Delete journal files older than N days (default 30) |
| `set_config` | `key`, `value`, `tenant_id?` | Set a config value (dot notation, e.g. `llm.model`) |

**Categories for `remember`:**
`user_model` / `domain` / `project_state` / `episodic` / `references`

**Permanent memories:**
Use `permanent=True` to write directly to USER.md (L2). These are never compacted and always have the highest priority in search. Use for stable facts like core user identity or long-term architectural decisions.

---

## Multi-Tenant

All MCP tools and CLI commands accept an optional `tenant_id`. Each tenant gets isolated storage under `tenants/<tenant_id>/storage/`. Omit `tenant_id` for single-user mode.

```bash
# CLI
memxcore --tenant alice search "preferences"

# MCP
search("preferences", tenant_id="alice")
```

Tenants can also have their own `config.yaml` override at `tenants/<tenant_id>/config.yaml`.

---

## CLI Reference

```bash
memxcore setup                        # auto-detect tools, configure everything
memxcore setup --dry-run              # preview without changes
memxcore doctor                       # check system readiness
memxcore config show                  # show current config
memxcore config set llm.model openai/gpt-4o   # change LLM provider
memxcore config path                  # show config file location
memxcore reindex                      # rebuild RAG + BM25 index
memxcore compact                      # force distillation (also archives to journal)
memxcore flush                        # move RECENT.md to journal (no LLM)
memxcore search "query"               # search memories
memxcore search --after 2026-04-08 "query"   # search with time range
memxcore timeline                     # show last 7 days of memory activity
memxcore timeline --days 3            # show last 3 days
memxcore purge-journal --keep-days 30 # delete journal files older than 30 days
memxcore benchmark                    # search precision benchmark
memxcore mine <path>                  # import conversations
memxcore --tenant alice doctor        # multi-tenant
```

All commands also work via `python -m memxcore.cli <command>`.

---

## HTTP API Server (optional)

For development/testing. In production, use the MCP server.

```bash
pip install 'memxcore[server]'
python -m memxcore.server --host 127.0.0.1 --port 8000
```

Endpoints:
```
POST /remember       { "text": "...", "level": null }
GET  /search?query=  returns List[MemoryResult]
POST /compact        ?force=false
POST /rebuild-rag    rebuild ChromaDB from archive files
```

---

## Storage Structure

```
memxcore/
+-- storage/
    +-- RECENT.md              Raw memory log. Cleared after each compaction.
    +-- USER.md                Permanent facts (permanent=True). Never compacted.
    +-- index.json             Keyword search index. Auto-rebuilt after compaction.
    +-- chroma/                ChromaDB vector index. Rebuildable from archive/.
    +-- journal/               Permanent raw archive (lossless, never auto-deleted).
        +-- 2026-04-10.md      Daily journal — original RECENT.md content preserved.
        +-- 2026-04-11.md
    +-- archive/
        +-- user_model.md      User identity, preferences, feedback
        +-- domain.md          Domain/technical knowledge
        +-- project_state.md   Active tasks, decisions (stale after weeks)
        +-- episodic.md        Past events and decisions
        +-- references.md      URLs, doc links, external pointers
        +-- general.md         Fallback when LLM distillation unavailable
```

**Archive file format:**
```markdown
---
topic: project_state
tags: [api, testing]
last_distilled: 2026-04-07T14:23:01
confidence_level: 1
---

## [2026-04-07T12:30:00]

Unit test initiative complete: 234 tests passing across 9 packages.
```

The `## [timestamp]` is the original event time (when `remember()` was called), not the compaction time. `last_distilled` in the front matter tracks when compaction ran.

Archive files are plain Markdown — edit them directly. After editing, run `memxcore reindex <category>` to update the RAG + BM25 index.

**Journal file format:**
```markdown
# === Archived at 2026-04-10T14:23:01.123456 ===

# [2026-04-10T12:30:00] [category:project_state] Memory
Unit test initiative complete: 234 tests passing.

# [2026-04-10T13:15:00] Memory
User prefers concise responses.
```

Journal files preserve the exact RECENT.md content with original timestamps. Use `memxcore timeline` to browse them, or `memxcore purge-journal --keep-days N` to clean up old ones.

---

## Security Considerations

**Storage permissions:** Memory files (`storage/`, `knowledge.db`) are created with default OS permissions (typically 0644). On shared/multi-user systems, restrict access manually:
```bash
chmod 700 memxcore/storage/
```

**HTTP server:** The optional HTTP API (`memxcore.server`) has **no authentication** and is intended for local development only. Do **not** expose it to a network. It binds to `127.0.0.1` by default — never change this to `0.0.0.0` in production.

**`memxcore mine` command:** Reads any file the user specifies. This is by design (same as `cat`), but be careful not to pipe untrusted paths into it if used in scripts.

**LLM prompt injection:** Like all LLM-based systems, stored memory content is passed to the LLM during distillation and search. Malicious content in memories could influence LLM outputs. Category names from LLM responses are sanitized to prevent path traversal, but fabricated memory content is a fundamental limitation of LLM-based extraction.

**Dependency versions:** Core dependencies use `>=` pins without upper bounds. For production deployments, use a lock file (`pip freeze > requirements.lock`) to pin exact versions.

---

## Troubleshooting

Run `memxcore doctor` first — it checks everything and tells you what to fix.

**RAG not working**
```bash
pip install 'memxcore[rag,bm25]'
# First run downloads the embedding model (~80MB) + torch (~500MB)
```

**LLM distillation falling back to basic**
```bash
memxcore doctor   # check which API key is detected
# Any one of these will work (via litellm):
#   ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or Ollama (no key needed)
# Without API key, memories go to archive/general.md (uncategorized)
```

**Rebuild RAG index**
```bash
memxcore reindex
```

---

## License

MIT. See [LICENSE](LICENSE).
