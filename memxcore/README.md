# MemXCore

[![PyPI version](https://img.shields.io/pypi/v/memxcore)](https://pypi.org/project/memxcore/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/Danny0218/memxcore/blob/main/memxcore/LICENSE)

**The memory layer that learns how you work.**

**English** | [中文](https://github.com/Danny0218/memxcore/blob/main/memxcore/README.zh-CN.md)

Your AI remembers without being told. It picks up your preferences, habits, and past decisions from natural conversation — no manual setup needed. Works across Claude Code, Cursor, Windsurf, Gemini CLI, and any MCP-compatible tool.

---

## Why MemXCore?

Every time you start a new AI session, you lose context. You re-explain your coding style, repeat project decisions, and watch your AI make the same mistakes you corrected yesterday.

MemXCore fixes this. It sits between you and your AI tools as a shared memory layer — automatically extracting what matters from your conversations and making it available in every future session, across every tool.

---

## Features

- 🧠 **Your AI learns you automatically** — LLM distillation extracts facts, preferences, and behavioral patterns from conversations. Recurring habits get promoted to permanent memory. You never need to manually teach it.

- 📖 **Fully transparent Markdown** — Every memory is a readable `.md` file. Open them in any editor, `git push` them for version control, or `grep` through them. No black-box vector databases you can't inspect.

- 🔗 **One memory, every tool** — Use Cursor in the morning and Claude Code in the afternoon. Your memories follow you. One `memxcore setup` configures all detected tools automatically.

- 🔍 **Three-tier search** — Semantic search (ChromaDB + BM25 fusion), LLM relevance judgment, and keyword fallback. Entity-aware tag boosting and a knowledge graph for relationship queries.

- 🕐 **Timeline you can browse** — Raw conversations are permanently archived in daily journal files. Run `memxcore timeline --days 3` to see what happened this week. Original timestamps are preserved through distillation.

- ⚡ **One command to start** — `pip install memxcore && memxcore setup`. Auto-detects Claude Code, Cursor, Windsurf, Codex, and Gemini CLI. Hooks install automatically.

---

## Quick Start

```bash
pip install 'memxcore[rag,bm25]'   # recommended: includes hybrid search
memxcore setup                      # auto-detect and configure all your tools
```

Or from source:
```bash
git clone https://github.com/Danny0218/memxcore.git
cd memxcore && pip install '.[rag,bm25]'
memxcore setup
```

Verify your setup:
```bash
memxcore doctor
```

**Requirements:** Python 3.11+

**LLM API key** (any provider via [litellm](https://docs.litellm.ai/)):
```bash
export ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY, GEMINI_API_KEY
# Ollama works too — no key needed, just set model in config
```

Without an API key, distillation falls back to basic mode (truncation, no categorization).

---

## How It Works

```
You talk to your AI
        |
    remember()          Raw conversation → RECENT.md
        |
    compact()           LLM extracts facts, habits, decisions
        |
   +----+----+
   |         |
journal/   archive/     Lossless archive + categorized knowledge
   |         |
   |     search()       Semantic + keyword + knowledge graph
   |         |
   +----+----+
        |
  Your AI knows you
```

Memories flow through three tiers:
- **RECENT.md** — Raw write buffer. Every `remember()` lands here.
- **archive/** — Distilled facts, categorized and tagged. Searchable.
- **USER.md** — Permanent memory. Facts that prove themselves over time get promoted here automatically.

Before any distillation, raw content is saved to **journal/** — a permanent, lossless daily log that never gets deleted. You can always go back.

<details>
<summary><strong>Technical architecture</strong></summary>

```
RECENT.md          <- L0: append-only write buffer (WAL)
journal/<date>.md  <- L0.5: permanent raw archive, lossless
archive/<cat>.md   <- L1: LLM-distilled facts with YAML front matter
USER.md            <- L2: permanent facts, auto-promoted from L1
chroma/            <- RAG vector index (rebuildable)
knowledge.db       <- SQLite entity-relationship triples
index.json         <- Keyword search index
```

**Write path:**
1. `remember(text)` → append to RECENT.md with timestamp
2. Token threshold reached → raw content archived to `journal/<date>.md`
3. LLM distills into structured facts, preserving original timestamps (`occurred_at`)
4. Each fact → dual-write to `archive/<category>.md` + ChromaDB
5. Knowledge graph triples extracted (e.g., `Alice → leads → frontend-team`)

**Search path (3 tiers):**
1. **Hybrid** — RAG semantic (ChromaDB) + BM25 keyword, fused via Reciprocal Rank Fusion
2. **LLM judgment** — Claude/GPT reads all facts and picks relevant ones
3. **Keyword fallback** — direct file scan, always available

**Lifecycle:**
- `project_state` entries older than 90 days → auto-demoted to `episodic`
- `user_model` facts reinforced 3+ times → auto-promoted to USER.md (L2)
- Duplicate facts (word overlap > 65%) → auto-pruned on write

</details>

---

## Benchmarks

Evaluated on [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (500 questions, ~50 sessions per question). Each question plants facts across dozens of conversation sessions, then asks a question that requires retrieving the right session.

### Search Engine Ceiling (Strategy B)

Raw session text is written directly to archive and indexed — no LLM distillation. This measures the **maximum recall** the search engine can achieve when given perfect input.

| Mode | R@1 | R@3 | R@5 | R@10 |
|------|----:|----:|----:|-----:|
| **hybrid (RAG + BM25)** | 93.6% | 98.8% | **99.2%** | 100.0% |
| RAG only | 89.8% | 97.0% | 98.8% | 100.0% |
| BM25 only | 92.2% | 98.6% | 99.4% | 99.8% |

4 misses out of 500 — all edge cases with indirect phrasing or numerical reasoning.

### End-to-End Pipeline (Strategy A)

Full `remember()` → `compact()` (LLM distillation) → `search()` pipeline. Measures **real-world recall** including information loss from distillation.

> **Note:** Strategy A requires an LLM API key and makes ~5,000 API calls for the full 500-question run (~40M input tokens, ~20M output tokens). With Haiku this takes ~7 hours and costs ~$100. We plan to publish results once the run completes. You can run it yourself — see instructions below.

### What the strategies test

```
Strategy B: session text ──→ archive ──→ RAG/BM25 index ──→ search
                              (skip distillation)
                              Measures: search engine ceiling

Strategy A: session text ──→ remember() ──→ compact() ──→ archive ──→ search
                              (full pipeline)
                              Measures: end-to-end recall with distillation loss
```

The gap between B and A reveals how much information is lost during LLM distillation — the key metric for tuning the compaction prompt.

### Run benchmarks yourself

```bash
cd memxcore
python -m venv .bench-venv && source .bench-venv/bin/activate
pip install -r requirements.txt huggingface-hub chromadb sentence-transformers rank-bm25 pyyaml

# Strategy B — no API key needed, ~11 min
python -m benchmarks.longmemeval --strategy B

# Strategy A — requires LLM API key, ~7 hours with Haiku
export ANTHROPIC_API_KEY=sk-ant-...
python -m benchmarks.longmemeval --strategy A --limit 50   # quick sample (~40 min)
python -m benchmarks.longmemeval --strategy A               # full 500 questions

# Save results to JSON
python -m benchmarks.longmemeval --strategy B --output results_B.json
```

---

## Comparison

| Feature | **MemXCore** | Mem0 | Letta | memsearch | mempalace |
|---------|:---:|:---:|:---:|:---:|:---:|
| MCP protocol | Yes | No | No | Yes | Yes |
| Cross-tool (Claude/Cursor/Gemini) | Yes | No | No | Partial | Partial |
| Transparent Markdown storage | Yes | No | No | No | Yes |
| LLM distillation | Yes | Yes | Yes | No | No |
| Behavioral pattern detection | Yes | No | No | No | No |
| Hybrid search (RAG + BM25) | Yes | Vector only | Vector only | Vector + BM25 | Vector only |
| Knowledge graph | Yes | Yes (Neo4j) | No | No | No |
| Journal / timeline | Yes | No | No | No | No |
| Auto-promote to permanent | Yes | No | No | No | No |
| Multi-tenant | Yes | Yes | Yes | No | No |
| Fully local / self-hosted | Yes | Cloud or self | Cloud or self | Yes | Yes |
| One-command setup | Yes | No | No | No | No |

**When to choose MemXCore:** You want your AI tools to learn your preferences automatically, you care about inspecting and editing your memories, and you use multiple AI coding tools that should share the same context.

---

## Configuration

Located at `memxcore/config.yaml`. All fields are optional — sensible defaults apply.

```yaml
compaction:
  strategy: llm             # 'llm' = LLM distills to structured facts (via litellm)
                            # 'basic' = truncate first 50 lines to general.md
  threshold_tokens: 2000    # Trigger compaction when RECENT.md exceeds this
  min_entries: 3            # Don't compact if fewer than N entries
  check_interval: 5         # Check every N remember() calls (reduces I/O)
  stale_minutes: 10         # Auto-compact on search() if RECENT.md older than this

llm:
  # model: anthropic/claude-sonnet   # uncomment & change to your provider
  #   Examples: openai/gpt-4o, gemini/gemini-2.5-flash, ollama/llama3
  #   If omitted, auto-detects from available API key
  # api_key_env: OPENAI_API_KEY     # env var name (auto-detects if omitted)
  # base_url:                       # Optional: custom API endpoint

rag:
  embedding_model: paraphrase-multilingual-MiniLM-L12-v2  # multilingual
  top_k: 10
  rrf_k: 60                # Reciprocal Rank Fusion constant

memory:
  categories:
    - id: user_model
      description: "User identity, preferences, communication style, feedback"
    - id: domain
      description: "Transferable domain/technical knowledge"
    - id: project_state
      description: "Active tasks, in-progress decisions — decays within weeks"
    - id: episodic
      description: "Time-bound events and past decisions"
    - id: references
      description: "URLs, doc links, dashboards, ticket trackers"
```

---

## MCP Tools

| Tool | Args | Description |
|------|------|-------------|
| `remember` | `text`, `category?`, `permanent?`, `tenant_id?` | Store a memory |
| `search` | `query`, `max_results?`, `after?`, `before?`, `tenant_id?` | Search with optional time range |
| `compact` | `force?`, `tenant_id?` | Distill RECENT.md into categorized archives |
| `flush` | `tenant_id?` | Move RECENT.md to journal (lossless, no LLM) |
| `reindex` | `category?`, `tenant_id?` | Rebuild RAG + BM25 index |
| `purge_journal` | `keep_days?`, `tenant_id?` | Delete journal files older than N days |
| `set_config` | `key`, `value`, `tenant_id?` | Set a config value (dot notation) |
| `kg_add` | `subject`, `predicate`, `object`, `valid_from?` | Add a knowledge graph triple |
| `kg_query` | `entity`, `as_of?` | Query entity relationships |
| `kg_timeline` | `entity` | Chronological entity history |

---

## CLI

```bash
# Setup & diagnostics
memxcore setup                               # auto-configure all detected tools
memxcore setup --dry-run                     # preview without changes
memxcore doctor                              # check system readiness

# Configuration
memxcore config show                         # show effective config
memxcore config set llm.model openai/gpt-4o  # change LLM provider
memxcore config path                         # show config file location

# Memory operations
memxcore compact                             # force distillation
memxcore flush                               # move RECENT.md to journal (no LLM)
memxcore reindex                             # rebuild RAG + BM25 index

# Search & browse
memxcore search "query"                      # search memories
memxcore search --after 2026-04-08 "query"   # search with time range
memxcore timeline                            # last 7 days of activity
memxcore timeline --days 3                   # last 3 days

# Maintenance
memxcore purge-journal --keep-days 30        # clean old journal files
memxcore benchmark                           # search precision test
memxcore mine <path>                         # import conversations

# Multi-tenant
memxcore --tenant alice doctor
```

---

## Integration

### One-command setup (recommended)

```bash
memxcore setup
```

Auto-detects and configures:

| Tool | What it does |
|------|-------------|
| **Claude Code** | MCP server + hooks (auto-remember, auto-compact) + agent rules |
| **Cursor** | MCP + `hooks.json` (beforeSubmit + stop) + rules + merges `~/.cursor/cli-config.json` allowlist (`Mcp(memxcore:*)`, baseline `Read`/`Write`/`Shell` when needed). Uses the **same Python you run `setup` with**; adds `PYTHONPATH` only when memxcore is loaded from a source checkout (not from `site-packages`). |
| **Windsurf** | MCP config |
| **Codex (OpenAI)** | MCP config |
| **Gemini CLI** | MCP config |

### Manual setup

MCP server config for any tool:
```json
{
  "mcpServers": {
    "memxcore": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "memxcore.mcp_server"],
      "env": {
        "MEMXCORE_WORKSPACE": "/path/to/workspace"
      }
    }
  }
}
```

If memxcore is not installed in that interpreter’s `site-packages`, add `"PYTHONPATH": "/path/to/parent-of-memxcore-package"` to `env` so `-m memxcore.mcp_server` resolves.

---

## Multi-Tenant

All tools and commands accept `tenant_id` for isolated per-user/agent storage:

```bash
memxcore --tenant alice search "preferences"
```

Each tenant gets its own `tenants/<id>/storage/` and optional `config.yaml` override.

---

## Storage

```
storage/
+-- RECENT.md              Write buffer (cleared after compaction)
+-- USER.md                Permanent memory (auto-promoted facts)
+-- journal/               Lossless daily archive (never auto-deleted)
|   +-- 2026-04-10.md
|   +-- 2026-04-11.md
+-- archive/               Distilled knowledge by category
|   +-- user_model.md
|   +-- domain.md
|   +-- project_state.md
|   +-- episodic.md
|   +-- references.md
+-- chroma/                Vector index (rebuildable)
+-- knowledge.db           Entity relationship graph
+-- index.json             Keyword search index
```

All files are plain Markdown or SQLite — human-readable, editable, and `git`-friendly.

---

## Security

- **Local-only** — all data stays on your machine. No cloud, no telemetry.
- **HTTP server** binds to `127.0.0.1` only. Never expose to a network.
- **LLM prompt injection** — memory content is passed to LLMs during distillation. Category names are sanitized to prevent path traversal.
- **Storage permissions** — created with default OS permissions. On shared systems: `chmod 700 storage/`.

---

## Troubleshooting

Run `memxcore doctor` first — it checks everything and tells you what to fix.

```bash
memxcore doctor          # full diagnostic
memxcore reindex         # rebuild search indexes
memxcore config show     # verify configuration
```

---

## License

MIT. See [LICENSE](https://github.com/Danny0218/memxcore/blob/main/memxcore/LICENSE).
