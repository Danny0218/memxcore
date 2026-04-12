"""
memxcore MCP Server

Expose memxcore as MCP tools for Claude Code / Cursor / any MCP client.
Supports multi-tenant: each tool accepts an optional tenant_id parameter.

-- Setup (Claude Code) --------------------------------------------------
Add to ~/.claude/settings.json or project .claude/settings.json:

{
  "mcpServers": {
    "memxcore": {
      "command": "python",
      "args": ["-m", "memxcore.mcp_server"],
      "env": { "MEMXCORE_WORKSPACE": "/path/to/your/workspace" }
    }
  }
}

-- Tools ----------------------------------------------------------------
  remember(text, category?, tenant_id?)   Store a memory
  search(query, max_results?, tenant_id?) Semantic search
  compact(force?, tenant_id?)             Distill session memories
  reindex(category?, tenant_id?)          Rebuild vector index from MD files
"""

import logging
import os
import threading
import warnings
from typing import Any, Dict, List, Optional

# Suppress noisy third-party warnings
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ST_LOAD_REPORT", "0")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP

from memxcore.core import MemoryManager
from memxcore.core.paths import resolve_workspace


def _default_workspace() -> str:
    fallback = os.path.join(os.path.dirname(__file__), "..")
    return resolve_workspace(fallback)


_managers: Dict[str, MemoryManager] = {}
_managers_lock = threading.Lock()


def _get_manager(tenant_id: Optional[str] = None) -> MemoryManager:
    key = tenant_id or "__default__"
    if key not in _managers:
        # For the default manager, wait for the background warmup thread
        if key == "__default__" and not _warmup_event.is_set():
            _warmup_event.wait(timeout=60)
        with _managers_lock:
            if key not in _managers:
                _managers[key] = MemoryManager(
                    workspace_path=_default_workspace(),
                    tenant_id=tenant_id,
                )
    return _managers[key]


mcp = FastMCP(
    name="memxcore",
    instructions=(
        "Persistent memory across sessions. "
        "Use `search` at session start to load context. "
        "Use `remember` when learning something worth keeping. "
        "Use `compact` at session end."
    ),
)


@mcp.tool()
def remember(
    text: str,
    category: Optional[str] = None,
    permanent: bool = False,
    tenant_id: Optional[str] = None,
) -> str:
    """
    Store a memory for future sessions.

    category options:
      user_model    — user identity, preferences, communication style
      domain        — transferable technical/business knowledge
      project_state — current tasks, active decisions (decays in weeks)
      episodic      — time-bound events, past decisions, incidents
      references    — URLs, doc links, dashboards, ticket trackers

    permanent=True writes directly to USER.md (L2 — never compacted, highest
    priority in search). Use for stable facts that don't need distillation:
    core user identity, confirmed long-term preferences, architectural decisions
    that will not change. Normal memories go to RECENT.md → compacted to L1.
    """
    level = 2 if permanent else None
    path = _get_manager(tenant_id).remember(text, level=level, category=category)
    return f"stored → {path}"


@mcp.tool()
def search(
    query: str,
    max_results: int = 5,
    tenant_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search stored memories.

    Uses three-tier retrieval:
      1. RAG semantic search (ChromaDB, if available)
      2. LLM relevance judgment (if API key set)
      3. Keyword fallback (always available)

    Returns a list of relevant facts with relevance scores (0–1).
    """
    results = _get_manager(tenant_id).search(query, max_results=max_results)
    return [
        {
            "content": r.content,
            "category": r.metadata.get("category", ""),
            "score": r.relevance_score,
            "search_method": r.metadata.get("search", ""),
            "source": r.source,
        }
        for r in results
    ]


@mcp.tool()
def compact(
    force: bool = False,
    tenant_id: Optional[str] = None,
) -> str:
    """
    Distill recent memories (RECENT.md) into categorized long-term storage.

    Normally triggers automatically when RECENT.md exceeds the token
    threshold. Call explicitly at session end to ensure nothing is lost.

    force=True skips the threshold check and compacts immediately.
    """
    _get_manager(tenant_id).compact(force=force)
    return "compaction triggered (runs in background)"


@mcp.tool()
def reindex(
    category: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> str:
    """
    Re-embed archive files into the RAG index after manual edits.

    Call this after you or the user edits an archive/*.md file directly.

    category: reindex only this category (e.g. "project_state").
              Omit to reindex all archive files.
    """
    from memxcore.core.utils import sanitize_category
    mgr = _get_manager(tenant_id)
    if category:
        category = sanitize_category(category)
        path = os.path.join(mgr.archive_dir, f"{category}.md")
        if not os.path.isfile(path):
            return f"File not found: archive/{category}.md"
        count = mgr.rag.reindex_file(path, category)
    else:
        count = mgr.rebuild_rag_index()
    # Keep BM25 and index.json in sync with RAG
    mgr.bm25.rebuild(mgr.archive_dir, mgr.user_path)
    mgr.update_index()
    return f"Reindexed {'all' if not category else repr(category)}: {count} facts"


@mcp.tool()
def set_config(
    key: str,
    value: str,
    tenant_id: Optional[str] = None,
) -> str:
    """
    Set a memxcore config value. Uses dot notation for nested keys.

    Examples:
      set_config("llm.model", "openai/gpt-4o")
      set_config("llm.model", "gemini/gemini-2.5-flash")
      set_config("llm.api_key_env", "OPENAI_API_KEY")
      set_config("compaction.strategy", "llm")
      set_config("rag.top_k", "20")
    """
    from memxcore.core.utils import write_config_key
    from memxcore.core.paths import resolve_install_dir
    ws = _default_workspace()
    root_dir = resolve_install_dir(ws)
    if tenant_id:
        config_path = os.path.join(root_dir, "tenants", tenant_id, "config.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
    else:
        config_path = os.path.join(root_dir, "config.yaml")
    write_config_key(config_path, key, value)
    # Reload manager config so changes take effect immediately
    mgr = _get_manager(tenant_id)
    mgr.config = mgr._load_config()
    return f"Set {key} = {value!r} (written to {config_path})"


@mcp.tool()
def kg_add(
    subject: str,
    predicate: str,
    object: str,
    valid_from: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> str:
    """
    Add a relationship triple to the knowledge graph.

    Examples:
      kg_add("Alice", "leads", "frontend-team", valid_from="2026-03-01")
      kg_add("Redis", "used_for", "session-storage")
      kg_add("Kai", "left", "Orion-project", valid_from="2026-01-20")
    """
    mgr = _get_manager(tenant_id)
    row_id = mgr.kg.add_triple(subject, predicate, object, valid_from=valid_from)
    return f"Triple added (id={row_id}): {subject} → {predicate} → {object}"


@mcp.tool()
def kg_query(
    entity: str,
    as_of: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query all relationships for an entity (as subject or object).

    entity: person, project, tool, or service name
    as_of: ISO date to get relationships valid at that point in time (e.g. "2026-03-15")
    """
    mgr = _get_manager(tenant_id)
    triples = mgr.kg.query_entity(entity, as_of=as_of)
    return [
        {
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "valid_from": t.get("valid_from"),
            "ended": t.get("ended"),
        }
        for t in triples
    ]


@mcp.tool()
def kg_timeline(
    entity: str,
    tenant_id: Optional[str] = None,
) -> List[str]:
    """
    Get a chronological timeline of all events involving an entity.

    Returns formatted strings like: "Alice → joined → frontend-team (from 2026-03-01)"
    """
    from memxcore.core.knowledge_graph import KnowledgeGraph
    mgr = _get_manager(tenant_id)
    triples = mgr.kg.timeline(entity)
    return [KnowledgeGraph.format_triple(t) for t in triples]


_warmup_event = threading.Event()


def _warmup():
    """Pre-create the default MemoryManager after all imports are done.
    This runs the full synchronous init (including RAG model loading ~8s)
    in a background thread so the MCP server can respond to protocol
    messages while the model loads."""
    try:
        # Create MemoryManager directly (bypass _get_manager to avoid
        # deadlock — _get_manager waits on _warmup_event which we set)
        with _managers_lock:
            if "__default__" not in _managers:
                _managers["__default__"] = MemoryManager(
                    workspace_path=_default_workspace(),
                    tenant_id=None,
                )
    except Exception as e:
        logging.getLogger("memxcore").warning("Warmup failed: %s", e)
    finally:
        _warmup_event.set()


if __name__ == "__main__":
    # Start warmup in background AFTER all imports are done (avoids import lock deadlock).
    # The MCP server can handle initialize/tools/list while the model loads.
    # First tool call will block until warmup completes.
    threading.Thread(target=_warmup, daemon=True, name="memxcore-warmup").start()
    mcp.run()
