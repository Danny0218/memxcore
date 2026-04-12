"""
memxcore CLI

Usage:
    python -m memxcore.cli doctor                        # check system readiness
    python -m memxcore.cli reindex                       # reindex all archive files
    python -m memxcore.cli reindex project_state         # reindex one category
    python -m memxcore.cli compact                       # force distillation
    python -m memxcore.cli flush                         # move RECENT.md to journal (no LLM)
    python -m memxcore.cli purge-journal --keep-days 30  # delete old journal files
    python -m memxcore.cli search "query text"           # search memories (debug)
    python -m memxcore.cli --tenant alice search "test"  # multi-tenant
    python -m memxcore.cli benchmark                     # run search precision benchmark
    python -m memxcore.cli benchmark --verbose           # with per-query details
    python -m memxcore.cli mine <path>                   # import conversations/files
    python -m memxcore.cli mine ~/.claude/projects/*/    # import Claude Code history
"""

import argparse
import logging
import os
import sys
import warnings
from typing import Optional

# Suppress noisy third-party warnings during CLI usage
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ST_LOAD_REPORT", "0")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def _get_manager(tenant_id: Optional[str] = None):
    from memxcore.core import MemoryManager
    from memxcore.core.paths import resolve_workspace
    workspace = resolve_workspace(os.path.join(os.path.dirname(__file__), ".."))
    return MemoryManager(workspace_path=workspace, tenant_id=tenant_id)


def cmd_reindex(category: Optional[str], tenant_id: Optional[str] = None) -> None:
    manager = _get_manager(tenant_id)

    if not manager.rag.available:
        print("RAG not available (chromadb or sentence-transformers not installed)")
        sys.exit(1)

    if category:
        from memxcore.core.utils import sanitize_category
        category = sanitize_category(category)
        path = os.path.join(manager.archive_dir, f"{category}.md")
        if not os.path.isfile(path):
            print(f"File not found: {category}.md")
            sys.exit(1)
        count = manager.rag.reindex_file(path, category)
        print(f"Reindexed '{category}': {count} facts")
    else:
        count = manager.rebuild_rag_index()
        print(f"Reindexed all: {count} facts")

    manager.bm25.rebuild(manager.archive_dir, manager.user_path)
    manager.update_index()
    print("BM25 + index.json updated")


def cmd_compact(tenant_id: Optional[str] = None) -> None:
    manager = _get_manager(tenant_id)
    manager.compact(force=True, blocking=True)
    print("Compaction complete")


def cmd_flush(tenant_id: Optional[str] = None) -> None:
    manager = _get_manager(tenant_id)
    result = manager.flush()
    if result == "flushed":
        print("RECENT.md flushed to journal")
    elif result == "empty":
        print("Nothing to flush (RECENT.md is empty)")
    else:
        print(f"Flush failed: {result}")


def cmd_purge_journal(keep_days: int = 30, tenant_id: Optional[str] = None) -> None:
    manager = _get_manager(tenant_id)
    count = manager.purge_journal(keep_days=keep_days)
    print(f"Purged {count} journal file(s) older than {keep_days} days")


def cmd_search(query: str, after: Optional[str] = None, before: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
    manager = _get_manager(tenant_id)
    results = manager.search(query, max_results=5, after=after, before=before)
    if not results:
        print("No results found")
        return
    for i, r in enumerate(results, 1):
        method = r.metadata.get("search", "?")
        score = f"{r.relevance_score:.2f}"
        cat = r.metadata.get("category", r.source)
        print(f"\n[{i}] score={score} via={method} category={cat}")
        print(f"    {r.content[:200]}")


def cmd_timeline(days: int = 7, after: Optional[str] = None, before: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
    """Show chronological memory timeline from journal and archive files."""
    from datetime import datetime, timedelta
    from memxcore.core.rag import _split_archive_sections
    from memxcore.core.utils import parse_front_matter
    import re

    manager = _get_manager(tenant_id)

    # Determine date range
    if not before:
        before = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    if not after:
        after = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

    entries = []  # [(timestamp, label, content), ...]

    # Journal files use RECENT.md format: "# [TIMESTAMP] Memory" or "# [TIMESTAMP] [category:X] Memory"
    _journal_entry_re = re.compile(r'^# \[([^\]]+)\]', re.MULTILINE)

    # Read journal/*.md files within the date range
    if os.path.isdir(manager.journal_dir):
        for name in sorted(os.listdir(manager.journal_dir)):
            if not name.endswith(".md"):
                continue
            date_part = name[:-3]  # strip .md
            if re.match(r"^\d{4}-\d{2}-\d{2}$", date_part):
                if date_part > before[:10]:
                    continue
                if date_part < after[:10]:
                    continue

            path = os.path.join(manager.journal_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except OSError:
                continue

            # Parse journal entries: split by "# [timestamp]" (single #, RECENT.md format)
            # Skip "# === Archived at ... ===" separator lines
            parts = _journal_entry_re.split(raw)
            # parts = [pre-text, ts1, body1, ts2, body2, ...]
            for i in range(1, len(parts) - 1, 2):
                ts = parts[i]
                body = parts[i + 1].strip()
                # Skip archive separator headers captured as entries
                if ts.startswith("==="):
                    continue
                if not body:
                    continue
                _before_padded = before if "T" in before else before + "T23:59:59.999999"
                if ts >= after and ts <= _before_padded:
                    # Extract actual content: skip header remnants like "[category:X] Memory"
                    content_lines = [
                        l.strip() for l in body.split("\n")
                        if l.strip()
                        and not l.strip().startswith("[category:")
                        and l.strip() != "Memory"
                        and not l.strip().startswith("Memory [category=")
                    ]
                    preview = content_lines[0] if content_lines else "(empty)"
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    entries.append((ts, "[journal]", preview))

    # Read archive/*.md sections within the date range
    if os.path.isdir(manager.archive_dir):
        for name in sorted(os.listdir(manager.archive_dir)):
            if not name.endswith(".md"):
                continue
            category = name[:-3]
            path = os.path.join(manager.archive_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except OSError:
                continue
            _, body = parse_front_matter(raw)
            _before_arc = before if "T" in before else before + "T23:59:59.999999"
            for distilled_at, content in _split_archive_sections(body):
                if distilled_at >= after and distilled_at <= _before_arc:
                    preview = content.strip().replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    entries.append((distilled_at, f"[archive/{category}]", preview))

    # Read USER.md sections within the date range
    _before_user = before if "T" in before else before + "T23:59:59.999999"
    if os.path.isfile(manager.user_path):
        try:
            with open(manager.user_path, "r", encoding="utf-8") as f:
                raw = f.read()
            for distilled_at, content in _split_archive_sections(raw):
                if distilled_at >= after and distilled_at <= _before_user:
                    preview = content.strip().replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    entries.append((distilled_at, "[user]", preview))
        except OSError:
            pass

    # Sort chronologically and print
    entries.sort(key=lambda e: e[0])

    if not entries:
        print(f"No entries found between {after[:10]} and {before[:10]}")
        return

    print(f"Timeline: {after[:10]} to {before[:10]} ({len(entries)} entries)")
    print("=" * 60)
    for ts, label, content in entries:
        print(f"\n{ts}  {label}")
        print(f"  {content}")


def cmd_benchmark(dataset: Optional[str] = None, verbose: bool = False) -> None:
    """Run search precision benchmark."""
    from memxcore.core.benchmark import run_benchmark, format_report

    from memxcore.core.paths import resolve_install_dir, resolve_workspace
    workspace = resolve_workspace(os.path.join(os.path.dirname(__file__), ".."))

    if dataset is None:
        dataset = os.path.join(
            resolve_install_dir(workspace), "benchmarks", "default.json"
        )

    if not os.path.isfile(dataset):
        print(f"Dataset not found: {dataset}")
        sys.exit(1)

    print(f"Running benchmark: {os.path.basename(dataset)}")
    print("Loading memories and building indexes...")
    result = run_benchmark(dataset, workspace, verbose=verbose)
    print()
    print(format_report(result))


def cmd_mine(path: str, tenant_id: Optional[str] = None) -> None:
    """Import conversations/files into memxcore."""
    from memxcore.core.mining import mine_file, mine_directory

    manager = _get_manager(tenant_id)

    if not os.path.exists(path):
        print(f"Path not found: {path}")
        sys.exit(1)

    def progress(msg: str) -> None:
        print(msg)

    if os.path.isdir(path):
        print(f"Mining directory: {path}")
        results = mine_directory(path, manager, manager.config, on_progress=progress)
    else:
        print(f"Mining file: {path}")
        results = [mine_file(path, manager, manager.config, on_progress=progress)]

    # Summary
    total_facts = sum(r["facts"] for r in results)
    total_triples = sum(r["triples"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_files = len(results)

    print(f"\n{'=' * 40}")
    print(f"Mining complete:")
    print(f"  Files:    {total_files}")
    print(f"  Facts:    {total_facts}")
    print(f"  Triples:  {total_triples}")
    if total_errors:
        print(f"  Errors:   {total_errors}")

    # Per-file breakdown
    for r in results:
        status = f"{r['facts']} facts, {r['triples']} triples"
        if r.get("error"):
            status = f"ERROR: {r['error']}"
        print(f"  {r['file']}: {status}")


def cmd_config(args, tenant_id: Optional[str] = None) -> None:
    """View or change configuration."""
    import yaml as _yaml
    from memxcore.core.paths import resolve_workspace, resolve_install_dir
    from memxcore.core.utils import write_config_key

    workspace = resolve_workspace(os.path.join(os.path.dirname(__file__), ".."))
    root_dir = resolve_install_dir(workspace)
    ws_config_path = os.path.join(root_dir, "config.yaml")
    pkg_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    action = getattr(args, "config_action", None)

    # Tenant-specific config path
    tenant_config_path = None
    if tenant_id:
        tenant_config_path = os.path.join(root_dir, "tenants", tenant_id, "config.yaml")

    if action == "path":
        if tenant_id and tenant_config_path:
            if os.path.isfile(tenant_config_path):
                print(f"{tenant_config_path}  (tenant override)")
            else:
                print(f"{tenant_config_path}  (not created yet)")
            print(f"Base: {ws_config_path}")
        elif os.path.isfile(ws_config_path):
            print(ws_config_path)
        else:
            print(f"{ws_config_path}  (not created yet, using bundled defaults)")
        return

    if action == "set":
        key = args.key
        value = args.value
        if tenant_id:
            target_path = os.path.join(root_dir, "tenants", tenant_id, "config.yaml")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
        else:
            target_path = ws_config_path
        write_config_key(target_path, key, value)
        print(f"Set {key} = {value!r}")
        print(f"Written to {target_path}")
        return

    # Default: show (use MemoryManager's merged config for accurate tenant view)
    mgr = _get_manager(tenant_id)
    config = mgr.config
    if tenant_id and tenant_config_path and os.path.isfile(tenant_config_path):
        source = f"{ws_config_path} + {tenant_config_path} (merged)"
    elif os.path.isfile(ws_config_path):
        source = ws_config_path
    elif os.path.isfile(pkg_config_path):
        source = f"{pkg_config_path} (bundled)"
    else:
        source = "defaults"

    print(f"# Source: {source}\n")
    print(_yaml.safe_dump(config, allow_unicode=True, default_flow_style=False).strip())


def cmd_doctor(tenant_id: Optional[str] = None) -> None:
    """Check system readiness: deps, config, storage, search capabilities."""
    import platform

    from memxcore.core.paths import resolve_install_dir, resolve_workspace
    workspace = resolve_workspace(os.path.join(os.path.dirname(__file__), ".."))

    root_dir = resolve_install_dir(workspace)

    print("MemXCore Doctor")
    print(f"{'=' * 40}")

    # Python version
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 11)
    _status(py_ok, f"Python {py_ver}", None if py_ok else "Requires 3.11+")

    # config.yaml
    config_path = os.path.join(root_dir, "config.yaml")
    pkg_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.isfile(config_path):
        _status(True, "config.yaml", config_path)
    elif os.path.isfile(pkg_config_path):
        _status(True, "config.yaml (bundled)", pkg_config_path)
        config_path = pkg_config_path
    else:
        _status(False, "config.yaml", "Not found (using defaults)")

    # API key
    import yaml
    config = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            pass
    api_key_env = (config.get("llm") or {}).get("api_key_env", "")
    has_key = bool(api_key_env and os.environ.get(api_key_env))
    if not has_key:
        has_key = any(
            os.environ.get(k)
            for k in ("ANTHROPIC_API_KEY",
                       "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_BASE")
        )
    found_key_name = next(
        (k for k in (api_key_env, "ANTHROPIC_API_KEY",
                      "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_BASE")
         if k and os.environ.get(k)), None
    )
    if has_key:
        _status(True, f"LLM API key ({found_key_name})")
    else:
        _status(False, "LLM API key",
                "No API key found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.\n"
                "           LLM distillation will fall back to basic mode.")

    # Core dependencies
    print(f"\n{'Dependencies':}")
    _check_import("litellm", "litellm", "pip install litellm", required=True)
    _check_import("mcp", "mcp", "pip install mcp", required=True)
    _check_import("pydantic", "pydantic", "pip install pydantic", required=True)
    _check_import("yaml", "pyyaml", "pip install pyyaml", required=True)

    # Optional dependencies
    print(f"\n{'Optional (search quality):':}")
    chromadb_ok = _check_import("chromadb", "chromadb", "pip install chromadb", required=False)
    st_ok = _check_import("sentence_transformers", "sentence-transformers", "pip install sentence-transformers", required=False)
    rag_ok = chromadb_ok and st_ok
    if not rag_ok:
        print("           RAG semantic search disabled. Using keyword fallback.")
        print("           Fix: pip install 'memxcore[rag,bm25]'")
    _check_import("rank_bm25", "rank-bm25", "pip install rank-bm25", required=False)
    _check_import("watchdog", "watchdog", "pip install watchdog", required=False)
    _check_import("fastapi", "fastapi", "pip install fastapi", required=False)

    # Storage state
    if tenant_id:
        storage_dir = os.path.join(root_dir, "tenants", tenant_id, "storage")
    else:
        storage_dir = os.path.join(root_dir, "storage")
    archive_dir = os.path.join(storage_dir, "archive")

    print(f"\n{'Storage':}")
    from memxcore.core.paths import _is_site_packages
    if _is_site_packages(storage_dir):
        _status(False, "Storage directory",
                f"{storage_dir}\n"
                "           ⚠ Storage is inside site-packages — memories will be lost on pip upgrade!\n"
                "           Fix: export MEMXCORE_WORKSPACE=/path/to/your/project")
    else:
        _status(os.path.isdir(storage_dir), "Storage directory", storage_dir)

    if os.path.isdir(archive_dir):
        from memxcore.core.rag import _split_archive_sections
        from memxcore.core.utils import parse_front_matter

        total_facts = 0
        categories = []
        for name in sorted(os.listdir(archive_dir)):
            if not name.endswith(".md"):
                continue
            cat = name[:-3]
            path = os.path.join(archive_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                _, body = parse_front_matter(raw)
                facts = _split_archive_sections(body)
                total_facts += len(facts)
                categories.append(f"{cat}({len(facts)})")
            except Exception:
                categories.append(f"{cat}(?)")
        _status(True, f"{total_facts} memories across {len(categories)} categories")
        if categories:
            print(f"           {', '.join(categories)}")
    else:
        _status(False, "No archive directory yet (will be created on first compaction)")

    # RECENT.md
    recent_path = os.path.join(storage_dir, "RECENT.md")
    if os.path.isfile(recent_path):
        try:
            with open(recent_path, "r", encoding="utf-8") as f:
                content = f.read()
            entry_count = content.count("# [")
            if entry_count > 0:
                print(f"  !  RECENT.md has {entry_count} uncommitted entries (run compact to distill)")
            else:
                _status(True, "RECENT.md empty (all compacted)")
        except Exception:
            pass

    # Search capability summary
    print(f"\n{'Search capabilities':}")
    tiers = []
    if rag_ok:
        tiers.append("RAG semantic (tier 1)")
    if has_key:
        tiers.append("LLM relevance (tier 2)")
    tiers.append("Keyword fallback (tier 3)")
    for t in tiers:
        _status(True, t)
    if not rag_ok and not has_key:
        print("\n  !  Only keyword fallback available. Search quality will be poor.")
        print("     Install RAG deps or set API key for better results.")

    print()


def _status(ok: bool, message: str, detail: str = None) -> None:
    icon = "  ok" if ok else "  !!"
    print(f"{icon} {message}")
    if detail:
        print(f"           {detail}")


def _check_import(module: str, package: str, fix: str, required: bool) -> bool:
    try:
        __import__(module)
        ver = ""
        try:
            import importlib.metadata
            ver = f" ({importlib.metadata.version(package)})"
        except Exception:
            pass
        _status(True, f"{package}{ver}")
        return True
    except ImportError:
        label = "REQUIRED" if required else "optional"
        _status(False, f"{package} — not installed [{label}]")
        if required:
            print(f"           Fix: {fix}")
        return False


def cmd_setup(dry_run: bool = False, skip_hooks: bool = False, workspace_override: Optional[str] = None) -> None:
    """One-command setup: detect installed tools and configure MCP + hooks + rules."""
    import json
    import shutil
    import subprocess

    python_path = sys.executable
    from memxcore.core.paths import _is_site_packages
    memx_claude_md = os.path.join(os.path.dirname(__file__), "CLAUDE.md")

    # ── Resolve workspace (must be explicit) ─────────────────────────
    workspace = (
        workspace_override
        or os.environ.get("MEMXCORE_WORKSPACE", "").strip()
        or ""
    )
    if workspace and not _is_site_packages(workspace):
        workspace = os.path.abspath(os.path.expanduser(workspace))
        print("MemXCore Setup")
        print(f"{'=' * 50}")
        print(f"  Python:    {python_path}")
        print(f"  Workspace: {workspace}")
    else:
        default_ws = os.path.expanduser("~/.memxcore")
        print("MemXCore Setup")
        print(f"{'=' * 50}")
        print(f"  Python:    {python_path}")
        print()
        print("  Where should memxcore store memories?")
        print(f"  Press Enter for default: {default_ws}")
        if dry_run:
            workspace = default_ws
            print(f"  (dry-run: using default)")
        else:
            try:
                user_input = input("  Workspace path: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Aborted.")
                return
            workspace = user_input if user_input else default_ws
        workspace = os.path.abspath(os.path.expanduser(workspace))
        print(f"  Workspace: {workspace}")

    # ── Detect installed tools ────────────────────────────────────────
    tools_found = []

    # Claude Code
    if shutil.which("claude"):
        tools_found.append("claude")

    # Cursor
    cursor_dir = os.path.expanduser("~/.cursor")
    if os.path.isdir(cursor_dir) or shutil.which("cursor"):
        tools_found.append("cursor")

    # Windsurf / Codeium
    windsurf_dir = os.path.expanduser("~/.codeium")
    windsurf_dir2 = os.path.expanduser("~/.windsurf")
    if os.path.isdir(windsurf_dir) or os.path.isdir(windsurf_dir2) or shutil.which("windsurf"):
        tools_found.append("windsurf")

    # Codex (OpenAI)
    codex_dir = os.path.expanduser("~/.codex")
    if os.path.isdir(codex_dir) or shutil.which("codex"):
        tools_found.append("codex")

    # Gemini CLI
    if shutil.which("gemini"):
        tools_found.append("gemini")

    if not tools_found:
        print("\n  !!  No supported tools detected (Claude Code, Cursor, Windsurf, Gemini CLI)")
        print("       Install one of them first, then re-run memxcore setup.")
        return

    print(f"  Tools:     {', '.join(tools_found)}")
    print()

    actions_taken = []
    step = 0
    total_steps = len(tools_found)

    # MCP server config (shared across tools)
    mcp_config = {
        "command": python_path,
        "args": ["-m", "memxcore.mcp_server"],
        "env": {"MEMXCORE_WORKSPACE": workspace},
    }

    # ── Claude Code ───────────────────────────────────────────────────
    if "claude" in tools_found:
        step += 1
        print(f"[{step}/{total_steps}] Claude Code")

        # MCP registration
        mcp_cmd = ["claude", "mcp", "add", "-s", "user",
                   "-e", f"MEMXCORE_WORKSPACE={workspace}",
                   "memxcore", "--",
                   python_path, "-m", "memxcore.mcp_server"]
        if dry_run:
            print(f"  [dry-run] Would run: {' '.join(mcp_cmd)}")
        else:
            result = subprocess.run(mcp_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("  ok  MCP server registered")
                actions_taken.append("Claude Code: MCP")
            else:
                err = (result.stderr or result.stdout or "").strip()
                print(f"  !!  MCP registration failed: {err}")

        # Hooks
        if not skip_hooks:
            claude_dir = os.path.expanduser("~/.claude")
            settings_path = os.path.join(claude_dir, "settings.json")
            _write_claude_hooks(settings_path, workspace, python_path, dry_run)
            if not dry_run:
                actions_taken.append("Claude Code: hooks")

        # CLAUDE.md
        claude_md_path = os.path.join(os.path.expanduser("~/.claude"), "CLAUDE.md")
        _append_rules(memx_claude_md, claude_md_path, "~/.claude/CLAUDE.md", dry_run)
        if not dry_run:
            actions_taken.append("Claude Code: CLAUDE.md")

    # ── Cursor ────────────────────────────────────────────────────────
    if "cursor" in tools_found:
        step += 1
        print(f"\n[{step}/{total_steps}] Cursor")

        mcp_json_path = os.path.join(cursor_dir, "mcp.json")
        _write_mcp_json(mcp_json_path, mcp_config, "~/.cursor/mcp.json", dry_run)
        if not dry_run:
            actions_taken.append("Cursor: MCP")

        rules_path = os.path.join(cursor_dir, "rules", "memxcore.mdc")
        _copy_cursor_rules(memx_claude_md, rules_path, "~/.cursor/rules/memxcore.mdc", dry_run)
        if not dry_run:
            actions_taken.append("Cursor: rules")

    # ── Windsurf ──────────────────────────────────────────────────────
    if "windsurf" in tools_found:
        step += 1
        print(f"\n[{step}/{total_steps}] Windsurf")

        ws_mcp_dir = os.path.join(windsurf_dir, "windsurf")
        mcp_json_path = os.path.join(ws_mcp_dir, "mcp_config.json")
        _write_mcp_json(mcp_json_path, mcp_config, "~/.codeium/windsurf/mcp_config.json", dry_run)
        if not dry_run:
            actions_taken.append("Windsurf: MCP")
        print("  NOTE: If MCP is not enabled in Windsurf, go to Settings -> enable 'Model Context Protocol'")

    # ── Codex (OpenAI) ────────────────────────────────────────────────
    if "codex" in tools_found:
        step += 1
        print(f"\n[{step}/{total_steps}] Codex (OpenAI)")

        codex_config_path = os.path.join(codex_dir, "config.toml")
        _write_codex_toml(codex_config_path, python_path, workspace, dry_run)
        if not dry_run:
            actions_taken.append("Codex: MCP")

    # ── Gemini CLI ────────────────────────────────────────────────────
    if "gemini" in tools_found:
        step += 1
        print(f"\n[{step}/{total_steps}] Gemini CLI")

        gemini_settings = os.path.expanduser("~/.gemini/settings.json")
        _write_mcp_json(gemini_settings, mcp_config, "~/.gemini/settings.json", dry_run)
        if not dry_run:
            actions_taken.append("Gemini CLI: MCP")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    if dry_run:
        print("Dry run complete. No changes were made.")
    elif actions_taken:
        print(f"Done! ({', '.join(actions_taken)})")
        print("\nRestart your tools for changes to take effect.")
    else:
        print("Nothing new to configure — MemXCore is already set up.")


def _write_claude_hooks(settings_path: str, workspace: str, python_path: str, dry_run: bool) -> None:
    """Write auto-remember + auto-compact hooks to Claude Code settings.json."""
    import json
    import shlex
    import shutil

    # Shell-quote paths to handle spaces and special characters
    q_ws = shlex.quote(workspace)
    q_py = shlex.quote(python_path)
    env_prefix = f"MEMXCORE_WORKSPACE={q_ws}"
    hooks_config = {
        "UserPromptSubmit": [{"hooks": [{
            "type": "command",
            "command": f"{env_prefix} {q_py} -m memxcore.hooks.user_prompt_submit",
            "timeout": 10,
        }]}],
        "Stop": [{"hooks": [
            {"type": "command", "command": f"{env_prefix} {q_py} -m memxcore.hooks.auto_remember", "timeout": 30},
            {"type": "command", "command": f"{env_prefix} {q_py} -m memxcore.cli compact 2>/dev/null || true"},
        ]}],
    }

    if dry_run:
        print("  [dry-run] Would write hooks to settings.json")
        return

    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    settings = {}
    if os.path.isfile(settings_path):
        shutil.copy2(settings_path, settings_path + ".bak")
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except (json.JSONDecodeError, OSError):
            settings = {}

    existing_hooks = settings.get("hooks", {})
    for event, hook_groups in hooks_config.items():
        if event in existing_hooks:
            cleaned = []
            for group in existing_hooks[event]:
                filtered = [h for h in group.get("hooks", []) if "memxcore." not in h.get("command", "")]
                if filtered:
                    cleaned.append({"hooks": filtered})
            cleaned.extend(hook_groups)
            existing_hooks[event] = cleaned
        else:
            existing_hooks[event] = hook_groups

    settings["hooks"] = existing_hooks
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("  ok  Hooks configured")


def _write_codex_toml(path: str, python_path: str, workspace: str, dry_run: bool) -> None:
    """Write MCP server config into Codex's config.toml (TOML format)."""
    if dry_run:
        print("  [dry-run] Would write MCP config to ~/.codex/config.toml")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Read existing TOML if present
    existing_content = ""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing_content = f.read()
        except OSError:
            pass

    # Check if memxcore section already exists
    if "[mcp_servers.memxcore]" in existing_content:
        print("  --  MCP config already present in ~/.codex/config.toml")
        return

    # Append TOML section
    section = (
        f'\n[mcp_servers.memxcore]\n'
        f'command = "{python_path}"\n'
        f'args = ["-m", "memxcore.mcp_server"]\n'
        f'\n'
        f'[mcp_servers.memxcore.env]\n'
        f'MEMXCORE_WORKSPACE = "{workspace}"\n'
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(section)
    print("  ok  MCP config written to ~/.codex/config.toml")


def _write_mcp_json(path: str, mcp_config: dict, display_path: str, dry_run: bool) -> None:
    """Write or merge MCP server config into a JSON file (Cursor, Windsurf, Gemini)."""
    import json

    if dry_run:
        print(f"  [dry-run] Would write MCP config to {display_path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    servers = existing.get("mcpServers", {})
    servers["memxcore"] = mcp_config
    existing["mcpServers"] = servers

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  ok  MCP config written to {display_path}")


def _append_rules(src_path: str, dst_path: str, display_path: str, dry_run: bool) -> None:
    """Append agent rules to a target file if not already present."""
    if not os.path.isfile(src_path):
        print(f"  !!  Source rules not found: {src_path}")
        return

    marker = "# MemXCore"
    already = False
    if os.path.isfile(dst_path):
        try:
            with open(dst_path, "r", encoding="utf-8") as f:
                if marker in f.read():
                    already = True
        except OSError:
            pass

    if already:
        print(f"  --  Rules already present in {display_path}")
        return
    if dry_run:
        print(f"  [dry-run] Would append rules to {display_path}")
        return

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(dst_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + content)
    print(f"  ok  Rules appended to {display_path}")


def _copy_rules(src_path: str, dst_path: str, display_path: str, dry_run: bool) -> None:
    """Copy agent rules to a target file."""
    if not os.path.isfile(src_path):
        print(f"  !!  Source rules not found: {src_path}")
        return

    if os.path.isfile(dst_path):
        print(f"  --  Rules already exist at {display_path}")
        return
    if dry_run:
        print(f"  [dry-run] Would copy rules to {display_path}")
        return

    import shutil
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"  ok  Rules copied to {display_path}")


def _copy_cursor_rules(src_path: str, dst_path: str, display_path: str, dry_run: bool) -> None:
    """Copy agent rules as Cursor .mdc format with frontmatter."""
    if not os.path.isfile(src_path):
        print(f"  !!  Source rules not found: {src_path}")
        return

    if os.path.isfile(dst_path):
        print(f"  --  Rules already exist at {display_path}")
        return
    if dry_run:
        print(f"  [dry-run] Would copy rules to {display_path}")
        return

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()

    mdc_content = (
        "---\n"
        "description: MemXCore persistent memory system instructions\n"
        "alwaysApply: true\n"
        "---\n\n"
        + content
    )
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(mdc_content)
    print(f"  ok  Rules copied to {display_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m memxcore.cli",
        description="MemX CLI",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Tenant ID for multi-tenant mode (omit for default single-tenant)",
    )
    sub = parser.add_subparsers(dest="command")

    # doctor
    sub.add_parser("doctor", help="Check system readiness (deps, config, storage)")

    # reindex
    p_reindex = sub.add_parser("reindex", help="Re-embed archive files into RAG index")
    p_reindex.add_argument(
        "category",
        nargs="?",
        help="Category to reindex (e.g. project_state). Omit to reindex all.",
    )

    # compact
    sub.add_parser("compact", help="Force compaction of RECENT.md")

    # flush
    sub.add_parser("flush", help="Move RECENT.md to journal (no LLM)")

    # purge-journal
    p_purge = sub.add_parser("purge-journal", help="Delete old journal files")
    p_purge.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Keep journal files from the last N days (default: 30)",
    )

    # search
    p_search = sub.add_parser("search", help="Search memories (debug)")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--after", default=None, help="Filter results after this date (YYYY-MM-DD or ISO 8601)")
    p_search.add_argument("--before", default=None, help="Filter results before this date (YYYY-MM-DD or ISO 8601)")

    # timeline
    p_timeline = sub.add_parser("timeline", help="Show chronological memory timeline")
    p_timeline.add_argument("--days", type=int, default=7, help="Number of days to look back (default: 7)")
    p_timeline.add_argument("--after", default=None, help="Start date (YYYY-MM-DD)")
    p_timeline.add_argument("--before", default=None, help="End date (YYYY-MM-DD)")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run search precision benchmark")
    p_bench.add_argument("--dataset", default=None, help="Path to benchmark JSON file")
    p_bench.add_argument("--verbose", "-v", action="store_true", help="Show per-query results")

    # setup
    p_setup = sub.add_parser("setup", help="Auto-detect tools and configure MCP + hooks + rules")
    p_setup.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    p_setup.add_argument("--skip-hooks", action="store_true", help="Skip Claude Code hook configuration")
    p_setup.add_argument("--workspace", default=None, help="Workspace path (where memories are stored)")

    # config
    p_config = sub.add_parser("config", help="View or change configuration")
    config_sub = p_config.add_subparsers(dest="config_action")
    config_sub.add_parser("show", help="Show current config")
    config_sub.add_parser("path", help="Show config file path")
    p_config_set = config_sub.add_parser("set", help="Set a config value (dot notation)")
    p_config_set.add_argument("key", help="Config key (e.g. llm.model)")
    p_config_set.add_argument("value", help="Value to set")

    # mine
    p_mine = sub.add_parser("mine", help="Import conversations/files into memxcore")
    p_mine.add_argument("path", help="File or directory to import (.jsonl, .json, .md, .txt)")

    args = parser.parse_args()

    if args.command == "config":
        cmd_config(args, tenant_id=args.tenant)
    elif args.command == "setup":
        cmd_setup(
            dry_run=getattr(args, "dry_run", False),
            skip_hooks=getattr(args, "skip_hooks", False),
            workspace_override=getattr(args, "workspace", None),
        )
    elif args.command == "mine":
        cmd_mine(args.path, tenant_id=args.tenant)
    elif args.command == "benchmark":
        cmd_benchmark(
            dataset=getattr(args, "dataset", None),
            verbose=getattr(args, "verbose", False),
        )
    elif args.command == "doctor":
        cmd_doctor(tenant_id=args.tenant)
    elif args.command == "reindex":
        cmd_reindex(getattr(args, "category", None), tenant_id=args.tenant)
    elif args.command == "compact":
        cmd_compact(tenant_id=args.tenant)
    elif args.command == "flush":
        cmd_flush(tenant_id=args.tenant)
    elif args.command == "purge-journal":
        cmd_purge_journal(
            keep_days=getattr(args, "keep_days", 30),
            tenant_id=args.tenant,
        )
    elif args.command == "search":
        cmd_search(args.query, after=getattr(args, "after", None), before=getattr(args, "before", None), tenant_id=args.tenant)
    elif args.command == "timeline":
        cmd_timeline(
            days=getattr(args, "days", 7),
            after=getattr(args, "after", None),
            before=getattr(args, "before", None),
            tenant_id=args.tenant,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
