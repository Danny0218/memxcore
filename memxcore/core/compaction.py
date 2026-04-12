import fcntl
import logging
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import utils
from .rag import _split_archive_sections

logger = logging.getLogger(__name__)


# ── Journal archival ─────────────────────────────────────────────────────────

def _append_to_journal(manager: "object", content: str, snapshot_time: Optional[str] = None) -> None:
    """
    Append raw RECENT.md content to a daily journal file for permanent archival.
    Uses utils.append_with_lock() for cross-process safety.
    Failures are logged but never block compaction.

    snapshot_time: ISO date string (YYYY-MM-DD) for the journal filename.
                   If not provided, uses current UTC date.
    """
    try:
        journal_dir = getattr(manager, "journal_dir", None)
        if not journal_dir:
            return
        os.makedirs(journal_dir, exist_ok=True)
        date_str = snapshot_time or datetime.utcnow().strftime("%Y-%m-%d")
        journal_path = os.path.join(journal_dir, f"{date_str}.md")
        # Separator between entries when multiple compacts happen in the same day
        header = f"\n\n# === Archived at {datetime.utcnow().isoformat()} ===\n\n"
        utils.append_with_lock(journal_path, header + content)
    except Exception:
        logger.warning("Failed to write journal entry", exc_info=True)


# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count, supporting mixed CJK/English content.
    Each CJK character counts as ~1 token; English is split by whitespace.
    """
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))
    non_cjk = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', ' ', text)
    words = len(non_cjk.split())
    return max(1, cjk + words)


# ── LLM distillation ──────────────────────────────────────────────────────────

def _default_categories() -> List[Dict[str, str]]:
    return [
        {"id": "user_model",    "description": "Who the user is: role, expertise, communication style, preferences, feedback on agent behaviour"},
        {"id": "domain",        "description": "Transferable domain or technical knowledge: protocols, standards, business logic — valid across projects"},
        {"id": "project_state", "description": "Current working memory: active tasks, in-progress decisions, this week's focus — decays within weeks"},
        {"id": "episodic",      "description": "Time-bound events and historical decisions: what happened, what was decided, past incidents — has TTL"},
        {"id": "references",    "description": "Pointers to external resources: URLs, doc links, dashboard locations, ticket trackers, Slack channels"},
    ]


def _build_distill_prompt(content: str, categories: List[Dict[str, str]]) -> str:
    cat_lines = "\n".join(f'- {c["id"]}: {c["description"]}' for c in categories)
    cat_ids = ", ".join(c["id"] for c in categories)
    return f"""You are a memory distillation assistant. Extract key facts AND behavioral patterns from raw conversation notes.

Categories:
{cat_lines}

Rules:
1. Extract only concrete, reusable facts — skip trivial or transient details
2. Each fact must be self-contained (readable without the surrounding context)
3. Pick the single most fitting category from: {cat_ids}
4. Tags: provide TWO types of tags for each fact:
   - "tags": 1-3 topic keywords (e.g., "testing", "auth", "scheduler")
   - "entities": named entities mentioned in the fact — people, projects, tools, ticket IDs,
     service names, product names. Use exact names as they appear (e.g., "Alice", "backend-api",
     "PROJ-3078", "Redis", "memxcore"). Omit if no specific entity is mentioned.
5. Return ONLY a valid JSON array. No explanation, no markdown fences.
6. CRITICAL: User preferences, communication style, corrections, and feedback on agent behaviour
   ALWAYS go to user_model — never to episodic, even if phrased as past events.
   Examples that MUST be user_model:
   - "user prefers concise responses" (not episodic)
   - "user corrected approach: don't mock the database" (not episodic)
   - "user wants root cause explained before accepting a fix" (not episodic)
   - "user prefers discussing in Chinese, coding in English" (not episodic)
   When in doubt between user_model and episodic for anything about the user's preferences
   or working style, always choose user_model.

Additionally, look for BEHAVIORAL PATTERNS — recurring ways the user works, asks questions,
or responds that are NOT explicitly stated preferences. These go into user_model with
content prefixed "habit:". Examples:
- "habit: user asks for root cause before accepting a fix" (observed, not stated)
- "habit: user discusses in Chinese then switches to English for code"
- "habit: user validates design decisions by questioning edge cases"
Only include a habit if you see clear evidence in the notes — do not infer from single occurrences.

7. If the fact describes a relationship between named entities (person→role, person→project,
   tool→project, etc.), also extract "triples" — structured (subject, predicate, object) with
   an optional "when" date (ISO format). Only extract clear, factual relationships.
   Examples:
   - "Alice joined frontend team in March" → {{"s":"Alice","p":"joined","o":"frontend-team","when":"2026-03-01"}}
   - "Redis chosen for session storage" → {{"s":"Redis","p":"used_for","o":"session-storage"}}
   Omit "triples" if no entity relationship is present.
8. IMPORTANT: Each raw note starts with "# [TIMESTAMP]". Extract the timestamp
   closest to each fact and return it as "occurred_at" (ISO 8601 format).
   If a fact spans multiple notes, use the earliest timestamp.
   If no timestamp is found, use null.

Format: [{{"category": "<id>", "content": "<fact>", "tags": ["<topic>"], "entities": ["<name>"], "occurred_at": "<ISO timestamp or null>", "triples": [{{"s":"<subj>","p":"<pred>","o":"<obj>","when":"<date or null>"}}]}}]

Raw notes:
{content}
"""


def distill_with_llm(
    content: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Call Claude to distill RECENT.md into a structured list of facts.
    Returns [{category, content, tags}, ...] or an empty list on failure.
    """
    categories = config.get("memory", {}).get("categories", _default_categories())
    prompt = _build_distill_prompt(content, categories)

    raw = utils.call_llm(prompt, config, max_tokens=4096)
    if not raw:
        return []

    items = utils.parse_llm_json(raw)
    if not isinstance(items, list):
        return []
    return [
        i for i in items
        if isinstance(i, dict) and i.get("category") and i.get("content")
    ]


# ── L1→L2 promotion helpers ──────────────────────────────────────────────────

def _word_overlap(a: str, b: str) -> float:
    """Word overlap ratio between two text segments (Jaccard-like)."""
    words_a = set(re.findall(r'\w+', a.lower()))
    words_b = set(re.findall(r'\w+', b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def _write_to_user_permanent(manager: "object", content: str, timestamp: str) -> None:
    """Write a fact to USER.md (L2), using the same format as archive for _split_archive_sections parsing."""
    user_path = getattr(manager, "user_path", None)
    if not user_path:
        return
    utils.ensure_file(user_path)
    entry = f"\n\n## [{timestamp}]\n\n{content.strip()}\n"
    write_lock = getattr(manager, "_write_lock", None)
    if write_lock:
        with write_lock:
            with open(user_path, "a", encoding="utf-8") as f:
                f.write(entry)
    else:
        with open(user_path, "a", encoding="utf-8") as f:
            f.write(entry)


def _maybe_promote(
    manager: "object",
    archive_dir: str,
    category: str,
    new_content: str,
    tags: List[str],
    timestamp: str,
    rag: "object | None" = None,
    promote_threshold: int = 3,
    similarity_threshold: float = 0.55,
) -> None:
    """
    LSM L1->L2 promote:
    If the new fact has word overlap >= similarity_threshold with an existing archive fact,
    treat it as a "reinforcement signal" and increment that archive entry's confidence_level by 1.
    When confidence_level >= promote_threshold, promote that fact to USER.md (L2).

    Only promotes user_model category (behavioral norms, user preferences).
    Other categories (project_state, etc.) decay quickly and are not suitable for permanent memory.
    """
    if category != "user_model":
        return

    archive_path = os.path.join(archive_dir, f"{category}.md")
    if not os.path.isfile(archive_path):
        return

    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            existing = f.read()
    except OSError:
        return

    front_matter, body = utils.parse_front_matter(existing)
    if not front_matter:
        return

    sections = _split_archive_sections(body)

    for ts, content in sections:
        if _word_overlap(new_content, content) >= similarity_threshold:
            # Reinforcement signal: bump confidence
            confidence = int(front_matter.get("confidence_level", 1)) + 1
            front_matter["confidence_level"] = confidence

            # Update archive front matter
            front_yaml = yaml.safe_dump(front_matter, allow_unicode=True).strip()
            rendered = f"---\n{front_yaml}\n---\n\n{body.lstrip()}"
            try:
                with open(archive_path, "w", encoding="utf-8") as f:
                    f.write(rendered)
            except OSError:
                pass

            # Reached threshold -> promote to L2
            if confidence >= promote_threshold:
                _write_to_user_permanent(manager, content, timestamp)
                if rag is not None:
                    try:
                        rag.upsert(content, "permanent", tags, timestamp)
                    except Exception:
                        pass
            break

    return


# ── Archive pruning ──────────────────────────────────────────────────────────

# If word overlap between new and existing fact exceeds this threshold, treat as duplicate -> remove old, keep new
_DEDUP_THRESHOLD = 0.65

# project_state entries older than this many days are considered "settled", demoted to episodic rather than deleted
# Old does not mean unimportant; demotion preserves historical context without hard delete
_PROJECT_STATE_DEMOTE_DAYS = 90


def _prune_duplicate_sections(
    archive_dir: str,
    category: str,
    new_content: str,
) -> None:
    """
    Prune duplicates: before writing a new fact, remove old facts in archive
    that have high overlap with it. Keep the newer (about to be written) version,
    remove the superseded old version.
    """
    archive_path = os.path.join(archive_dir, f"{category}.md")
    if not os.path.isfile(archive_path):
        return

    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            existing = f.read()
    except OSError:
        return

    front_matter, body = utils.parse_front_matter(existing)
    sections = _split_archive_sections(body)
    kept = [(ts, c) for ts, c in sections if _word_overlap(new_content, c) < _DEDUP_THRESHOLD]

    if len(kept) == len(sections):
        return  # No duplicates, no rewrite needed

    new_body = "".join(f"\n\n## [{ts}]\n\n{c}\n" for ts, c in kept)
    front_yaml = yaml.safe_dump(front_matter or {}, allow_unicode=True).strip()
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(f"---\n{front_yaml}\n---\n\n{new_body.lstrip()}")


def _demote_stale_project_state(archive_dir: str) -> None:
    """
    Demote stale project_state: entries older than _PROJECT_STATE_DEMOTE_DAYS days
    are moved from project_state to episodic (preserving historical context, no hard delete).

    Design principle: old does not mean unimportant.
    - project_state represents "currently in progress"; once settled it becomes "what happened"
    - episodic is the correct home for these historical decisions
    - user_model / domain / references are never cleaned up (knowledge doesn't expire with time)
    """
    now = datetime.utcnow()
    ps_path = os.path.join(archive_dir, "project_state.md")
    if not os.path.isfile(ps_path):
        return

    try:
        with open(ps_path, "r", encoding="utf-8") as f:
            existing = f.read()
    except OSError:
        return

    front_matter, body = utils.parse_front_matter(existing)
    sections = _split_archive_sections(body)

    kept, to_demote = [], []
    for ts, c in sections:
        try:
            age = (now - datetime.fromisoformat(ts)).days
            if age > _PROJECT_STATE_DEMOTE_DAYS:
                to_demote.append((ts, c))
            else:
                kept.append((ts, c))
        except (ValueError, TypeError):
            kept.append((ts, c))

    if not to_demote:
        return

    # Write back project_state (only keep non-stale entries)
    new_body = "".join(f"\n\n## [{ts}]\n\n{c}\n" for ts, c in kept)
    front_yaml = yaml.safe_dump(front_matter or {}, allow_unicode=True).strip()
    try:
        with open(ps_path, "w", encoding="utf-8") as f:
            f.write(f"---\n{front_yaml}\n---\n\n{new_body.lstrip()}")
    except OSError:
        return

    # Demote to episodic
    ep_path = os.path.join(archive_dir, "episodic.md")
    utils.ensure_file(ep_path)
    try:
        with open(ep_path, "a", encoding="utf-8") as f:
            for ts, c in to_demote:
                f.write(f"\n\n## [{ts}]\n\n{c}\n")
    except OSError:
        pass


# ── Category archive writer ───────────────────────────────────────────────────

def _write_to_category_archive(
    archive_dir: str,
    category: str,
    content: str,
    tags: List[str],
    timestamp: str,
) -> None:
    """Write a single distilled fact to the corresponding category archive/*.md.

    timestamp: the section header timestamp — ideally the original event time (occurred_at),
               falls back to compact time (distilled_at) when unavailable.
    """
    archive_path = os.path.join(archive_dir, f"{category}.md")
    utils.ensure_file(archive_path)

    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            existing = f.read()
    except OSError:
        existing = ""

    front_matter, body = utils.parse_front_matter(existing)
    # last_distilled reflects when compaction actually ran, not the event time
    compact_time = datetime.utcnow().isoformat()
    if not front_matter:
        front_matter = {
            "topic": category,
            "tags": tags,
            "last_distilled": compact_time,
            "confidence_level": 1,
        }
        body = ""
    else:
        merged = list(set(front_matter.get("tags", [])) | set(tags))
        front_matter["tags"] = merged
        front_matter["last_distilled"] = compact_time

    # Section header uses the original event timestamp
    new_body = body + f"\n\n## [{timestamp}]\n\n{content.strip()}\n"

    front_yaml = yaml.safe_dump(front_matter, allow_unicode=True).strip()
    rendered = f"---\n{front_yaml}\n---\n\n{new_body.lstrip()}"
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(rendered)


# ── Basic fallback ────────────────────────────────────────────────────────────

def summarize_recent(content: str, max_lines: int = 50) -> str:
    """
    Fallback when no LLM is available: take the first N lines.
    """
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    return (
        "\n".join(lines[:max_lines])
        + "\n... (truncated; set strategy=llm for full distillation)\n"
    )


def _write_basic_fallback(manager: "object", content: str, distilled_at: str) -> None:
    """Basic strategy: write truncated results to archive/general.md."""
    general_path = os.path.join(manager.archive_dir, "general.md")
    utils.ensure_file(general_path)

    try:
        with open(general_path, "r", encoding="utf-8") as f:
            existing = f.read()
    except OSError:
        existing = ""

    front_matter, body = utils.parse_front_matter(existing)
    if not front_matter:
        front_matter = {
            "topic": "general",
            "tags": ["general"],
            "last_distilled": distilled_at,
            "confidence_level": 3,
        }
        body = ""
    else:
        front_matter["last_distilled"] = distilled_at

    chunk = summarize_recent(content)
    new_body = body + f"\n\n## [{distilled_at}]\n\n{chunk}\n"

    front_yaml = yaml.safe_dump(front_matter, allow_unicode=True).strip()
    rendered = f"---\n{front_yaml}\n---\n\n{new_body.lstrip()}"
    with open(general_path, "w", encoding="utf-8") as f:
        f.write(rendered)


def _clear_recent(manager: "object") -> None:
    with open(manager.recent_path, "w", encoding="utf-8") as f:
        f.write("")


# ── BM25 rebuild helper ───────────────────────────────────────────────────────

def _rebuild_bm25(manager: "object") -> None:
    """Rebuild BM25 index after compact completes. Silently skips on failure."""
    bm25 = getattr(manager, "bm25", None)
    if bm25 is not None and hasattr(bm25, "rebuild"):
        try:
            bm25.rebuild(manager.archive_dir, getattr(manager, "user_path", None))
        except Exception:
            pass


# ── Background compaction job ─────────────────────────────────────────────────

# Per-tenant lock: compaction tasks for different tenants don't block each other
_compact_locks: Dict[str, threading.Lock] = {}
_compact_locks_guard = threading.Lock()


def _get_compact_lock(tenant_id: "Optional[str]" = None) -> threading.Lock:
    """Get or create the compaction lock for the specified tenant."""
    key = tenant_id or "__default__"
    if key not in _compact_locks:
        with _compact_locks_guard:
            if key not in _compact_locks:
                _compact_locks[key] = threading.Lock()
    return _compact_locks[key]


def _run_compaction_job(
    manager: "object",
    snapshot: str,
    config: Dict[str, Any],
    snapshot_path: str,
) -> None:
    """
    Execute distillation in a background thread; delete snapshot file on completion.
    On failure, restore snapshot content to the beginning of RECENT.md.
    Dual-write: each distilled fact is written to both archive/*.md and RAG vector store.
    """
    distilled_at = datetime.utcnow().isoformat()
    strategy = (config.get("compaction") or {}).get("strategy", "llm")
    rag = getattr(manager, "rag", None)

    try:
        success = False

        if strategy == "llm":
            items = distill_with_llm(snapshot, config)
            if items:
                for item in items:
                    category = utils.sanitize_category(
                        item.get("category", "episodic")
                    )
                    content = item.get("content", "").strip()
                    tags = item.get("tags", [])
                    entities = item.get("entities", [])
                    # Use original event timestamp from RECENT.md when available
                    raw_ts = item.get("occurred_at")
                    occurred_at = raw_ts if (raw_ts and raw_ts != "null") else distilled_at
                    # Merge tags + entities for indexing (deduplicated, preserve original case)
                    all_tags = list(dict.fromkeys(tags + entities))
                    if content:
                        # ── Prune duplicates: remove old overlapping facts before write ──
                        _prune_duplicate_sections(
                            manager.archive_dir, category, content
                        )
                        # ── Dual-write ────────────────────────────────────
                        _write_to_category_archive(
                            manager.archive_dir, category, content, all_tags, occurred_at
                        )
                        if rag is not None:
                            rag.upsert(content, category, all_tags, occurred_at)
                        # ── Knowledge Graph triple extraction ─────────
                        kg = getattr(manager, "kg", None)
                        if kg is not None:
                            for triple in item.get("triples", []):
                                s = triple.get("s", "").strip()
                                p = triple.get("p", "").strip()
                                o = triple.get("o", "").strip()
                                w = triple.get("when")
                                if s and p and o:
                                    try:
                                        kg.add_triple(s, p, o, valid_from=w, source="compact")
                                    except Exception:
                                        pass
                        # L1->L2 promote check (user_model only)
                        _maybe_promote(
                            manager, manager.archive_dir, category,
                            content, all_tags, occurred_at, rag
                        )
                # ── Demote stale project_state -> episodic (no delete) ────
                _demote_stale_project_state(manager.archive_dir)
                manager.update_index()
                # ── BM25 rebuild ──────────────────────────────────────
                _rebuild_bm25(manager)
                success = True

        if not success:
            # LLM not configured / failed -> basic fallback (no RAG write, content unstructured)
            _write_basic_fallback(manager, snapshot, distilled_at)
            manager.update_index()
            _rebuild_bm25(manager)
            success = True

    except Exception:
        logger.warning("Compaction job failed", exc_info=True)
        success = False

    if not success:
        # Distillation failed: restore snapshot to beginning of RECENT.md to prevent data loss
        try:
            existing = ""
            if os.path.isfile(manager.recent_path):
                with open(manager.recent_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            with open(manager.recent_path, "w", encoding="utf-8") as f:
                f.write(snapshot + existing)
        except OSError:
            pass

    # Always delete snapshot file to prevent orphaned duplicates on next startup
    try:
        os.remove(snapshot_path)
    except OSError:
        pass


# ── Main entry point ──────────────────────────────────────────────────────────

def maybe_compact_recent(manager: "object", force: bool = False) -> "threading.Thread | None":
    """
    Check if RECENT.md has reached the compaction threshold; if so, trigger async compaction
    in the background.

    Optimizations:
    1. Write counter: only read file and estimate tokens every check_interval writes, avoiding I/O on every call
    2. Min entries: skip when entry count is insufficient, avoiding LLM calls for a single memory
    3. Async execution: clear RECENT.md immediately after threshold is reached (non-blocking for caller),
                        distillation runs in background thread; snapshot file ensures recovery on failure
    4. Module lock: only one compaction task at a time per tenant, duplicate triggers are skipped

    strategy=llm (default):
      Claude API distillation -> write to archive/<category>.md by category
      -> auto fallback to basic on LLM failure

    strategy=basic:
      Truncate to first N lines -> write to archive/general.md
    """
    comp_cfg = (getattr(manager, "config", {}) or {}).get("compaction", {})
    threshold = int(comp_cfg.get("threshold_tokens", 2000))
    min_entries = int(comp_cfg.get("min_entries", 3))
    check_interval = int(comp_cfg.get("check_interval", 5))

    # ── 1. Write counter: reduce I/O frequency ─────────────────────────
    write_lock = getattr(manager, "_write_lock", None)
    if write_lock:
        with write_lock:
            manager._write_counter = getattr(manager, "_write_counter", 0) + 1
            counter = manager._write_counter
    else:
        manager._write_counter = getattr(manager, "_write_counter", 0) + 1
        counter = manager._write_counter
    if not force and counter % check_interval != 0:
        return

    # ── 2. Read file ──────────────────────────────────────────────────
    try:
        with open(manager.recent_path, "r", encoding="utf-8") as f:
            recent_content = f.read()
    except OSError:
        return

    if not recent_content.strip():
        return

    # ── 3. Min entries guard ─────────────────────────────────────────
    entry_count = recent_content.count("# [")
    if not force and entry_count < min_entries:
        return

    # ── 4. Token threshold ───────────────────────────────────────────
    if not force and _estimate_tokens(recent_content) < threshold:
        return

    # ── 5. Compaction already running -> skip (per-tenant lock) ────────
    tenant_id = getattr(manager, "tenant_id", None)
    lock = _get_compact_lock(tenant_id)
    if not lock.acquire(blocking=False):
        return

    # ── 6. Take snapshot, immediately clear RECENT.md (non-blocking) ──
    # Hold _write_lock during snapshot+clear to prevent concurrent remember()
    # from appending between snapshot read and RECENT.md clear (data loss).
    # Also acquire fcntl file lock to protect against cross-process writers
    # (e.g. auto_remember hook running in a separate process).
    write_lock_snap = getattr(manager, "_write_lock", None)
    _snap_lock_held = False
    snapshot_path = manager.recent_path + ".snapshot"
    try:
        if write_lock_snap:
            write_lock_snap.acquire()
            _snap_lock_held = True
        # Re-read under lock to capture any writes since the initial read
        # Use fcntl file lock for cross-process safety
        with open(manager.recent_path, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                snapshot = f.read()
                if not snapshot.strip():
                    lock.release()
                    return
                with open(snapshot_path, "w", encoding="utf-8") as sf:
                    sf.write(snapshot)
                f.seek(0)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        lock.release()
        return
    finally:
        if _snap_lock_held:
            try:
                write_lock_snap.release()
                _snap_lock_held = False
            except RuntimeError:
                pass

    # ── 6b. Append raw snapshot to daily journal (permanent archival) ──
    # Use snapshot capture date (not write-time) to avoid midnight boundary issues
    _snap_date = datetime.utcnow().strftime("%Y-%m-%d")
    _append_to_journal(manager, snapshot, snapshot_time=_snap_date)

    config = getattr(manager, "config", {}) or {}

    # ── 7. Execute distillation in background thread ──────────────────
    def _job() -> None:
        try:
            _run_compaction_job(manager, snapshot, config, snapshot_path)
        finally:
            lock.release()

    thread_name = f"memxcore-compact-{tenant_id or 'default'}"
    t = threading.Thread(target=_job, daemon=True, name=thread_name)
    t.start()
    return t
