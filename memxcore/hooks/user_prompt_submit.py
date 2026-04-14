"""
memxcore — UserPromptSubmit hook (Claude Code) / beforeSubmitPrompt (Cursor)

On every user message, perform a lightweight memory search and inject context.
Uses BM25 ranking + Knowledge Graph entity lookup + RECENT.md keyword scan.
Full RAG search is left to the MCP search tool.

Input (stdin):  JSON with ``user_prompt`` (Claude) and/or ``prompt`` (Cursor).
Output (stdout): JSON ``{ "systemMessage": "..." }`` or ``{}``
Exit: always 0 (never block user input)

Search pipeline:
  1. Parse user prompt, filter trivial inputs
  2. BM25 search over archive/*.md + USER.md (via rank_bm25 library)
  3. RECENT.md keyword scan (separate, scored at 0.75)
  4. Knowledge Graph entity lookup (ticket IDs, PascalCase, proper nouns)
  5. Merge, deduplicate by content hash, apply category boosts
  6. BM25 threshold filtering (>= 30% of top score) before merge;
     RECENT/KG bypass threshold (own quality gates: keyword overlap / entity match)
  7. Output dynamic 0-3 results, 600 chars/result, 2400 chars total max

Environment variables (optional):
  MEMXCORE_WORKSPACE — workspace root (containing memxcore/ subdirectory)
  MEMXCORE_SCORE_THRESHOLD — relative threshold (default 0.30)
  MEMXCORE_MAX_RESULTS — max results to inject (default 3)
  MEMXCORE_HOOK_DEBUG — set to "1" for stderr trace output
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time

# ── Configuration ────────────────────────────────────────────────────────────

_DEFAULT_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE = (
    os.environ.get("MEMXCORE_WORKSPACE")
    or os.environ.get("MEMX_WORKSPACE")
    or os.environ.get("MEMNEST_WORKSPACE")
    or os.environ.get("CLAWDMEMORY_WORKSPACE")
    or _DEFAULT_WS
)
TENANT_ID = (
    os.environ.get("MEMXCORE_TENANT_ID")
    or os.environ.get("MEMX_TENANT_ID")
    or os.environ.get("MEMNEST_TENANT_ID")
    or os.environ.get("CLAWDMEMORY_TENANT_ID", None)
)

RELATIVE_THRESHOLD = float(
    os.environ.get("MEMXCORE_SCORE_THRESHOLD")
    or os.environ.get("MEMX_SCORE_THRESHOLD")
    or os.environ.get("MEMNEST_SCORE_THRESHOLD")
    or os.environ.get("CLAWDMEMORY_SCORE_THRESHOLD")
    or "0.30"
)
MAX_RESULTS = int(
    os.environ.get("MEMXCORE_MAX_RESULTS")
    or os.environ.get("MEMX_MAX_RESULTS")
    or os.environ.get("MEMNEST_MAX_RESULTS")
    or os.environ.get("CLAWDMEMORY_MAX_RESULTS")
    or "3"
)
DEBUG = os.environ.get("MEMXCORE_HOOK_DEBUG", "") == "1"

MIN_QUERY_TOKENS = 2
MAX_CHARS_PER_RESULT = 600
MAX_TOTAL_CHARS = 2400

# Category boost multipliers (for ranking only, not for display)
_CATEGORY_BOOST = {
    "permanent": 1.3,
    "recent": 1.15,
    "user_model": 1.1,
    "domain": 1.0,
    "project_state": 1.0,
    "general": 1.0,
    "episodic": 0.9,
    "references": 0.85,
}

# Trivial prompts to skip
_SKIPLIST = frozenset({
    "ok", "okay", "yes", "no", "y", "n", "thanks", "thank you", "thx", "ty",
    "sure", "got it", "right", "yep", "nope", "done", "next", "continue",
    "go", "go ahead", "proceed", "lgtm", "ack",
    # Chinese
    "\u597d", "\u55ef", "\u5631", "\u662f", "\u4e0d", "\u5c0d",
    "\u8b1d\u8b1d", "\u597d\u7684", "\u53ef\u4ee5", "\u7e7c\u7e8c",
    "\u61c9\u8a72", "\u4e86\u89e3", "\u6536\u5230",
    # Japanese
    "\u306f\u3044", "\u3044\u3044\u3048", "\u308f\u304b\u308a\u307e\u3057\u305f",
})


# ── Storage paths ────────────────────────────────────────────────────────────

def _install_dir(ws):
    for name in ("memxcore", "memx", "memnest", "ClawdMemory"):
        candidate = os.path.join(ws, name)
        if os.path.isdir(os.path.join(candidate, "storage")):
            return candidate
    return os.path.join(ws, "memxcore")


_INSTALL = _install_dir(WORKSPACE)

if TENANT_ID:
    STORAGE_DIR = os.path.join(_INSTALL, "tenants", TENANT_ID, "storage")
else:
    STORAGE_DIR = os.path.join(_INSTALL, "storage")
ARCHIVE_DIR = os.path.join(STORAGE_DIR, "archive")
USER_MD = os.path.join(STORAGE_DIR, "USER.md")
RECENT_MD = os.path.join(STORAGE_DIR, "RECENT.md")
KG_DB_PATH = os.path.join(STORAGE_DIR, "knowledge.db")


# ── Debug helper ─────────────────────────────────────────────────────────────

def _dbg(msg):
    if DEBUG:
        print(f"[memxcore-hook] {msg}", file=sys.stderr)


# ── Self-contained utility functions (duplicated to avoid import chains) ─────
# These are small functions also in parsers.py / bm25.py.
# Duplicated here so the hook never triggers __init__.py -> MemoryManager -> chromadb.

_CJK_RANGES = (
    ("\u4e00", "\u9fff"),    # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),    # CJK Extension A
    ("\uf900", "\ufaff"),    # CJK Compatibility Ideographs
)


def _is_cjk(ch):
    for lo, hi in _CJK_RANGES:
        if lo <= ch <= hi:
            return True
    return False


def _tokenize(text):
    """Mixed CJK/English tokenization with CJK bigrams."""
    tokens = []
    cjk_chars = []
    for ch in text:
        if _is_cjk(ch):
            tokens.append(ch)
            cjk_chars.append(ch)
    # CJK bigrams for compound terms
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    tokens.extend(re.findall(r"[a-z0-9][-a-z0-9_.]*[a-z0-9]|[a-z0-9]+", text.lower()))
    return tokens


def _split_archive_sections(body):
    """Split archive/*.md or USER.md by '## [timestamp]' sections."""
    parts = re.split(r'^## \[([^\]]+)\]', body, flags=re.MULTILINE)
    result = []
    for i in range(1, len(parts) - 1, 2):
        distilled_at = parts[i]
        content = parts[i + 1].strip()
        if content:
            result.append((distilled_at, content))
    return result


# Matches both RECENT.md header formats:
#   # [timestamp] [category:xxx] Memory
#   # [timestamp] Memory [category=xxx]
_RECENT_HEADER_RE = re.compile(
    r"^# \[([^\]]+)\]\s*"
    r"(?:"
    r"\[category:([^\]]*)\]\s*Memory"
    r"|"
    r"Memory\s*\[category=([^\]]*)\]"
    r"|"
    r"(?:\[category:([^\]]*)\])?\s*Memory"
    r")\s*$",
    re.MULTILINE,
)


def _split_recent_sections(body):
    """Split RECENT.md into [(timestamp, category, content), ...]."""
    matches = list(_RECENT_HEADER_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        if stripped and len(stripped) > 10:
            return [("", "recent", stripped)]
        return []
    result = []
    for idx, m in enumerate(matches):
        timestamp = m.group(1)
        category = m.group(2) or m.group(3) or m.group(4) or "recent"
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        if content:
            result.append((timestamp, category, content))
    return result


def _fact_id(content):
    """Content hash for deduplication."""
    return hashlib.md5(content.strip().encode("utf-8")).hexdigest()[:16]


def _skip_front_matter(raw):
    """Strip YAML front matter from an archive file."""
    raw = raw.lstrip()
    m = re.match(r"^---\n.*?\n---\n?", raw, re.DOTALL)
    return raw[m.end():] if m else raw


# ── Entity extraction (inline, no memory_manager import) ─────────────────────

_TICKET_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_PROPER_NOUN_RE = re.compile(r"(?<!\.\s)\b[A-Z][a-z]{1,20}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_ENTITY_STOPWORDS = frozenset({
    "The", "This", "That", "What", "When", "Where", "Which", "Who", "How",
    "Are", "Was", "Were", "Will", "Can", "Could", "Should", "Would", "May",
    "Not", "But", "And", "For", "With", "From", "Into", "About",
    "Also", "Just", "All", "Any", "Each", "Every", "Some", "Most",
    "Here", "There", "Then", "Now", "Yes", "No",
    "Let", "Use", "Get", "Set", "Run", "Add", "New", "Old",
    "Has", "Had", "Have", "Does", "Did", "Done",
    "True", "False", "None", "Null",
    "Memory", "Category",
})


def _extract_entities(text):
    """Extract entity candidates from text for KG lookup."""
    entities = []
    seen = set()

    for m in _TICKET_RE.finditer(text):
        e = m.group()
        if e not in seen:
            entities.append(e)
            seen.add(e)

    for m in _CAMEL_RE.finditer(text):
        e = m.group()
        if e not in seen and e not in _ENTITY_STOPWORDS:
            entities.append(e)
            seen.add(e)

    for m in _PROPER_NOUN_RE.finditer(text):
        e = m.group()
        if e not in seen and e not in _ENTITY_STOPWORDS:
            entities.append(e)
            seen.add(e)

    for m in _ACRONYM_RE.finditer(text):
        e = m.group()
        if len(e) > 2 and e not in seen and e not in _ENTITY_STOPWORDS:
            entities.append(e)
            seen.add(e)

    return entities


# ── Output helpers ───────────────────────────────────────────────────────────

def _empty():
    json.dump({}, sys.stdout)
    sys.stdout.flush()


# ── BM25 search (with fallback to weighted keyword matching) ─────────────────

def _collect_bm25_corpus():
    """
    Scan USER.md + archive/*.md and return:
      documents: [{"content": ..., "category": ..., "distilled_at": ...}, ...]
      tokenized_corpus: [[token, ...], ...]
    """
    documents = []
    tokenized_corpus = []

    # L2: USER.md (permanent memory, highest priority)
    if os.path.isfile(USER_MD):
        try:
            with open(USER_MD, "r", encoding="utf-8") as f:
                raw = f.read()
            raw = _skip_front_matter(raw)

            # Capture preamble: content before the first ## [timestamp] section.
            # USER.md has structured sections like ## 核心要求, ## 工作模式 that
            # use plain ## headers (no timestamp in brackets). These would be
            # invisible to _split_archive_sections which only matches ## [ts].
            first_ts = re.search(r'^## \[[^\]]+\]', raw, re.MULTILINE)
            preamble = raw[:first_ts.start()] if first_ts else raw
            if preamble.strip():
                # Split preamble by ## headers into separate indexable facts
                preamble_parts = re.split(r'^## ', preamble, flags=re.MULTILINE)
                for part in preamble_parts:
                    part = part.strip()
                    if part:
                        documents.append({
                            "content": part,
                            "category": "permanent",
                            "distilled_at": "preamble",
                        })
                        tokenized_corpus.append(_tokenize(part))

            for distilled_at, content in _split_archive_sections(raw):
                content = content.strip()
                if content:
                    documents.append({
                        "content": content,
                        "category": "permanent",
                        "distilled_at": distilled_at,
                    })
                    tokenized_corpus.append(_tokenize(content))
        except OSError:
            pass

    # L1: archive/*.md
    if os.path.isdir(ARCHIVE_DIR):
        for name in sorted(os.listdir(ARCHIVE_DIR)):
            if not name.endswith(".md"):
                continue
            category = name[:-3]
            path = os.path.join(ARCHIVE_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except OSError:
                continue
            body = _skip_front_matter(raw)
            for distilled_at, content in _split_archive_sections(body):
                content = content.strip()
                if content:
                    documents.append({
                        "content": content,
                        "category": category,
                        "distilled_at": distilled_at,
                    })
                    tokenized_corpus.append(_tokenize(content))

    return documents, tokenized_corpus


def _bm25_search(query_tokens, documents, tokenized_corpus, top_k=10):
    """
    BM25 search using rank_bm25 library.
    Returns [(score, category, content), ...] sorted by score descending.
    Score is normalized: highest = 1.0.
    Falls back to weighted keyword matching if rank_bm25 is not available.
    """
    if not documents or not query_tokens:
        return []

    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)
    except ImportError:
        _dbg("rank_bm25 not available, falling back to weighted keyword matching")
        scores = _fallback_keyword_scores(query_tokens, tokenized_corpus)

    # Collect top_k non-zero scores
    indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
    indexed.sort(key=lambda x: x[1], reverse=True)
    indexed = indexed[:top_k]

    if not indexed:
        return []

    max_score = indexed[0][1]
    results = []
    for idx, score in indexed:
        doc = documents[idx]
        norm_score = round(score / max_score, 4) if max_score > 0 else 0.0
        results.append((norm_score, doc["category"], doc["content"]))

    return results


def _fallback_keyword_scores(query_tokens, tokenized_corpus):
    """
    Weighted keyword matching fallback when rank_bm25 is not installed.
    Longer tokens are worth more (reduces noise from single-char CJK matches).
    """
    # Build weighted query token set
    weighted_query = {}
    for t in query_tokens:
        weight = min(len(t), 5)  # longer tokens worth more, cap at 5
        weighted_query[t] = max(weighted_query.get(t, 0), weight)

    total_query_weight = sum(weighted_query.values())
    if total_query_weight == 0:
        return [0.0] * len(tokenized_corpus)

    scores = []
    for doc_tokens in tokenized_corpus:
        doc_set = set(doc_tokens)
        matched_weight = sum(w for t, w in weighted_query.items() if t in doc_set)
        scores.append(matched_weight / total_query_weight)
    return scores


# ── RECENT.md search ─────────────────────────────────────────────────────────

def _search_recent(query_tokens):
    """
    Scan RECENT.md for keyword matches.
    Returns [(score, category, content), ...] with score scaled by overlap ratio
    (range 0.50-0.75).
    """
    if not os.path.isfile(RECENT_MD):
        return []
    try:
        with open(RECENT_MD, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return []

    if not raw.strip():
        return []

    sections = _split_recent_sections(raw)
    if not sections:
        return []

    query_set = set(query_tokens)
    results = []

    for _ts, category, content in sections:
        content = content.strip()
        if not content:
            continue
        fact_tokens = set(_tokenize(content))
        overlap = query_set & fact_tokens
        if len(overlap) >= MIN_QUERY_TOKENS:
            overlap_ratio = len(overlap) / max(len(query_set), 1)
            score = 0.50 + 0.25 * overlap_ratio  # Range: 0.50 to 0.75
            results.append((round(score, 4), category or "recent", content))

    return results


# ── Knowledge Graph search ───────────────────────────────────────────────────

def _search_kg(query):
    """
    Extract entities from query, look them up in KG (read-only SQLite).
    Returns [(score, "kg", formatted_triple_text), ...]. Max 2 results.
    """
    if not os.path.isfile(KG_DB_PATH):
        return []

    entities = _extract_entities(query)
    _dbg(f"KG entities extracted: {entities}")

    if not entities:
        return []

    try:
        conn = sqlite3.connect(f"file:{KG_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except (sqlite3.Error, OSError):
        _dbg("KG: failed to open database")
        return []

    results = []
    seen_triples = set()
    max_kg_results = 2

    try:
        for entity in entities:
            if len(results) >= max_kg_results:
                break

            # Exact entity match (subject or object)
            rows = conn.execute(
                "SELECT * FROM triples "
                "WHERE (subject = ? COLLATE NOCASE OR object = ? COLLATE NOCASE) "
                "AND ended IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (entity, entity, max_kg_results - len(results)),
            ).fetchall()

            if not rows:
                # Fallback: fuzzy search
                escaped = entity.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                pattern = f"%{escaped}%"
                rows = conn.execute(
                    "SELECT * FROM triples "
                    "WHERE (subject LIKE ? ESCAPE '\\' OR object LIKE ? ESCAPE '\\') "
                    "AND ended IS NULL "
                    "ORDER BY created_at DESC LIMIT ?",
                    (pattern, pattern, max_kg_results - len(results)),
                ).fetchall()

            for row in rows:
                if len(results) >= max_kg_results:
                    break
                triple_key = (row["subject"], row["predicate"], row["object"])
                if triple_key in seen_triples:
                    continue
                seen_triples.add(triple_key)

                # Format triple as readable text
                validity = ""
                if row["valid_from"]:
                    validity = f" (from {row['valid_from']})"
                text = f"{row['subject']} -> {row['predicate']} -> {row['object']}{validity}"
                results.append((0.70, "kg", text))

            _dbg(f"KG entity '{entity}': {len(rows)} rows found")

    except sqlite3.Error as e:
        _dbg(f"KG query error: {e}")
    finally:
        conn.close()

    return results


# ── Merge, deduplicate, rank, filter ─────────────────────────────────────────

def _merge_and_rank(bm25_results, recent_results, kg_results):
    """
    Merge all result sources, deduplicate by content hash, apply category boosts,
    filter by relative threshold, and return final ranked list.

    Threshold filtering strategy:
    - BM25 results: filtered by relative threshold (>= 30% of top normalized score)
      before merge. Since BM25 scores are normalized (top = 1.0), this effectively
      means >= RELATIVE_THRESHOLD (0.30).
    - RECENT results: bypass BM25 threshold — they have their own quality gate
      (MIN_QUERY_TOKENS overlap check + variable scoring by overlap ratio).
    - KG results: bypass BM25 threshold — they are entity-matched (exact or fuzzy)
      which is already a strong quality signal.

    Returns [(boosted_score, category, content), ...].
    """
    # Filter BM25 results by relative threshold BEFORE merge.
    # Since BM25 scores are normalized (top = 1.0), min_bm25 = RELATIVE_THRESHOLD.
    if bm25_results:
        min_bm25 = RELATIVE_THRESHOLD * bm25_results[0][0]  # top is always 1.0
        bm25_filtered = [(s, cat, c) for s, cat, c in bm25_results if s >= min_bm25]
    else:
        bm25_filtered = []

    all_results = []
    all_results.extend(bm25_filtered)
    all_results.extend(recent_results)  # own quality gate: keyword overlap check
    all_results.extend(kg_results)      # own quality gate: entity match

    if not all_results:
        return []

    # Deduplicate by content hash
    seen_hashes = set()
    unique = []
    for score, category, content in all_results:
        h = _fact_id(content)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        unique.append((score, category, content))

    # Apply category boost for ranking
    boosted = []
    for score, category, content in unique:
        boost = _CATEGORY_BOOST.get(category, 1.0)
        boosted.append((score * boost, category, content))

    # Sort by boosted score
    boosted.sort(key=lambda x: x[0], reverse=True)

    return boosted[:MAX_RESULTS]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        _empty()
        return

    user_prompt = (input_data.get("prompt") or input_data.get("user_prompt") or "").strip()

    # Filter trivial prompts
    if user_prompt.lower().strip().rstrip("!.?") in _SKIPLIST:
        _dbg(f"skipped trivial prompt: {user_prompt!r}")
        _empty()
        return

    query_tokens = _tokenize(user_prompt)
    if len(query_tokens) < MIN_QUERY_TOKENS:
        _dbg(f"skipped: only {len(query_tokens)} tokens")
        _empty()
        return

    # 1. BM25 search over archive + USER.md
    documents, tokenized_corpus = _collect_bm25_corpus()
    _dbg(f"BM25 corpus: {len(documents)} facts loaded")

    bm25_results = _bm25_search(query_tokens, documents, tokenized_corpus, top_k=10)
    _dbg(f"BM25 results: {len(bm25_results)}")
    if DEBUG and bm25_results:
        for i, (s, cat, c) in enumerate(bm25_results[:10]):
            _dbg(f"  BM25[{i}] score={s:.3f} cat={cat} {c[:60]}...")

    # 2. RECENT.md keyword scan
    recent_results = _search_recent(query_tokens)
    _dbg(f"RECENT results: {len(recent_results)}")

    # 3. Knowledge Graph entity lookup
    kg_results = _search_kg(user_prompt)
    _dbg(f"KG results: {len(kg_results)}")

    # 4. Merge, deduplicate, rank, filter
    final = _merge_and_rank(bm25_results, recent_results, kg_results)
    _dbg(f"After merge+filter: {len(final)} results")

    if not final:
        _dbg(f"no results passed threshold (RELATIVE_THRESHOLD={RELATIVE_THRESHOLD})")
        _empty()
        elapsed = time.time() - t0
        _dbg(f"total time: {elapsed:.3f}s")
        return

    # 5. Format output (no score numbers, category tag only)
    lines = ["<memory-context>"]
    total_chars = 0
    for _score, category, content in final:
        display = content.replace("\n", " ").strip()
        if len(display) > MAX_CHARS_PER_RESULT:
            display = display[:MAX_CHARS_PER_RESULT - 3] + "..."
        if total_chars + len(display) > MAX_TOTAL_CHARS:
            # Truncate to fit total budget
            remaining = MAX_TOTAL_CHARS - total_chars
            if remaining > 50:
                display = display[:remaining - 3] + "..."
            else:
                break
        lines.append(f"[{category}] {display}")
        total_chars += len(display)
        _dbg(f"  injected [{category}] {display[:80]}...")
    lines.append("</memory-context>")

    elapsed = time.time() - t0
    _dbg(f"total time: {elapsed:.3f}s")

    json.dump({"systemMessage": "\n".join(lines)}, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if DEBUG:
            print(f"[memxcore-hook] FATAL: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        _empty()
    finally:
        sys.exit(0)
