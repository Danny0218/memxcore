"""
memxcore — UserPromptSubmit hook

On every user message, perform a lightweight memory search and inject into Claude context.
Uses direct file scanning + keyword matching without loading an embedding model (< 0.5s).
Full RAG search is left to the MCP search tool.

Input (stdin):  JSON  { "user_prompt": "..." }
Output (stdout): JSON { "systemMessage": "..." } or {}
Exit: always 0 (never block user input)

Environment variables (optional):
  MEMXCORE_WORKSPACE (or MEMNEST_* / CLAWDMEMORY_WORKSPACE) — workspace root (containing memxcore/ subdirectory)
  MEMXCORE_SCORE_THRESHOLD (or MEMNEST_* / CLAWDMEMORY_SCORE_THRESHOLD)
  MEMXCORE_MAX_RESULTS (or MEMNEST_* / CLAWDMEMORY_MAX_RESULTS)
"""

import json
import os
import re
import sys

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
SCORE_THRESHOLD = float(
    os.environ.get("MEMXCORE_SCORE_THRESHOLD")
    or os.environ.get("MEMNEST_SCORE_THRESHOLD")
    or os.environ.get("CLAWDMEMORY_SCORE_THRESHOLD")
    or "0.20"
)
MAX_RESULTS = int(
    os.environ.get("MEMXCORE_MAX_RESULTS")
    or os.environ.get("MEMNEST_MAX_RESULTS")
    or os.environ.get("CLAWDMEMORY_MAX_RESULTS")
    or "5"
)
MIN_QUERY_LENGTH = 5


def _install_dir(ws: str) -> str:
    new_root = os.path.join(ws, "memxcore")
    mid_root = os.path.join(ws, "memnest")
    old_root = os.path.join(ws, "ClawdMemory")
    if os.path.isdir(os.path.join(new_root, "storage")):
        return new_root
    if os.path.isdir(os.path.join(mid_root, "storage")):
        return mid_root
    if os.path.isdir(os.path.join(old_root, "storage")):
        return old_root
    return new_root


_INSTALL = _install_dir(WORKSPACE)

if TENANT_ID:
    STORAGE_DIR = os.path.join(
        _INSTALL, "tenants", TENANT_ID, "storage"
    )
else:
    STORAGE_DIR = os.path.join(_INSTALL, "storage")
ARCHIVE_DIR = os.path.join(STORAGE_DIR, "archive")
USER_MD = os.path.join(STORAGE_DIR, "USER.md")
RECENT_MD = os.path.join(STORAGE_DIR, "RECENT.md")


def _empty():
    json.dump({}, sys.stdout)
    sys.stdout.flush()


def _split_sections(body):
    """
    Split archive/USER.md body into individual facts.
    Supports multiple formats:
      - ## [timestamp]           (standard archive format)
      - ## Distilled Chunk [ts]  (basic compaction format)
      - # [timestamp] Memory     (RECENT.md raw format)
    """
    # Split using a generic pattern: any line starting with # or ## that contains [...]
    parts = re.split(
        r"^#{1,2}\s+(?:.*?\[([^\]]+)\].*|Memory)", body, flags=re.MULTILINE
    )
    results = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            content = parts[i + 1].strip()
        else:
            content = ""
        # Also scan split substrings directly for meaningful text
        if content:
            # Filter out paragraphs that are only whitespace or frontmatter remnants
            cleaned = re.sub(r"^#.*$", "", content, flags=re.MULTILINE).strip()
            if cleaned:
                results.append(cleaned)
    # Fallback: if section splitting found nothing, treat the entire body as one fact
    if not results:
        stripped = body.strip()
        if stripped and len(stripped) > 10:
            results.append(stripped)
    return results


def _tokenize(text):
    """Simple tokenization: Chinese characters individually, English by whitespace and punctuation."""
    tokens = set()
    for word in re.findall(r"[\w]+", text.lower()):
        tokens.add(word)
    # Also add Chinese characters (each character as a separate token)
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.add(ch)
    return tokens


def _score(query_tokens, fact_text):
    """Compute the token overlap score between query and fact."""
    if not query_tokens:
        return 0.0
    fact_tokens = _tokenize(fact_text)
    if not fact_tokens:
        return 0.0
    overlap = query_tokens & fact_tokens
    if not overlap:
        return 0.0
    # Jaccard-like: overlap relative to query size (recall-oriented)
    return len(overlap) / len(query_tokens)


def _collect_facts():
    """Scan USER.md + archive/*.md + RECENT.md and return [(category, content), ...]."""
    facts = []

    # L2: USER.md (permanent memory, highest priority)
    if os.path.isfile(USER_MD):
        try:
            with open(USER_MD, "r", encoding="utf-8") as f:
                for content in _split_sections(f.read()):
                    facts.append(("permanent", content))
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
                # Skip YAML front matter
                fm_match = re.match(r"^---\n.*?\n---\n", raw, re.DOTALL)
                body = raw[fm_match.end():] if fm_match else raw
                for content in _split_sections(body):
                    facts.append((category, content))
            except OSError:
                continue

    # L0: RECENT.md (latest uncompacted memories)
    if os.path.isfile(RECENT_MD):
        try:
            with open(RECENT_MD, "r", encoding="utf-8") as f:
                for content in _split_sections(f.read()):
                    facts.append(("recent", content))
        except OSError:
            pass

    return facts


def main():
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        _empty()
        return

    user_prompt = input_data.get("user_prompt", "").strip()
    if len(user_prompt) < MIN_QUERY_LENGTH:
        _empty()
        return

    facts = _collect_facts()
    if not facts:
        _empty()
        return

    query_tokens = _tokenize(user_prompt)

    scored = []
    for category, content in facts:
        s = _score(query_tokens, content)
        if s >= SCORE_THRESHOLD:
            scored.append((s, category, content))

    if not scored:
        _empty()
        return

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:MAX_RESULTS]

    lines = ["<memory-context>"]
    for s, cat, content in top:
        display = content.replace("\n", " ").strip()
        if len(display) > 300:
            display = display[:297] + "..."
        lines.append(f"[score:{s:.2f} cat:{cat}] {display}")
    lines.append("</memory-context>")

    json.dump({"systemMessage": "\n".join(lines)}, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _empty()
    finally:
        sys.exit(0)
