"""
parsers — Zero-dependency section parsers for memxcore markdown files.

Shared by rag.py, bm25.py, and other modules that need to split
archive/*.md, USER.md, or RECENT.md into individual fact sections.
"""

import hashlib
import re
from typing import List, Tuple


# ── Fact ID ──────────────────────────────────────────────────────────────────

def _fact_id(category: str, content: str) -> str:
    """Generate a stable ID from category + content, used for upsert deduplication."""
    h = hashlib.md5(content.strip().encode("utf-8")).hexdigest()[:12]
    return f"{category}_{h}"


# ── Archive section parser ───────────────────────────────────────────────────

def _split_archive_sections(body: str) -> List[Tuple[str, str]]:
    """
    Split the body of archive/*.md or USER.md files by '## [timestamp]' sections.
    Returns [(distilled_at, content), ...]
    """
    parts = re.split(r'^## \[([^\]]+)\]', body, flags=re.MULTILINE)
    # parts = [pre-text, ts1, body1, ts2, body2, ...]
    result = []
    for i in range(1, len(parts) - 1, 2):
        distilled_at = parts[i]
        content = parts[i + 1].strip()
        if content:
            result.append((distilled_at, content))
    return result


# ── RECENT.md section parser ─────────────────────────────────────────────────

# Matches both RECENT.md header formats:
#   # [2026-04-14T05:12:31] [category:episodic] Memory
#   # [2026-04-14T05:30:47+00:00] Memory [category=user_model]
_RECENT_HEADER_RE = re.compile(
    r"^# \[([^\]]+)\]\s*"              # timestamp
    r"(?:"
    r"\[category:([^\]]*)\]\s*Memory"  # old format: [category:xxx] Memory
    r"|"
    r"Memory\s*\[category=([^\]]*)\]"  # new format: Memory [category=xxx]
    r"|"
    r"(?:\[category:([^\]]*)\])?\s*Memory"  # minimal: just Memory (no category)
    r")\s*$",
    re.MULTILINE,
)


def _split_recent_sections(body: str) -> List[Tuple[str, str, str]]:
    """
    Split RECENT.md into individual memory entries.

    Supports two header formats:
      # [timestamp] [category:xxx] Memory     (older format)
      # [timestamp] Memory [category=xxx]     (newer format)

    Returns [(timestamp, category, content), ...]
    Category defaults to "recent" when not specified.
    """
    matches = list(_RECENT_HEADER_RE.finditer(body))
    if not matches:
        # Fallback: treat the entire body as one entry
        stripped = body.strip()
        if stripped and len(stripped) > 10:
            return [("", "recent", stripped)]
        return []

    result = []
    for idx, m in enumerate(matches):
        timestamp = m.group(1)
        # Category can be in group 2 (old format), 3 (new format), or 4 (minimal)
        category = m.group(2) or m.group(3) or m.group(4) or "recent"

        # Content runs from end of this header to start of next header (or EOF)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()

        if content:
            result.append((timestamp, category, content))

    return result
