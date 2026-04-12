"""
Cursor ``stop`` hook — run RECENT compaction.

Cursor's stop payload does not include Claude-style ``transcript_path``, so we
cannot reuse ``auto_remember`` here. This matches the second command in the
Claude Code Stop hook chain (``memxcore compact``).

stdin:  JSON with at least ``hook_event_name`` (``stop``) and ``status``.
stdout: ``{}`` (informational; Cursor treats stop as non-blocking audit)
exit:   always 0 (never block the agent)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from typing import Any, Dict


def _should_compact(payload: Dict[str, Any]) -> bool:
    if payload.get("hook_event_name") != "stop":
        return False
    # Align with Claude: compact after every agent turn (including errors).
    return payload.get("status") in ("completed", "aborted", "error", None)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError, OSError):
        print("{}")
        return

    if not _should_compact(payload):
        print("{}")
        return

    try:
        from memxcore.cli import cmd_compact

        tid = (
            os.environ.get("MEMXCORE_TENANT_ID")
            or os.environ.get("MEMX_TENANT_ID")
            or os.environ.get("MEMNEST_TENANT_ID")
            or os.environ.get("CLAWDMEMORY_TENANT_ID")
            or None
        )
        if isinstance(tid, str) and not tid.strip():
            tid = None

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cmd_compact(tenant_id=tid)
    except Exception:
        pass
    print("{}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.exit(0)
