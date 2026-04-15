"""
memxcore — Stop hook: auto-compact

Lightweight gatekeeper for compaction. Checks RECENT.md entry count
and only invokes the full compact CLI when the threshold is reached.
Avoids loading MemoryManager/chromadb on every Stop (~5ms when skipping).

Input (stdin):  JSON (ignored — Stop hook payload has no useful fields)
Output (stdout): ignored
Exit: always 0 (never block the user)

Condition: compact when RECENT.md has >= 30 entries (# [ headers).
"""

import json
import os
import subprocess
import sys

# ── Config ──────────────────────────────────────────────────────────────────

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
MIN_ENTRIES = int(os.environ.get("MEMXCORE_COMPACT_MIN_ENTRIES", "30"))


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
RECENT_MD = os.path.join(STORAGE_DIR, "RECENT.md")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Drain stdin (Stop hook sends JSON payload)
    try:
        json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        pass

    # Quick check: does RECENT.md have enough entries?
    if not os.path.isfile(RECENT_MD):
        return

    try:
        with open(RECENT_MD, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return

    entry_count = content.count("# [")
    if entry_count < MIN_ENTRIES:
        return

    # Threshold reached — invoke compact
    python = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = WORKSPACE
    try:
        subprocess.run(
            [python, "-m", "memxcore.cli", "compact"],
            env=env,
            timeout=120,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    finally:
        sys.exit(0)
