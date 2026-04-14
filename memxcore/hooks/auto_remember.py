"""
memxcore - Stop hook: auto-remember

Fires after every Claude response. Reads the transcript, extracts the last
N user+assistant exchanges, and calls LLM to extract memorable facts.

Input (stdin):  JSON  { "transcript_path": "/path/to/session.jsonl", ... }
Output (stdout): ignored (this is an async-compatible hook)
Exit: always 0 (never block the user)

Environment variables (optional):
  MEMXCORE_WORKSPACE (or MEMNEST_* / CLAWDMEMORY_*) - workspace root (parent of memxcore/)
  MEMXCORE_TENANT_ID (or MEMNEST_TENANT_ID / CLAWDMEMORY_TENANT_ID)
  MEMXCORE_AUTO_REMEMBER (or MEMNEST_* / CLAWDMEMORY_*) - set to "0" to disable
"""

import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# -- Config --------------------------------------------------------------------

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

MIN_USER_CHARS = 6
MIN_EXCHANGE_CHARS = 50
MAX_CONTEXT_CHARS = 4000
MAX_PRIOR_CONTEXT_CHARS = 2000
NUM_EXCHANGES = 3


def _install_dir(ws: str) -> str:
    new_root = os.path.join(ws, "memxcore")
    memx_root = os.path.join(ws, "memx")
    mid_root = os.path.join(ws, "memnest")
    old_root = os.path.join(ws, "ClawdMemory")
    if os.path.isdir(os.path.join(new_root, "storage")):
        return new_root
    if os.path.isdir(os.path.join(memx_root, "storage")):
        return memx_root
    if os.path.isdir(os.path.join(mid_root, "storage")):
        return mid_root
    if os.path.isdir(os.path.join(old_root, "storage")):
        return old_root
    return new_root


_INSTALL = _install_dir(WORKSPACE)

if TENANT_ID:
    STORAGE_DIR = os.path.join(_INSTALL, "tenants", TENANT_ID, "storage")
else:
    STORAGE_DIR = os.path.join(_INSTALL, "storage")
RECENT_MD = os.path.join(STORAGE_DIR, "RECENT.md")
CONFIG_PATH = os.path.join(_INSTALL, "config.yaml")


# -- Transcript parsing --------------------------------------------------------

def _read_last_lines(path: str, max_lines: int = 500) -> List[str]:
    """Read last N lines of a file efficiently."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[-max_lines:]
    except OSError:
        return []


def _extract_text(content: Any) -> str:
    """Extract text from a message content field (string or content block array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def parse_last_exchanges(transcript_path: str, n: int = NUM_EXCHANGES) -> List[Tuple[str, str]]:
    """
    Parse transcript .jsonl and extract the last N user+assistant exchanges.
    Returns list of (user_text, assistant_text) in chronological order, or [].
    """
    lines = _read_last_lines(transcript_path)
    if not lines:
        return []

    messages: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg_type = obj.get("type")
        if msg_type in ("user", "assistant"):
            messages.append(obj)

    if not messages:
        return []

    exchanges: List[Tuple[str, str]] = []
    seen_uuids: set = set()

    for msg in reversed(messages):
        if len(exchanges) >= n:
            break
        if msg.get("type") != "assistant":
            continue

        assistant_text = _extract_text(msg.get("message", {}).get("content", []))
        if not assistant_text.strip():
            continue

        msg_uuid = msg.get("uuid")
        if msg_uuid and msg_uuid in seen_uuids:
            continue
        if msg_uuid:
            seen_uuids.add(msg_uuid)

        # Find the user message it responds to (via parentUuid)
        parent_uuid = msg.get("parentUuid")
        user_msg = None
        if parent_uuid:
            for m in reversed(messages):
                if m.get("type") == "user" and m.get("uuid") == parent_uuid:
                    user_msg = m
                    break

        # Fallback: last user message before this assistant
        if not user_msg:
            idx = messages.index(msg)
            for i in range(idx - 1, -1, -1):
                if messages[i].get("type") == "user":
                    user_msg = messages[i]
                    break

        if not user_msg:
            continue

        user_text = _extract_text(user_msg.get("message", {}).get("content", ""))
        exchanges.append((user_text, assistant_text))

    # Return in chronological order
    exchanges.reverse()
    return exchanges


# -- LLM extraction ------------------------------------------------------------

def _build_extract_prompt(exchanges: List[Tuple[str, str]]) -> str:
    n = len(exchanges)
    exchange_text = ""
    for i, (user_text, assistant_text) in enumerate(exchanges, 1):
        # Give more context budget to the latest exchange
        if i == n:
            max_chars = MAX_CONTEXT_CHARS
        else:
            max_chars = MAX_PRIOR_CONTEXT_CHARS
        exchange_text += f"\n--- Exchange {i}/{n} ---\n"
        exchange_text += f"User: {user_text[:max_chars]}\n\n"
        exchange_text += f"Assistant: {assistant_text[:max_chars]}\n"

    return (
        "You are a memory extraction assistant. Given recent user-assistant exchanges, "
        "extract facts worth remembering for future sessions. "
        "Earlier exchanges provide context; focus extraction on the latest exchange.\n\n"
        "EXTRACT these types of facts:\n"
        "- User corrections or feedback on the assistant's approach\n"
        "- User preferences or working style\n"
        "- Technical decisions with rationale\n"
        "- Project state changes (task completed, feature shipped, bug found)\n"
        "- User identity info (role, expertise, background)\n"
        "- Important external references shared (URLs, tools, systems)\n\n"
        "DO NOT extract:\n"
        "- Transient debugging details\n"
        "- Code content or file paths (derivable from codebase)\n"
        "- Things already obvious from the conversation context\n"
        "- Trivial acknowledgments or greetings\n\n"
        'Return ONLY a JSON array: [{"content": "<fact>", "category": "<cat>"}]\n'
        "Categories: user_model, domain, project_state, episodic, references\n"
        "Return [] if nothing is worth remembering.\n"
        f"{exchange_text}"
    )


def _load_config() -> Dict[str, Any]:
    if not os.path.isfile(CONFIG_PATH):
        return {}
    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def extract_facts(exchanges: List[Tuple[str, str]], config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Call LLM to extract memorable facts from recent exchanges."""
    prompt = _build_extract_prompt(exchanges)

    # Import via package path (workspace = parent of memxcore/)
    sys.path.insert(0, WORKSPACE)
    from memxcore.core.utils import call_llm, parse_llm_json

    raw = call_llm(prompt, config, max_tokens=1024)
    if not raw:
        return []

    items = parse_llm_json(raw)
    if not isinstance(items, list):
        return []

    return [
        item for item in items
        if isinstance(item, dict) and item.get("content")
    ]


# -- Write to RECENT.md -------------------------------------------------------

def write_facts(facts: List[Dict[str, str]]) -> int:
    """Append extracted facts to RECENT.md. Returns number written."""
    if not facts:
        return 0

    os.makedirs(os.path.dirname(RECENT_MD), exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    lines = []
    for fact in facts:
        content = fact.get("content", "").strip()
        category = fact.get("category", "")
        if not content:
            continue
        cat_hint = f" [category={category}]" if category else ""
        lines.append(f"\n\n# [{timestamp}] Memory{cat_hint}\n{content}\n")

    if not lines:
        return 0

    with open(RECENT_MD, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.writelines(lines)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    return len(lines)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    # Kill switch
    _ar = (
        os.environ.get("MEMXCORE_AUTO_REMEMBER")
        or os.environ.get("MEMX_AUTO_REMEMBER")
        or os.environ.get("MEMNEST_AUTO_REMEMBER")
        or os.environ.get("CLAWDMEMORY_AUTO_REMEMBER")
    )
    if _ar == "0":
        return

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    transcript_path = input_data.get("transcript_path", "")
    if not transcript_path or not os.path.isfile(transcript_path):
        return

    exchanges = parse_last_exchanges(transcript_path)
    if not exchanges:
        return

    # Pre-filter on the latest exchange
    latest_user, latest_assistant = exchanges[-1]
    if len(latest_user) < MIN_USER_CHARS:
        return
    if len(latest_user) + len(latest_assistant) < MIN_EXCHANGE_CHARS:
        return

    config = _load_config()
    facts = extract_facts(exchanges, config)
    write_facts(facts)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    finally:
        sys.exit(0)
