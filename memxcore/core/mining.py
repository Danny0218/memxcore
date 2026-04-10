"""
Mining — Batch import conversation history into memxcore.

Supported formats:
- Claude Code JSONL (~/.claude/projects/*/  *.jsonl)
- Markdown files (.md)
- Plain text files (.txt)
- JSON conversation exports (.json, must contain a messages array)

Pipeline: read file -> extract conversation text -> chunk -> LLM distill -> write to archive/RAG/BM25/KG
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from . import utils
from .compaction import distill_with_llm, _write_to_category_archive, _prune_duplicate_sections

logger = logging.getLogger("memxcore.mining")

# Conservative estimate: ~3000 tokens per chunk for LLM distill prompt + content
_CHUNK_MAX_TOKENS = 3000


def _estimate_tokens(text: str) -> int:
    count = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            count += 1
    count += len(re.findall(r"[a-zA-Z0-9]+", text))
    return max(count, len(text) // 4)


# ── File parsers ─────────────────────────────────────────────────────────────

def _parse_claude_jsonl(path: str) -> str:
    """Parse a Claude Code JSONL conversation file, extracting user/assistant text."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            msg_type = obj.get("type", "")
            if msg_type not in ("user", "assistant", "human"):
                continue
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "\n".join(t for t in texts if t.strip())
            elif isinstance(content, str):
                text = content
            else:
                continue
            if text.strip():
                role = "User" if msg_type in ("user", "human") else "Assistant"
                lines.append(f"[{role}] {text.strip()}")
    return "\n\n".join(lines)


def _parse_json_messages(path: str) -> str:
    """Parse a JSON conversation export (e.g. ChatGPT export), extracting the messages array."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data if isinstance(data, list) else data.get("messages", [])
    lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", msg.get("author", {}).get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", str(p)) for p in content if isinstance(p, (str, dict))
            )
        if content and role in ("user", "assistant", "human", "system"):
            lines.append(f"[{role.capitalize()}] {content.strip()}")
    return "\n\n".join(lines)


def _parse_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _detect_parser(path: str) -> Callable[[str], str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return _parse_claude_jsonl
    elif ext == ".json":
        return _parse_json_messages
    else:
        return _parse_text


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_tokens: int = _CHUNK_MAX_TOKENS) -> List[str]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ── Mining pipeline ──────────────────────────────────────────────────────────

def mine_file(
    path: str,
    manager: "object",
    config: Dict[str, Any],
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Import a single file into memxcore.

    Returns:
        {"file": str, "chunks": int, "facts": int, "triples": int, "errors": int}
    """
    parser = _detect_parser(path)
    try:
        text = parser(path)
    except Exception as e:
        return {"file": path, "chunks": 0, "facts": 0, "triples": 0, "errors": 1, "error": str(e)}

    if not text.strip():
        return {"file": path, "chunks": 0, "facts": 0, "triples": 0, "errors": 0}

    chunks = _chunk_text(text)
    total_facts = 0
    total_triples = 0
    errors = 0
    distilled_at = datetime.utcnow().isoformat()

    rag = getattr(manager, "rag", None)
    kg = getattr(manager, "kg", None)

    for i, chunk in enumerate(chunks):
        if on_progress:
            on_progress(f"  chunk {i+1}/{len(chunks)}")

        try:
            items = distill_with_llm(chunk, config)
        except Exception:
            errors += 1
            continue

        if not items:
            errors += 1
            continue

        for item in items:
            category = utils.sanitize_category(
                item.get("category", "episodic")
            )
            content = item.get("content", "").strip()
            tags = item.get("tags", [])
            entities = item.get("entities", [])
            all_tags = list(dict.fromkeys(tags + entities))

            if not content:
                continue

            _prune_duplicate_sections(manager.archive_dir, category, content)
            _write_to_category_archive(
                manager.archive_dir, category, content, all_tags, distilled_at
            )

            if rag is not None:
                try:
                    rag.upsert(content, category, all_tags, distilled_at)
                except Exception:
                    pass

            total_facts += 1

            if kg is not None:
                for triple in item.get("triples", []):
                    s = triple.get("s", "").strip()
                    p = triple.get("p", "").strip()
                    o = triple.get("o", "").strip()
                    w = triple.get("when")
                    if s and p and o:
                        try:
                            kg.add_triple(s, p, o, valid_from=w, source="mining")
                            total_triples += 1
                        except Exception:
                            pass

    # Rebuild BM25 + index.json
    try:
        bm25 = getattr(manager, "bm25", None)
        if bm25 is not None:
            bm25.rebuild(manager.archive_dir, getattr(manager, "user_path", None))
        manager.update_index()
    except Exception:
        pass

    return {
        "file": os.path.basename(path),
        "chunks": len(chunks),
        "facts": total_facts,
        "triples": total_triples,
        "errors": errors,
    }


def mine_directory(
    dir_path: str,
    manager: "object",
    config: Dict[str, Any],
    extensions: Optional[List[str]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Import all supported files in a directory."""
    if extensions is None:
        extensions = [".jsonl", ".json", ".md", ".txt"]

    files = []
    for name in sorted(os.listdir(dir_path)):
        ext = os.path.splitext(name)[1].lower()
        if ext in extensions:
            files.append(os.path.join(dir_path, name))

    results = []
    for i, path in enumerate(files):
        if on_progress:
            on_progress(f"[{i+1}/{len(files)}] {os.path.basename(path)}")
        result = mine_file(path, manager, config, on_progress=on_progress)
        results.append(result)

    return results
