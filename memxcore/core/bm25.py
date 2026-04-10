"""
BM25Index — BM25 keyword search layer.

Design principles:
- Parallel to RAGIndex: RAG handles semantic search, BM25 handles exact keyword matching
- Pure in-memory index, rebuilt from archive/*.md + USER.md (no persistence, rebuild <10ms)
- Graceful degradation when rank_bm25 is not installed (available=False)
- Mixed CJK/English tokenization: CJK characters split individually + English split by word boundaries
"""

import os
import re
import threading
from typing import Any, Dict, List, Optional

from . import utils
from .rag import _split_archive_sections


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_CJK_RANGES = (
    ("\u4e00", "\u9fff"),    # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),    # CJK Extension A
    ("\uf900", "\ufaff"),    # CJK Compatibility Ideographs
)


def _is_cjk(ch: str) -> bool:
    for lo, hi in _CJK_RANGES:
        if lo <= ch <= hi:
            return True
    return False


def _tokenize(text: str) -> List[str]:
    """
    Mixed CJK/English tokenization:
    - CJK characters are split individually (each character becomes one token)
    - English/numeric text is split by word boundaries, all lowercased
    """
    tokens: List[str] = []
    for ch in text:
        if _is_cjk(ch):
            tokens.append(ch)
    tokens.extend(re.findall(r"[a-z0-9][-a-z0-9_.]*[a-z0-9]|[a-z0-9]+", text.lower()))
    return tokens


# ── BM25Index ─────────────────────────────────────────────────────────────────

class BM25Index:
    """
    BM25 keyword index.

    Main interface:
        rebuild(archive_dir, user_path)  — rebuild index from MD files
        search(query, top_k)             — BM25 search
        available                        — whether rank_bm25 is available
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._available = False
        self._bm25 = None
        self._documents: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._config = config
        self._bm25_cls = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_cls = BM25Okapi
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def rebuild(self, archive_dir: str, user_path: Optional[str] = None) -> int:
        """Rebuild the entire BM25 index from archive/*.md + USER.md. Returns the number of indexed facts."""
        if not self._available:
            return 0

        documents: List[Dict[str, Any]] = []
        tokenized_corpus: List[List[str]] = []

        # L2 USER.md takes priority
        if user_path and os.path.isfile(user_path):
            try:
                with open(user_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                for distilled_at, content in _split_archive_sections(raw):
                    content = content.strip()
                    if content:
                        documents.append({
                            "content": content,
                            "category": "permanent",
                            "tags": [],
                            "distilled_at": distilled_at,
                        })
                        tokenized_corpus.append(_tokenize(content))
            except OSError:
                pass

        # L1 archive facts
        if os.path.isdir(archive_dir):
            for name in sorted(os.listdir(archive_dir)):
                if not name.endswith(".md"):
                    continue
                category = name[:-3]
                path = os.path.join(archive_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read()
                except OSError:
                    continue
                _, body = utils.parse_front_matter(raw)
                for distilled_at, content in _split_archive_sections(body):
                    content = content.strip()
                    if content:
                        documents.append({
                            "content": content,
                            "category": category,
                            "tags": [],
                            "distilled_at": distilled_at,
                        })
                        tokenized_corpus.append(_tokenize(content))

        with self._lock:
            self._documents = documents
            if tokenized_corpus:
                self._bm25 = self._bm25_cls(tokenized_corpus)
            else:
                self._bm25 = None

        return len(documents)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25 search, returns [{content, category, tags, distilled_at, score}, ...].
        score is normalized to [0, 1].
        """
        if not self._available:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            if self._bm25 is None or not self._documents:
                return []
            scores = self._bm25.get_scores(query_tokens)

        # Take top_k, filter out score=0
        indexed_scores = [
            (i, s) for i, s in enumerate(scores) if s > 0
        ]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        indexed_scores = indexed_scores[:top_k]

        if not indexed_scores:
            return []

        # Normalize: highest score = 1.0
        max_score = indexed_scores[0][1]
        results = []
        for idx, score in indexed_scores:
            doc = self._documents[idx]
            results.append({
                "content": doc["content"],
                "category": doc["category"],
                "tags": doc["tags"],
                "distilled_at": doc["distilled_at"],
                "score": round(score / max_score, 4) if max_score > 0 else 0.0,
            })
        return results
