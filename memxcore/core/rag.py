"""
RAGIndex — ChromaDB + sentence-transformers semantic search layer.

Design principles:
- MD files are the source of truth; ChromaDB is a rebuildable search index
- If chromadb / sentence-transformers are not installed, graceful degradation to keyword search
- All public methods silently return empty values when RAG is unavailable; no exceptions raised
- Uses an internal lock to protect ChromaDB concurrent reads/writes (compaction background thread vs main thread search)
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from . import utils
from .parsers import _fact_id, _split_archive_sections  # noqa: F401

logger = logging.getLogger(__name__)


# ── RAGIndex ───────────────────────────────────────────────────────────────────

class RAGIndex:
    """
    Wrapper around ChromaDB vector index.

    Main interface:
        upsert(content, category, tags, distilled_at)  — write/update a single fact
        search(query, top_k, category)                 — semantic search
        rebuild(archive_dir)                           — rebuild entire index from MD files
        available                                      — whether RAG is available (depends on installed packages)
    """

    def __init__(
        self,
        storage_dir: str,
        config: Dict[str, Any],
        collection_name: str = "memxcore",
    ) -> None:
        self._available = False
        self._client = None
        self._collection = None
        self._lock = threading.Lock()
        self._config = config
        self._collection_name = collection_name
        self._chroma_dir = os.path.join(storage_dir, "chroma")
        self._try_init()

    def _try_init(self) -> None:
        """Initialize ChromaDB + embedding model. Silently skip on failure."""
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError:
            return

        rag_cfg = self._config.get("rag", {})
        model_name = rag_cfg.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")

        try:
            # Suppress noisy model loading reports from safetensors/transformers
            import io, sys
            _orig_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
            finally:
                sys.stderr = _orig_stderr

            self._client = chromadb.PersistentClient(path=self._chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
        except Exception:
            self._available = False

    def close(self) -> None:
        """Release ChromaDB client resources (background threads, file handles)."""
        with self._lock:
            self._collection = None
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
            self._available = False

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    def upsert(
        self,
        content: str,
        category: str,
        tags: List[str],
        distilled_at: str,
    ) -> None:
        """Write or update a distilled fact. Repeated writes with the same content are idempotent."""
        if not self._available or not content.strip():
            return
        fact_id = _fact_id(category, content)
        try:
            with self._lock:
                self._collection.upsert(
                    ids=[fact_id],
                    documents=[content.strip()],
                    metadatas=[{
                        "category": category,
                        "tags": ",".join(tags),
                        "distilled_at": distilled_at,
                    }],
                )
        except Exception:
            logger.warning("RAG upsert failed", exc_info=True)

    def search(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
        tag_filter: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search, returns [{content, category, tags, distilled_at, score}, ...].
        score is in [0, 1]; higher means more relevant (cosine similarity).
        tag_filter: if provided, only search facts whose tags field contains this string ($contains).
        after/before: ISO 8601 date strings for time-range filtering (inclusive).
        Returns [] when the collection is empty or RAG is unavailable.
        """
        if not self._available:
            return []
        try:
            with self._lock:
                count = self._collection.count()
                if count == 0:
                    return []
                n = min(top_k, count)
                kwargs: Dict[str, Any] = dict(query_texts=[query], n_results=n)
                where_clauses = []
                if category:
                    where_clauses.append({"category": category})
                if tag_filter:
                    where_clauses.append({"tags": {"$contains": tag_filter}})
                # Note: ChromaDB $gte/$lte only works on int/float, not strings.
                # Time filtering is applied post-query below.
                if len(where_clauses) == 1:
                    kwargs["where"] = where_clauses[0]
                elif len(where_clauses) > 1:
                    kwargs["where"] = {"$and": where_clauses}
                results = self._collection.query(**kwargs)

            out = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                ts = meta.get("distilled_at", "")
                # Post-query time filtering (ChromaDB $gte/$lte doesn't support strings)
                if after and ts and ts < after:
                    continue
                if before:
                    b = before if "T" in before else before + "T23:59:59.999999"
                    if ts and ts > b:
                        continue
                out.append({
                    "content": doc,
                    "category": meta.get("category", ""),
                    "tags": [t for t in meta.get("tags", "").split(",") if t],
                    "distilled_at": ts,
                    "score": round(1.0 - distance, 4),
                })
            return out
        except Exception:
            logger.warning("RAG search failed", exc_info=True)
            return []

    def reindex_file(self, archive_path: str, category: str) -> int:
        """
        Incrementally reindex a single archive file.
        First deletes all old vectors for the category, then re-embeds the current content.
        Called after a file is edited; much faster than a full rebuild.
        Returns the number of reindexed entries.
        """
        if not self._available:
            return 0

        # Delete all old vectors for this category
        try:
            with self._lock:
                existing = self._collection.get(where={"category": category})
                if existing["ids"]:
                    self._collection.delete(ids=existing["ids"])
        except Exception:
            logger.warning("RAG reindex_file: failed to delete old vectors for %s", category, exc_info=True)

        # Parse and re-embed current file content
        try:
            with open(archive_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return 0

        front_matter, body = utils.parse_front_matter(raw)
        file_tags = front_matter.get("tags", []) if front_matter else []
        count = 0
        for distilled_at, content in _split_archive_sections(body):
            if content.strip():
                self.upsert(content.strip(), category, file_tags, distilled_at)
                count += 1
        return count

    def rebuild(self, archive_dir: str, user_path: Optional[str] = None) -> int:
        """
        Clear the entire collection and re-embed all distilled facts from
        archive/*.md and USER.md into ChromaDB.
        Returns the number of successfully written entries.
        """
        if not self._available:
            return 0

        # Clear existing collection to remove stale vectors from deleted/renamed files
        try:
            with self._lock:
                existing = self._collection.get()
                if existing["ids"]:
                    self._collection.delete(ids=existing["ids"])
        except Exception:
            logger.warning("RAG rebuild: failed to clear existing collection", exc_info=True)

        count = 0

        # Re-embed USER.md (L2 permanent memories)
        if user_path and os.path.isfile(user_path):
            try:
                with open(user_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                for distilled_at, content in _split_archive_sections(raw):
                    if content.strip():
                        self.upsert(content.strip(), "permanent", [], distilled_at)
                        count += 1
            except OSError:
                pass

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
            front_matter, body = utils.parse_front_matter(raw)
            file_tags = front_matter.get("tags", []) if front_matter else []
            for distilled_at, content in _split_archive_sections(body):
                if content.strip():
                    self.upsert(content.strip(), category, file_tags, distilled_at)
                    count += 1
        return count
