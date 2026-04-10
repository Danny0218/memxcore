import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import yaml

logger = logging.getLogger("memxcore")

_TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_CATEGORY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_REMEMBER_BYTES = 1_000_000  # 1MB per remember() call

from . import utils
from .bm25 import BM25Index
from .compaction import maybe_compact_recent
from .knowledge_graph import KnowledgeGraph
from .paths import resolve_install_dir
from .rag import RAGIndex, _split_archive_sections
from .watcher import ArchiveWatcher

# ── Entity extraction patterns ────────────────────────────────────────────────

# Ticket IDs: PROJ-123, TRNSCN-3078
_TICKET_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
# Capitalized proper nouns (2+ chars, not at sentence start heuristic)
_PROPER_NOUN_RE = re.compile(r"(?<!\.\s)\b[A-Z][a-z]{1,20}\b")
# Tech terms often written in PascalCase or camelCase
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
# All-caps acronyms (2-6 chars)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

# Common stop words that look like proper nouns but aren't
_ENTITY_STOPWORDS = {
    "The", "This", "That", "What", "When", "Where", "Which", "Who", "How",
    "Are", "Was", "Were", "Will", "Can", "Could", "Should", "Would", "May",
    "Not", "But", "And", "For", "With", "From", "Into", "About",
}


def _extract_query_entities(query: str) -> List[str]:
    """
    Extract possible entity names from a search query for tag-based filtering.
    Returns entities in original case (since ChromaDB $contains is case-sensitive
    and tags are stored in original case, we return original case for matching).
    """
    entities = set()

    # Ticket IDs (high confidence)
    for m in _TICKET_RE.finditer(query):
        entities.add(m.group())

    # PascalCase / camelCase (high confidence)
    for m in _CAMEL_RE.finditer(query):
        entities.add(m.group())

    # Proper nouns (medium confidence, filter stopwords)
    for m in _PROPER_NOUN_RE.finditer(query):
        word = m.group()
        if word not in _ENTITY_STOPWORDS:
            entities.add(word)

    # ALL_CAPS acronyms > 2 chars (avoid overly broad ones like "AI")
    for m in _ACRONYM_RE.finditer(query):
        if len(m.group()) > 2:
            entities.add(m.group())

    return list(entities)


@dataclass
class MemoryResult:
    content: str
    source: str
    level: int
    relevance_score: float
    metadata: Dict[str, Any]


class MemoryManager:
    """
    Core memory manager for memxcore.

    Search strategy (dual-read):
      1. RAG semantic search (ChromaDB) -- returns precise individual facts with relevance scores
      2. Keyword fallback -- scans archive/*.md when RAG is unavailable or returns no results

    Write strategy (dual-write, during compaction):
      Each distilled fact is written to both archive/<category>.md and ChromaDB
    """

    def __init__(
        self,
        workspace_path: str,
        tenant_id: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if tenant_id is not None and not _TENANT_ID_PATTERN.match(tenant_id):
            raise ValueError(
                f"Invalid tenant_id: {tenant_id!r} "
                "(must match [a-zA-Z0-9_-]+)"
            )

        self.workspace_path = workspace_path
        self.tenant_id = tenant_id
        self.root_dir = resolve_install_dir(workspace_path)

        # tenant_id=None -> existing flat path (backward compatible)
        # tenant_id set  -> tenants/<tenant_id>/storage/
        if tenant_id:
            self.storage_dir = os.path.join(
                self.root_dir, "tenants", tenant_id, "storage"
            )
        else:
            self.storage_dir = os.path.join(self.root_dir, "storage")

        self.archive_dir = os.path.join(self.storage_dir, "archive")
        self.recent_path = os.path.join(self.storage_dir, "RECENT.md")
        self.user_path = os.path.join(self.storage_dir, "USER.md")
        self.index_path = os.path.join(self.storage_dir, "index.json")

        self.config: Dict[str, Any] = self._load_config(config_path)

        # Ensure required directories exist
        os.makedirs(self.archive_dir, exist_ok=True)
        for p in (self.recent_path, self.user_path, self.index_path):
            if not os.path.exists(p):
                if p.endswith(".json"):
                    utils.write_json(p, {"files": []})
                else:
                    utils.ensure_file(p)

        # RAG index (available=False when chromadb not installed, graceful degradation)
        rag_collection = f"memx_{tenant_id}" if tenant_id else "memxcore"
        self.rag = RAGIndex(self.storage_dir, self.config, collection_name=rag_collection)

        # BM25 index (available=False when rank-bm25 not installed, graceful degradation)
        self.bm25 = BM25Index(self.config)
        self.bm25.rebuild(self.archive_dir, self.user_path)

        # Knowledge Graph (SQLite temporal triples)
        self.kg = KnowledgeGraph(self.storage_dir)

        # File watcher: auto-triggers RAG reindex when archive/*.md changes
        # Can be disabled via watch: false in config.yaml
        self.watcher = ArchiveWatcher(self)
        if self.config.get("watch", False):
            self.watcher.start()

        # Write lock for RECENT.md (prevents interleaved concurrent writes)
        self._write_lock = threading.Lock()

        # Startup auto-recovery: compact stale data from previous session that wasn't compacted
        self._stale_checked = False
        self._compact_stale_on_startup()

        # Startup capability report
        self._log_capabilities()

    # -------------------------------------------------
    # Startup auto-recovery
    # -------------------------------------------------

    def _compact_stale_on_startup(self) -> None:
        """
        Auto-recover stale data on MCP server startup:
        1. .snapshot file -> previous compact was interrupted, prepend back to RECENT.md
        2. RECENT.md non-empty -> previous session forgot to compact, force trigger
        """
        try:
            snapshot_path = self.recent_path + ".snapshot"

            # Step 1: Recover orphaned snapshot (previous compact was killed)
            if os.path.isfile(snapshot_path):
                with open(snapshot_path, "r", encoding="utf-8") as f:
                    snapshot_content = f.read()
                if snapshot_content.strip():
                    # Prepend snapshot to the beginning of RECENT.md
                    existing = ""
                    if os.path.isfile(self.recent_path):
                        with open(self.recent_path, "r", encoding="utf-8") as f:
                            existing = f.read()
                    with open(self.recent_path, "w", encoding="utf-8") as f:
                        f.write(snapshot_content + existing)
                os.remove(snapshot_path)

            # Step 2: RECENT.md non-empty -> force compact
            if os.path.isfile(self.recent_path):
                with open(self.recent_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    self.compact(force=True)
        except Exception:
            pass  # Never crash __init__

    def _log_capabilities(self) -> None:
        """Log which search/write capabilities are active at startup."""
        capabilities = []
        degraded = []

        if self.rag.available:
            capabilities.append("RAG semantic search")
        else:
            degraded.append("RAG disabled (chromadb/sentence-transformers not installed)")

        if self.bm25.available:
            capabilities.append("BM25 keyword search")
        else:
            degraded.append("BM25 disabled (rank-bm25 not installed)")

        llm_cfg = self.config.get("llm", {})
        api_key_env = llm_cfg.get("api_key_env", "")
        has_key = bool(api_key_env and os.environ.get(api_key_env))
        if not has_key:
            has_key = any(
                os.environ.get(k)
                for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                           "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_BASE")
            )
        strategy = self.config.get("compaction", {}).get("strategy", "basic")

        if has_key and strategy == "llm":
            capabilities.append("LLM distillation")
        elif not has_key:
            degraded.append(
                "LLM unavailable (no API key found in environment). "
                "Compaction will use basic strategy (truncate, no categorization)"
            )

        if capabilities:
            logger.info("memxcore ready: %s", ", ".join(capabilities))
        if degraded:
            for msg in degraded:
                logger.warning("memxcore: %s", msg)

    def _maybe_compact_stale(self) -> None:
        """
        Defensive check: if RECENT.md hasn't been compacted for stale_minutes
        during search(), auto-trigger compact. Covers the case where MCP server
        has been running for a long time without restart.
        """
        if self._stale_checked:
            return
        self._stale_checked = True

        try:
            if not os.path.isfile(self.recent_path):
                return
            stat = os.stat(self.recent_path)
            if stat.st_size == 0:
                return
            comp_cfg = self.config.get("compaction", {})
            stale_minutes = int(comp_cfg.get("stale_minutes", 10))
            age_minutes = (time.time() - stat.st_mtime) / 60.0
            if age_minutes >= stale_minutes:
                self.compact(force=True)
        except Exception:
            pass

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def remember(
        self,
        text: str,
        level: Optional[int] = None,
        category: Optional[str] = None,
    ) -> str:
        """
        Write a memory entry.
        - level is None / 0: append to RECENT.md (L0), awaiting compact distillation
        - level == 2: write directly to USER.md (L2), permanent storage, not cleared by compact
        - category: optional hint for LLM distillation (e.g. "user_model", "domain")
        """
        if len(text.encode("utf-8")) > _MAX_REMEMBER_BYTES:
            raise ValueError(
                f"Text exceeds maximum size ({_MAX_REMEMBER_BYTES} bytes). "
                "Split into smaller chunks."
            )
        timestamp = datetime.utcnow().isoformat()

        if level == 2:
            # L2 uses the same ## [timestamp] format as archive, so search can parse it
            entry = f"\n\n## [{timestamp}]\n\n{text.strip()}\n"
            with self._write_lock:
                with open(self.user_path, "a", encoding="utf-8") as f:
                    f.write(entry)
            # L2 facts are also written to RAG with category="permanent" for high priority
            try:
                self.rag.upsert(text.strip(), "permanent", [], timestamp)
            except Exception:
                pass
            return "USER.md"
        else:
            cat_tag = f" [category:{category}]" if category else ""
            header = f"\n\n# [{timestamp}]{cat_tag} Memory\n"
            with self._write_lock:
                with open(self.recent_path, "a", encoding="utf-8") as f:
                    f.write(header)
                    f.write(text.strip() + "\n")
            maybe_compact_recent(self)
            return "RECENT.md"

    def search(self, query: str, max_results: int = 10) -> List[MemoryResult]:
        """
        Three-tier search:
        1. Hybrid search (RAG + BM25, RRF fusion) -- with tag-based boosting
        2. LLM relevance judgment -- when hybrid is unavailable or returns no results
        3. Keyword fallback -- last resort
        """
        # ── 0. Stale RECENT.md check (defensive) ────────────────────────
        self._maybe_compact_stale()

        # ── 1. Hybrid search (RAG + BM25, RRF fusion + Tag boost) ──────
        top_k = self.config.get("rag", {}).get("top_k", max_results)
        fetch_k = max(top_k, max_results) * 2

        # Extract entities from query for tag-filtered search
        entities = _extract_query_entities(query)

        rag_hits = []
        if self.rag.available:
            if entities:
                # First try tag-filtered RAG (search once per entity, take union)
                tag_hits: Dict[str, Dict] = {}
                for entity in entities:
                    for hit in self.rag.search(query, top_k=fetch_k, tag_filter=entity):
                        tag_hits.setdefault(hit["content"], hit)
                # Then run unfiltered RAG to supplement results
                for hit in self.rag.search(query, top_k=fetch_k):
                    tag_hits.setdefault(hit["content"], hit)
                rag_hits = list(tag_hits.values())
            else:
                rag_hits = self.rag.search(query, top_k=fetch_k)

        bm25_hits = []
        if self.bm25.available:
            bm25_hits = self.bm25.search(query, top_k=fetch_k)

        # ── 1b. Knowledge Graph supplement ──────────────────────────────
        kg_results = self._kg_search(query, entities)

        if rag_hits or bm25_hits:
            fused = self._rrf_fuse(rag_hits, bm25_hits, max_results)
            if fused:
                # Append KG results after hybrid results (deduplicated)
                if kg_results:
                    seen = {r.content for r in fused}
                    for kr in kg_results:
                        if kr.content not in seen:
                            fused.append(kr)
                return fused[:max_results]

        # ── 2. LLM relevance judgment ─────────────────────────────────
        llm_results = self._llm_search(query, max_results)
        if llm_results:
            return llm_results

        # ── 3. Keyword fallback ────────────────────────────────────────
        kw_results = self._keyword_search(query, max_results)
        # KG results may still have value even when keyword search has no results
        if kg_results and not kw_results:
            return kg_results
        return kw_results

    def update_index(self) -> None:
        """Rebuild index.json (scan archive/*.md, parse YAML front matter)."""
        files_meta: List[Dict[str, Any]] = []
        for name in os.listdir(self.archive_dir):
            if not name.endswith(".md"):
                continue
            full_path = os.path.join(self.archive_dir, name)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except OSError:
                continue

            front_matter, body = utils.parse_front_matter(raw)
            topic = front_matter.get("topic", "")
            tags = front_matter.get("tags", [])
            summary = utils.extract_simple_summary(body)
            rel_path = os.path.relpath(full_path, self.workspace_path)

            files_meta.append({
                "path": rel_path.replace(os.sep, "/"),
                "tags": tags,
                "summary": summary or topic,
                "updated_at": datetime.utcnow().isoformat(),
            })

        utils.write_json(self.index_path, {"files": files_meta})

    def compact(self, force: bool = False) -> None:
        """Explicitly trigger the compaction process."""
        maybe_compact_recent(self, force=force)

    def rebuild_rag_index(self) -> int:
        """
        Rebuild RAG vector index from archive/*.md.
        Use when index is corrupted or deploying to an environment with existing MD files.
        Returns the number of entries rebuilt.
        """
        return self.rag.rebuild(self.archive_dir)

    # -------------------------------------------------
    # Internal utilities
    # -------------------------------------------------

    def _kg_search(self, query: str, entities: List[str]) -> List[MemoryResult]:
        """Query related triples from Knowledge Graph and convert to MemoryResult."""
        try:
            triples = []
            # Query KG using extracted entities
            for entity in entities:
                for t in self.kg.query_entity(entity):
                    triples.append(t)
            # Also do fuzzy search (covers cases where entity extraction misses)
            if not triples:
                triples = self.kg.search(query, limit=5)

            if not triples:
                return []

            # Deduplicate (same triple may be found by multiple entities)
            seen_ids = set()
            results = []
            for t in triples:
                tid = t.get("id")
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                content = KnowledgeGraph.format_triple(t)
                results.append(MemoryResult(
                    content=content,
                    source="knowledge.db",
                    level=1,
                    relevance_score=0.9,
                    metadata={
                        "category": "knowledge_graph",
                        "search": "kg",
                        "subject": t.get("subject", ""),
                        "predicate": t.get("predicate", ""),
                        "object": t.get("object", ""),
                    },
                ))
            return results[:5]  # Max 5 KG results
        except Exception:
            return []

    def _rrf_fuse(
        self,
        rag_hits: List[Dict[str, Any]],
        bm25_hits: List[Dict[str, Any]],
        max_results: int,
    ) -> List[MemoryResult]:
        """Reciprocal Rank Fusion: merge ranked results from RAG and BM25."""
        k = self.config.get("rag", {}).get("rrf_k", 60)

        # content → (rank, hit) for each ranker
        rag_by_content: Dict[str, tuple] = {}
        for rank, hit in enumerate(rag_hits):
            rag_by_content[hit["content"]] = (rank + 1, hit)

        bm25_by_content: Dict[str, tuple] = {}
        for rank, hit in enumerate(bm25_hits):
            bm25_by_content[hit["content"]] = (rank + 1, hit)

        all_contents = set(rag_by_content.keys()) | set(bm25_by_content.keys())
        default_rank = max(len(rag_hits), len(bm25_hits), 1) + 1

        scored = []
        for content in all_contents:
            r_rag = rag_by_content[content][0] if content in rag_by_content else default_rank
            r_bm25 = bm25_by_content[content][0] if content in bm25_by_content else default_rank
            rrf_score = 1.0 / (k + r_rag) + 1.0 / (k + r_bm25)

            # Get metadata (prefer RAG as it has more complete tags)
            hit = (
                rag_by_content[content][1]
                if content in rag_by_content
                else bm25_by_content[content][1]
            )

            methods = []
            if content in rag_by_content:
                methods.append("rag")
            if content in bm25_by_content:
                methods.append("bm25")

            scored.append((rrf_score, content, hit, "+".join(methods)))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rrf_score, content, hit, method in scored[:max_results]:
            results.append(MemoryResult(
                content=content,
                source=f"archive/{hit['category']}.md",
                level=1,
                relevance_score=round(rrf_score, 6),
                metadata={
                    "category": hit["category"],
                    "tags": hit.get("tags", []),
                    "distilled_at": hit.get("distilled_at", ""),
                    "search": method,
                },
            ))
        return results

    def _collect_archive_facts(self) -> List[Tuple[str, str, str]]:
        """
        Scan archive/*.md and USER.md, return all distilled facts.
        USER.md (L2) facts are placed first to ensure LLM search sees them with priority.
        Format: [(category, content, distilled_at), ...]
        """
        facts = []

        # L2 permanent memory first
        if os.path.isfile(self.user_path):
            try:
                with open(self.user_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                for distilled_at, content in _split_archive_sections(raw):
                    if content.strip():
                        facts.append(("permanent", content.strip(), distilled_at))
            except OSError:
                pass

        # L1 archive facts
        for name in sorted(os.listdir(self.archive_dir)):
            if not name.endswith(".md"):
                continue
            category = name[:-3]
            path = os.path.join(self.archive_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except OSError:
                continue
            _, body = utils.parse_front_matter(raw)
            for distilled_at, content in _split_archive_sections(body):
                if content.strip():
                    facts.append((category, content.strip(), distilled_at))
        return facts

    def _llm_search(self, query: str, max_results: int) -> List[MemoryResult]:
        """
        LLM relevance judgment: send archive facts to Claude and let it pick relevant ones.

        Token protection: when facts exceed LLM_SEARCH_MAX_FACTS, pre-filter with keywords
        before sending to LLM, to avoid an oversized prompt.
        """
        LLM_SEARCH_MAX_FACTS = 60

        all_facts = self._collect_archive_facts()
        if not all_facts:
            return []

        # If too many facts, pre-filter with keywords to narrow candidates
        if len(all_facts) > LLM_SEARCH_MAX_FACTS:
            q = query.lower()
            candidates = [f for f in all_facts if q in f[1].lower()]
            # Still too many after pre-filter, truncate
            if len(candidates) > LLM_SEARCH_MAX_FACTS:
                candidates = candidates[:LLM_SEARCH_MAX_FACTS]
            # Pre-filter results too few (< 5), pad to give LLM more context
            if len(candidates) < 5:
                candidates = all_facts[:LLM_SEARCH_MAX_FACTS]
        else:
            candidates = all_facts

        # Group by category, organize into readable prompt sections
        by_category: Dict[str, List[str]] = {}
        for cat, content, _ in candidates:
            by_category.setdefault(cat, []).append(content)

        sections = []
        for cat, facts in by_category.items():
            bullet_lines = "\n".join(f"• {f}" for f in facts)
            sections.append(f"[{cat}]\n{bullet_lines}")
        facts_block = "\n\n".join(sections)

        prompt = f"""You are a memory retrieval assistant. Given a query, find all stored facts that are relevant.

Query: {query}

Stored facts:
{facts_block}

Rules:
1. Return only facts genuinely relevant to the query — do not force matches
2. Copy the fact text exactly as shown above
3. Score: 1.0 = directly answers the query, 0.5 = loosely related
4. Return ONLY a JSON array, empty array if nothing is relevant

Format: [{{"category": "<category>", "content": "<exact fact text>", "score": <0.0-1.0>}}]"""

        raw = utils.call_llm(prompt, self.config, max_tokens=2048)
        if not raw:
            return []

        items = utils.parse_llm_json(raw)
        if not isinstance(items, list):
            return []

        results = []
        for item in items[:max_results]:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            cat = item.get("category", "")
            results.append(MemoryResult(
                content=item["content"],
                source=f"archive/{cat}.md" if cat else "archive",
                level=1,
                relevance_score=float(item.get("score", 0.8)),
                metadata={"category": cat, "search": "llm"},
            ))
        return results

    def _keyword_search(self, query: str, max_results: int) -> List[MemoryResult]:
        """Keyword fallback: check index.json first, full-text scan on insufficient hits. USER.md always scanned first."""
        query_lower = query.lower()
        index_data = utils.read_json(self.index_path) or {"files": []}
        indexed_files = index_data.get("files", [])

        candidate_paths: List[str] = []
        # L2 USER.md always prioritized (independent of index.json)
        if os.path.isfile(self.user_path):
            candidate_paths.append(self.user_path)

        for item in indexed_files:
            tags = [str(t).lower() for t in item.get("tags", [])]
            summary = str(item.get("summary", "")).lower()
            if query_lower in summary or any(query_lower in t for t in tags):
                candidate_paths.append(
                    os.path.join(self.workspace_path, item.get("path", ""))
                )

        if candidate_paths:
            search_targets = candidate_paths
        else:
            # USER.md + archive fallback
            search_targets = [self.user_path] if os.path.isfile(self.user_path) else []
            search_targets += [
                os.path.join(self.archive_dir, name)
                for name in os.listdir(self.archive_dir)
                if name.endswith(".md")
            ]

        results: List[MemoryResult] = []
        for path in search_targets:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                continue

            if query_lower not in content.lower():
                continue

            rel_path = os.path.relpath(path, self.workspace_path)
            results.append(MemoryResult(
                content=content,
                source=rel_path,
                level=1,
                relevance_score=1.0,
                metadata={"path": rel_path, "search": "keyword"},
            ))

            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge: overlay keys override base; dict values are merged recursively."""
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = MemoryManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        # Workspace-level config
        ws_config_path = config_path or os.path.join(self.root_dir, "config.yaml")
        if os.path.exists(ws_config_path):
            try:
                with open(ws_config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except OSError:
                data = {}
        else:
            data = {}

        # Tenant-level config override (if exists)
        if self.tenant_id and config_path is None:
            tenant_cfg_path = os.path.join(
                self.root_dir, "tenants", self.tenant_id, "config.yaml"
            )
            if os.path.exists(tenant_cfg_path):
                try:
                    with open(tenant_cfg_path, "r", encoding="utf-8") as f:
                        tenant_data = yaml.safe_load(f) or {}
                    data = self._deep_merge(data, tenant_data)
                except OSError:
                    pass

        compaction = data.get("compaction") or {}
        compaction.setdefault("threshold_tokens", 2000)
        compaction.setdefault("strategy", "basic")
        data["compaction"] = compaction
        return data
