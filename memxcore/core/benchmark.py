"""
Benchmark — Quantitative search accuracy evaluation.

Load a test dataset (memories + queries + ground truth),
run each search mode, and compute R@K.

Usage:
    from memxcore.core.benchmark import run_benchmark
    results = run_benchmark("benchmarks/default.json", workspace_path=".")
"""

import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from . import utils
from .rag import _split_archive_sections


def run_benchmark(
    dataset_path: str,
    workspace_path: str,
    ks: Optional[List[int]] = None,
    modes: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a benchmark.

    Args:
        dataset_path: JSON file containing memories + queries
        workspace_path: memx workspace root directory
        ks: list of K values to compute, default [1, 3, 5, 10]
        modes: list of search modes, default ["hybrid", "rag", "bm25", "keyword"]
        verbose: whether to print results for each query

    Returns:
        {
            "dataset": str,
            "num_memories": int,
            "num_queries": int,
            "results": {mode: {f"R@{k}": float, ...}, ...},
            "per_query": [{query, expected, mode, found, rank}, ...],
            "elapsed_seconds": float,
        }
    """
    if ks is None:
        ks = [1, 3, 5, 10]
    if modes is None:
        modes = ["hybrid", "rag", "bm25", "keyword"]

    # ── 1. Load dataset ────────────────────────────────────────────────
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    memories = dataset["memories"]
    queries = dataset["queries"]

    # ── 2. Create an isolated test environment ────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="memx_bench_")
    try:
        results = _run_in_sandbox(
            tmpdir, memories, queries, ks, modes, verbose, workspace_path
        )
        results["dataset"] = os.path.basename(dataset_path)
        results["num_memories"] = len(memories)
        results["num_queries"] = len(queries)
        return results
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_in_sandbox(
    tmpdir: str,
    memories: List[Dict],
    queries: List[Dict],
    ks: List[int],
    modes: List[str],
    verbose: bool,
    workspace_path: str,
) -> Dict[str, Any]:
    """Create a MemoryManager in a temp directory, load memories, and run the benchmark."""
    from .memory_manager import MemoryManager

    start = time.time()

    # Initialize manager (isolated via bench tenant)
    mgr = MemoryManager(tmpdir, tenant_id=None)

    # ── Load memories into archive (write files directly, skip LLM distillation) ──
    _load_memories_direct(mgr, memories)

    # Rebuild indexes
    mgr.rebuild_rag_index()
    mgr.bm25.rebuild(mgr.archive_dir, mgr.user_path)
    mgr.update_index()

    # ── Run each search mode ───────────────────────────────────────────
    all_results: Dict[str, Dict[str, float]] = {}
    per_query: List[Dict] = []
    max_k = max(ks)

    for mode in modes:
        hits_at_k = {k: 0 for k in ks}

        for q in queries:
            query_text = q["query"]
            expected = q["expected_content"]

            results = _search_by_mode(mgr, mode, query_text, max_k)
            rank = _find_rank(results, expected)

            for k in ks:
                if rank is not None and rank <= k:
                    hits_at_k[k] += 1

            per_query.append({
                "query": query_text,
                "expected": expected,
                "mode": mode,
                "found": rank is not None,
                "rank": rank,
                "type": q.get("type", ""),
                "difficulty": q.get("difficulty", ""),
            })

            if verbose:
                status = f"rank={rank}" if rank else "MISS"
                print(f"  [{mode}] {query_text}: {status}")

        # Compute R@K
        n = len(queries)
        mode_scores = {}
        for k in ks:
            mode_scores[f"R@{k}"] = round(hits_at_k[k] / n * 100, 1) if n > 0 else 0.0
        all_results[mode] = mode_scores

    elapsed = time.time() - start

    return {
        "results": all_results,
        "per_query": per_query,
        "elapsed_seconds": round(elapsed, 2),
    }


def _load_memories_direct(mgr: "object", memories: List[Dict]) -> None:
    """Write directly to archive files, bypassing LLM distillation."""
    from datetime import datetime

    by_category: Dict[str, List[Dict]] = {}
    for m in memories:
        cat = m.get("category", "general")
        by_category.setdefault(cat, []).append(m)

    timestamp = datetime.utcnow().isoformat()

    for category, items in by_category.items():
        path = os.path.join(mgr.archive_dir, f"{category}.md")
        lines = [
            "---",
            f"topic: {category}",
            f"tags: []",
            f"last_distilled: '{timestamp}'",
            "confidence_level: 4",
            "---",
            "",
        ]
        for item in items:
            lines.append(f"## [{timestamp}]")
            lines.append("")
            lines.append(item["content"])
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _search_by_mode(
    mgr: "object",
    mode: str,
    query: str,
    max_results: int,
) -> List[str]:
    """Search using the specified mode and return a list of content strings."""
    if mode == "hybrid":
        # Normal search (hybrid RAG + BM25)
        results = mgr.search(query, max_results=max_results)
        return [r.content for r in results]

    elif mode == "rag":
        if not mgr.rag.available:
            return []
        hits = mgr.rag.search(query, top_k=max_results)
        return [h["content"] for h in hits]

    elif mode == "bm25":
        if not mgr.bm25.available:
            return []
        hits = mgr.bm25.search(query, top_k=max_results)
        return [h["content"] for h in hits]

    elif mode == "keyword":
        results = mgr._keyword_search(query, max_results)
        return [r.content for r in results]

    return []


def _find_rank(results: List[str], expected_content: str) -> Optional[int]:
    """
    Find the position of expected_content in search results (1-indexed).
    Uses substring match: a hit is counted if expected appears within a result.
    """
    expected_lower = expected_content.lower()
    for i, content in enumerate(results):
        if expected_lower in content.lower():
            return i + 1
    return None


def format_report(bench_result: Dict[str, Any]) -> str:
    """Format benchmark results into a human-readable report."""
    lines = []
    lines.append("MemXCore Benchmark Report")
    lines.append(f"{'=' * 60}")
    lines.append(f"Dataset: {bench_result['dataset']}")
    lines.append(f"Memories: {bench_result['num_memories']}")
    lines.append(f"Queries: {bench_result['num_queries']}")
    lines.append(f"Time: {bench_result['elapsed_seconds']}s")
    lines.append("")

    # ── Main table: R@K for each mode ──────────────────────────────────
    results = bench_result["results"]
    if not results:
        lines.append("No results.")
        return "\n".join(lines)

    # Collect all K values
    all_ks = sorted(set(
        k for mode_scores in results.values() for k in mode_scores.keys()
    ))

    # Header
    header = f"{'Mode':<12}" + "".join(f"{k:>8}" for k in all_ks)
    lines.append(header)
    lines.append("-" * len(header))

    for mode, scores in results.items():
        row = f"{mode:<12}" + "".join(f"{scores.get(k, 0):>7.1f}%" for k in all_ks)
        lines.append(row)

    # ── Breakdown by query type ────────────────────────────────────────
    per_query = bench_result.get("per_query", [])
    if per_query:
        lines.append("")
        lines.append("Breakdown by query type (hybrid mode):")
        lines.append("-" * 40)

        hybrid_queries = [q for q in per_query if q["mode"] == "hybrid"]
        by_type: Dict[str, List[Dict]] = {}
        for q in hybrid_queries:
            by_type.setdefault(q.get("type", "unknown"), []).append(q)

        for qtype, qs in sorted(by_type.items()):
            hits = sum(1 for q in qs if q["found"])
            total = len(qs)
            pct = hits / total * 100 if total > 0 else 0
            lines.append(f"  {qtype:<20} {hits}/{total} ({pct:.0f}%)")

    # ── Miss list ────────────────────────────────────────────────────
    hybrid_misses = [
        q for q in per_query
        if q["mode"] == "hybrid" and not q["found"]
    ]
    if hybrid_misses:
        lines.append("")
        lines.append(f"Misses (hybrid, {len(hybrid_misses)} queries):")
        for q in hybrid_misses:
            lines.append(f"  MISS: \"{q['query']}\" (expected: \"{q['expected']}\")")

    return "\n".join(lines)
