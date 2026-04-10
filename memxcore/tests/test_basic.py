"""
memx test suite.

Tests use temporary directories to avoid polluting real storage.
LLM-dependent features (distillation) are tested with strategy=basic
to avoid requiring an API key in CI.
"""

import json
import os
import shutil
import tempfile
from typing import Generator

import pytest
import yaml

from memxcore.core.memory_manager import MemoryManager, MemoryResult
from memxcore.core.compaction import (
    _estimate_tokens,
    summarize_recent,
    _prune_duplicate_sections,
    _demote_stale_project_state,
    _word_overlap,
)
from memxcore.core.rag import _split_archive_sections
from memxcore.core.bm25 import _tokenize
from memxcore.core.utils import parse_front_matter, parse_llm_json, extract_simple_summary


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace(tmp_path) -> str:
    """Create a temporary workspace with minimal memx config (basic strategy)."""
    cm_dir = tmp_path / "memxcore"
    cm_dir.mkdir()
    config = {
        "compaction": {
            "threshold_tokens": 50,
            "strategy": "basic",
            "min_entries": 1,
            "check_interval": 1,
            "stale_minutes": 10,
        },
        "watch": False,
    }
    (cm_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return str(tmp_path)


@pytest.fixture
def manager(workspace) -> MemoryManager:
    return MemoryManager(workspace_path=workspace)


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestParseFromFrontMatter:
    def test_with_front_matter(self):
        raw = "---\ntopic: test\ntags: [a, b]\n---\n\nBody content"
        meta, body = parse_front_matter(raw)
        assert meta["topic"] == "test"
        assert "Body content" in body

    def test_without_front_matter(self):
        raw = "Just plain text"
        meta, body = parse_front_matter(raw)
        assert meta == {}
        assert body == "Just plain text"

    def test_malformed_yaml(self):
        raw = "---\n: invalid yaml {{{\n---\nBody"
        meta, body = parse_front_matter(raw)
        assert meta == {}

    def test_no_closing_delimiter(self):
        raw = "---\ntopic: test\nNo closing"
        meta, body = parse_front_matter(raw)
        assert meta == {}


class TestParseLlmJson:
    def test_plain_json(self):
        assert parse_llm_json('[{"a": 1}]') == [{"a": 1}]

    def test_markdown_fenced(self):
        raw = "```json\n[{\"a\": 1}]\n```"
        assert parse_llm_json(raw) == [{"a": 1}]

    def test_invalid_json(self):
        assert parse_llm_json("not json at all") is None

    def test_empty_string(self):
        assert parse_llm_json("") is None


class TestExtractSummary:
    def test_first_line(self):
        assert extract_simple_summary("First line\nSecond line") == "First line"

    def test_skips_empty_lines(self):
        assert extract_simple_summary("\n\n\nActual content") == "Actual content"

    def test_truncation(self):
        long = "x" * 200
        result = extract_simple_summary(long, max_len=50)
        assert len(result) == 50
        assert result.endswith("...")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_english(self):
        tokens = _tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_chinese(self):
        tokens = _tokenize("你好世界")
        assert all(ch in tokens for ch in "你好世界")

    def test_mixed(self):
        tokens = _tokenize("用戶 prefers English")
        assert "用" in tokens
        assert "prefers" in tokens


class TestEstimateTokens:
    def test_english(self):
        assert _estimate_tokens("hello world foo bar") >= 4

    def test_chinese(self):
        assert _estimate_tokens("你好世界") >= 4

    def test_empty(self):
        assert _estimate_tokens("") >= 1


# ── Archive sections ──────────────────────────────────────────────────────────

class TestSplitArchiveSections:
    def test_basic(self):
        body = "## [2026-01-01]\n\nFact one\n\n## [2026-01-02]\n\nFact two\n"
        sections = _split_archive_sections(body)
        assert len(sections) == 2
        assert sections[0][0] == "2026-01-01"
        assert "Fact one" in sections[0][1]

    def test_empty(self):
        assert _split_archive_sections("") == []

    def test_no_sections(self):
        assert _split_archive_sections("Just some text without sections") == []


# ── Word overlap ──────────────────────────────────────────────────────────────

class TestWordOverlap:
    def test_identical(self):
        assert _word_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _word_overlap("hello", "world") == 0.0

    def test_partial(self):
        score = _word_overlap("hello world foo", "hello world bar")
        assert 0.3 < score < 0.8

    def test_empty(self):
        assert _word_overlap("", "hello") == 0.0


# ── MemoryManager ─────────────────────────────────────────────────────────────

class TestMemoryManager:
    def test_init_creates_storage(self, workspace):
        mm = MemoryManager(workspace_path=workspace)
        assert os.path.isdir(mm.storage_dir)
        assert os.path.isdir(mm.archive_dir)
        assert os.path.isfile(mm.recent_path)
        assert os.path.isfile(mm.user_path)

    def test_remember_appends_to_recent(self, manager):
        manager.remember("test fact 1")
        with open(manager.recent_path, "r") as f:
            content = f.read()
        assert "test fact 1" in content

    def test_remember_permanent_writes_to_user(self, manager):
        manager.remember("permanent fact", level=2)
        with open(manager.user_path, "r") as f:
            content = f.read()
        assert "permanent fact" in content

    def test_remember_permanent_not_in_recent(self, manager):
        manager.remember("permanent only", level=2)
        with open(manager.recent_path, "r") as f:
            content = f.read()
        assert "permanent only" not in content

    def test_compact_basic_strategy(self, manager):
        for i in range(5):
            manager.remember(f"fact number {i} for basic compaction test")
        manager.compact(force=True)

        # Give async thread time to finish
        import time
        time.sleep(0.5)

        archive_files = [f for f in os.listdir(manager.archive_dir) if f.endswith(".md")]
        assert archive_files, "Expected archive files after compaction"

    def test_search_keyword_fallback(self, manager):
        manager.remember("unique_keyword_xyz is important", level=2)
        results = manager.search("unique_keyword_xyz")
        assert any("unique_keyword_xyz" in r.content for r in results)

    def test_search_empty(self, manager):
        results = manager.search("nonexistent_query_12345")
        assert results == []

    def test_update_index(self, manager):
        # Write a fake archive file
        archive_path = os.path.join(manager.archive_dir, "test_cat.md")
        with open(archive_path, "w") as f:
            f.write("---\ntopic: test_cat\ntags: [test]\nlast_distilled: 2026-01-01\n---\n\n## [2026-01-01]\n\nTest fact\n")
        manager.update_index()
        with open(manager.index_path, "r") as f:
            index = json.load(f)
        assert any("test_cat" in item.get("path", "") for item in index["files"])


# ── Multi-tenant ──────────────────────────────────────────────────────────────

class TestMultiTenant:
    def test_tenant_isolation(self, workspace):
        mm_alice = MemoryManager(workspace_path=workspace, tenant_id="alice")
        mm_bob = MemoryManager(workspace_path=workspace, tenant_id="bob")

        mm_alice.remember("alice secret", level=2)
        mm_bob.remember("bob secret", level=2)

        alice_results = mm_alice.search("secret")
        bob_results = mm_bob.search("secret")

        alice_content = " ".join(r.content for r in alice_results)
        bob_content = " ".join(r.content for r in bob_results)

        assert "alice" in alice_content
        assert "bob" not in alice_content
        assert "bob" in bob_content
        assert "alice" not in bob_content

    def test_invalid_tenant_id(self, workspace):
        with pytest.raises(ValueError, match="Invalid tenant_id"):
            MemoryManager(workspace_path=workspace, tenant_id="bad/tenant")

    def test_tenant_storage_path(self, workspace):
        mm = MemoryManager(workspace_path=workspace, tenant_id="test_tenant")
        assert "tenants/test_tenant/storage" in mm.storage_dir


# ── Compaction helpers ────────────────────────────────────────────────────────

class TestSummarizeRecent:
    def test_short_content(self):
        content = "line1\nline2\nline3"
        assert summarize_recent(content, max_lines=50) == content

    def test_truncation(self):
        content = "\n".join(f"line {i}" for i in range(100))
        result = summarize_recent(content, max_lines=10)
        assert "truncated" in result
        assert result.count("\n") < 20


class TestPruneDuplicates:
    def test_removes_duplicate(self, workspace):
        mm = MemoryManager(workspace_path=workspace)
        archive_path = os.path.join(mm.archive_dir, "domain.md")
        with open(archive_path, "w") as f:
            f.write("---\ntopic: domain\ntags: []\nlast_distilled: 2026-01-01\n---\n\n## [2026-01-01]\n\nOld fact about Python testing\n")

        _prune_duplicate_sections(mm.archive_dir, "domain", "Old fact about Python testing updated")

        with open(archive_path, "r") as f:
            content = f.read()
        # Old fact should be removed (high overlap with new content)
        assert "Old fact about Python testing\n" not in content


class TestDemoteStaleProjectState:
    def test_demotes_old_entries(self, workspace):
        mm = MemoryManager(workspace_path=workspace)
        archive_path = os.path.join(mm.archive_dir, "project_state.md")

        with open(archive_path, "w") as f:
            f.write(
                "---\ntopic: project_state\ntags: []\nlast_distilled: 2020-01-01\n---\n\n"
                "## [2020-01-01T00:00:00]\n\nVery old project state entry\n"
            )

        _demote_stale_project_state(mm.archive_dir)

        episodic_path = os.path.join(mm.archive_dir, "episodic.md")
        assert os.path.isfile(episodic_path)
        with open(episodic_path, "r") as f:
            content = f.read()
        assert "Very old project state entry" in content


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults_without_config_file(self, tmp_path):
        cm_dir = tmp_path / "memxcore"
        cm_dir.mkdir()
        mm = MemoryManager(workspace_path=str(tmp_path))
        assert mm.config["compaction"]["threshold_tokens"] == 2000
        assert mm.config["compaction"]["strategy"] == "basic"

    def test_tenant_config_override(self, tmp_path):
        cm_dir = tmp_path / "memxcore"
        cm_dir.mkdir()
        base_config = {"compaction": {"threshold_tokens": 1000, "strategy": "basic"}}
        (cm_dir / "config.yaml").write_text(yaml.safe_dump(base_config))

        tenant_dir = cm_dir / "tenants" / "custom"
        tenant_dir.mkdir(parents=True)
        tenant_config = {"compaction": {"threshold_tokens": 500}}
        (tenant_dir / "config.yaml").write_text(yaml.safe_dump(tenant_config))

        mm = MemoryManager(workspace_path=str(tmp_path), tenant_id="custom")
        assert mm.config["compaction"]["threshold_tokens"] == 500
        assert mm.config["compaction"]["strategy"] == "basic"  # inherited from base


# ── CLI doctor (smoke test) ───────────────────────────────────────────────────

class TestCLI:
    def test_doctor_runs(self, capsys):
        from memxcore.cli import cmd_doctor
        cmd_doctor()
        captured = capsys.readouterr()
        assert "MemXCore Doctor" in captured.out
        assert "Python" in captured.out
        assert "Dependencies" in captured.out


# -- Auto-remember hook --------------------------------------------------------

class TestAutoRememberTranscript:
    """Tests for transcript parsing in the auto-remember Stop hook."""

    def _write_transcript(self, tmp_path, lines: list) -> str:
        path = str(tmp_path / "transcript.jsonl")
        with open(path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return path

    def test_parse_basic_exchange(self, tmp_path):
        from memxcore.hooks.auto_remember import parse_last_exchange
        path = self._write_transcript(tmp_path, [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "message": {"role": "user", "content": "What is Python?"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Python is a programming language."}],
                },
            },
        ])
        result = parse_last_exchange(path)
        assert result is not None
        user_text, assistant_text = result
        assert "Python" in user_text
        assert "programming language" in assistant_text

    def test_parse_content_blocks(self, tmp_path):
        from memxcore.hooks.auto_remember import parse_last_exchange
        path = self._write_transcript(tmp_path, [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "message": {"role": "user", "content": "Tell me about memory systems"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Part one."},
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                        {"type": "text", "text": "Part two."},
                    ],
                },
            },
        ])
        result = parse_last_exchange(path)
        assert result is not None
        _, assistant_text = result
        assert "Part one" in assistant_text
        assert "Part two" in assistant_text
        assert "tool_use" not in assistant_text

    def test_parse_empty_transcript(self, tmp_path):
        from memxcore.hooks.auto_remember import parse_last_exchange
        path = self._write_transcript(tmp_path, [])
        assert parse_last_exchange(path) is None

    def test_parse_no_assistant(self, tmp_path):
        from memxcore.hooks.auto_remember import parse_last_exchange
        path = self._write_transcript(tmp_path, [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "message": {"role": "user", "content": "Hello"},
            },
        ])
        assert parse_last_exchange(path) is None

    def test_parse_nonexistent_file(self):
        from memxcore.hooks.auto_remember import parse_last_exchange
        assert parse_last_exchange("/nonexistent/path.jsonl") is None

    def test_parse_skips_metadata_lines(self, tmp_path):
        from memxcore.hooks.auto_remember import parse_last_exchange
        path = self._write_transcript(tmp_path, [
            {"type": "file-history-snapshot", "snapshot": {}},
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "message": {"role": "user", "content": "Real question here"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Real answer"}],
                },
            },
            {"type": "last-prompt", "lastPrompt": "Real question here"},
        ])
        result = parse_last_exchange(path)
        assert result is not None
        assert "Real question" in result[0]


class TestAutoRememberExtractText:
    def test_string_content(self):
        from memxcore.hooks.auto_remember import _extract_text
        assert _extract_text("hello") == "hello"

    def test_block_content(self):
        from memxcore.hooks.auto_remember import _extract_text
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _extract_text(blocks) == "a\nb"

    def test_mixed_blocks(self):
        from memxcore.hooks.auto_remember import _extract_text
        blocks = [
            {"type": "text", "text": "visible"},
            {"type": "tool_use", "id": "x", "name": "Bash"},
        ]
        assert _extract_text(blocks) == "visible"

    def test_empty(self):
        from memxcore.hooks.auto_remember import _extract_text
        assert _extract_text(None) == ""
        assert _extract_text([]) == ""


class TestAutoRememberWriteFacts:
    def test_write_facts(self, workspace):
        from memxcore.hooks.auto_remember import write_facts
        import memxcore.hooks.auto_remember as hook_mod

        recent_path = os.path.join(workspace, "memxcore", "storage", "RECENT.md")
        os.makedirs(os.path.dirname(recent_path), exist_ok=True)
        with open(recent_path, "w") as f:
            f.write("")

        orig = hook_mod.RECENT_MD
        hook_mod.RECENT_MD = recent_path
        try:
            count = write_facts([
                {"content": "User prefers concise answers", "category": "user_model"},
                {"content": "Project uses React 19", "category": "domain"},
            ])
            assert count == 2
            with open(recent_path, "r") as f:
                content = f.read()
            assert "User prefers concise answers" in content
            assert "Project uses React 19" in content
            assert "Memory" in content
        finally:
            hook_mod.RECENT_MD = orig

    def test_write_empty_facts(self, workspace):
        from memxcore.hooks.auto_remember import write_facts
        assert write_facts([]) == 0
