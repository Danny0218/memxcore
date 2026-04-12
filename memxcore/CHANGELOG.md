# Changelog

All notable changes to this project are documented in this file.

## [0.2.3] — 2026-04-13

### Fixed

- **`memxcore setup`**: Detect whether `memxcore` is loaded from `site-packages` (typical pip install) or a **source checkout**. For source layouts, set **`PYTHONPATH`** on MCP `env`, Cursor hook scripts, and Claude `mcp add` / hook commands so subprocesses can import the package.
- **Cursor CLI**: Merge **`~/.cursor/cli-config.json`** to add **`Mcp(memxcore:*)`** when no memxcore MCP rule exists; under **`approvalMode: allowlist`**, add baseline **`Read(**)`**, **`Write(**)`**, and **`Shell(**)`** when those tool families are not already allowed (avoids “MCP never loads” due to wrong server name or missing builtins).

### Added

- **`memxcore.hooks.cursor_stop`**: Cursor `stop` hook runs compaction (aligned with Claude stop chain’s compact step). Included in repo; `setup` continues to install Cursor `hooks.json` entries.

### Changed

- **`user_prompt_submit`**: Docstring clarifies stdin fields for Claude (`user_prompt`) vs Cursor (`prompt`).

### Docs

- README / README.zh-CN: Cursor row describes MCP, hooks, CLI permissions, and pip vs source behavior.

### Tests

- Unit tests for launch-path detection and `cli-config` permission merge.

[0.2.3]: https://github.com/Danny0218/memxcore/compare/v0.2.2...v0.2.3
