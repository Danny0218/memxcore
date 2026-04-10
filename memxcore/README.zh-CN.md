# MemXCore

AI 編程助手的持久化記憶系統。透過 MCP（Model Context Protocol）跨 session 存取記憶，支援 ChromaDB + sentence-transformers 語義搜索。

支援 **Claude Code**、**Cursor**、**Gemini CLI** 及所有 MCP 相容工具。

[English](README.md) | **中文**

---

## 架構

```
RECENT.md          <- L0: 原始追加日誌（WAL），每次 remember() 寫入這裡
archive/<cat>.md   <- L1: 蒸餾後的長期記憶，按類別分檔，含 YAML front matter
USER.md            <- L2: 永久記憶（remember(permanent=True)），不會被壓縮
chroma/            <- RAG 向量索引（從 archive/ 重建，可丟失）
index.json         <- 關鍵字搜索索引
```

**搜索層級（依序）：**
1. **RAG 語義搜索**（ChromaDB + BM25，RRF 融合）— 最精確
2. **LLM 相關性判斷** — Claude 從所有 facts 中挑選相關的
3. **關鍵字 fallback** — 永遠可用，掃描 archive 文件

**寫入流程：**
1. `remember(text)` -> 追加到 RECENT.md
2. RECENT.md 超過 token 閾值 -> LLM 蒸餾為結構化 facts
3. 每個 fact -> 同時寫入 `archive/<category>.md` + ChromaDB（雙寫）

---

## 快速開始

```bash
pip install memxcore          # 核心
pip install 'memxcore[rag]'   # + 語義搜索（推薦）
pip install 'memxcore[all]'   # 全部功能

# 或從原始碼安裝：
git clone https://github.com/Danny0218/memxcore.git
cd memxcore
pip install -r memxcore/requirements.txt
```

**檢查安裝：**
```bash
memxcore doctor
```

**環境要求：** Python 3.11+

**LLM API Key**（透過 [litellm](https://docs.litellm.ai/) 支援所有主流 provider）：
```bash
# 任選一個：
export ANTHROPIC_API_KEY=sk-ant-...     # Anthropic（直連 API）
export ANTHROPIC_AUTH_TOKEN=xxx         # Anthropic（Gateway/代理，搭配 ANTHROPIC_BASE_URL）
export OPENAI_API_KEY=sk-...            # OpenAI
export GEMINI_API_KEY=...               # Google Gemini
# 或使用 Ollama 完全本地運行，不需 key，只需在 config.yaml 設定 model
```

未設定任何 API key 時，壓縮功能退化為 basic 模式（截斷而非 LLM 蒸餾）。

---

## 一鍵配置

```bash
memxcore setup            # 自動偵測已安裝的工具並配置
memxcore setup --dry-run  # 只預覽，不實際執行
```

自動偵測並配置：

| 工具 | 自動配置內容 |
|------|-------------|
| **Claude Code** | MCP server 註冊 + hooks（auto-remember + auto-compact）+ agent 規則 |
| **Cursor** | MCP config (`~/.cursor/mcp.json`) + 規則 (`~/.cursor/rules/memxcore.md`) |
| **Windsurf** | MCP config (`~/.codeium/windsurf/mcp_config.json`) |
| **Codex (OpenAI)** | MCP config (`~/.codex/config.toml`) |
| **Gemini CLI** | MCP config (`~/.gemini/settings.json`) |

配置完成後重啟工具即可。

---

## 手動配置

任何支援 MCP 的工具都可以用以下 JSON 配置：

```json
{
  "mcpServers": {
    "memxcore": {
      "command": "/path/to/your/.venv/bin/python",
      "args": ["-m", "memxcore.mcp_server"],
      "env": { "MEMXCORE_WORKSPACE": "/path/to/your/workspace" }
    }
  }
}
```
```

---

## MCP 工具

| 工具 | 參數 | 說明 |
|------|------|------|
| `remember` | `text`, `category?`, `permanent?`, `tenant_id?` | 存入記憶 |
| `search` | `query`, `max_results?`, `tenant_id?` | 語義 + 關鍵字搜索 |
| `compact` | `force?`, `tenant_id?` | 蒸餾 RECENT.md 到分類歸檔 |
| `reindex` | `category?`, `tenant_id?` | 手動編輯 archive 後重建索引 |

**記憶類別：** `user_model` / `domain` / `project_state` / `episodic` / `references`

**永久記憶：** 使用 `permanent=True` 直接寫入 USER.md（L2），搜索時優先返回。

---

## 多租戶

所有工具和 CLI 指令支援 `tenant_id` 參數。每個租戶的資料隔離在 `tenants/<tenant_id>/storage/`。

```bash
memxcore --tenant alice search "preferences"
```

---

## CLI 指令

```bash
memxcore setup                        # 自動偵測工具，一鍵配置
memxcore setup --dry-run              # 預覽不執行
memxcore doctor                       # 檢查系統狀態
memxcore reindex                      # 重建 RAG 索引
memxcore compact                      # 強制蒸餾
memxcore search "query"               # 搜索（除錯用）
memxcore benchmark                    # 搜索精度基準測試
memxcore mine <path>                  # 匯入對話歷史
memxcore --tenant alice doctor        # 多租戶
```

---

## 安全注意事項

- **檔案權限：** 記憶檔案使用預設 OS 權限。多用戶系統請手動 `chmod 700 memxcore/storage/`
- **HTTP server：** 無認證，僅限 localhost 開發用途
- **LLM prompt injection：** category 名稱已做 sanitize 防路徑穿越，但記憶內容注入是 LLM 系統的通病

---

## 授權

MIT. 見 [LICENSE](LICENSE)。
