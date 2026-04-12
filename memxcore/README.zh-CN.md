# MemXCore

[![PyPI version](https://img.shields.io/pypi/v/memxcore)](https://pypi.org/project/memxcore/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**懂你工作方式的 AI 記憶層。**

你的 AI 不需要被教就能記住。它會從自然對話中學習你的偏好、習慣和過去的決策 — 不需要任何手動設定。支援 Claude Code、Cursor、Windsurf、Gemini CLI 及任何 MCP 相容工具。

[English](README.md) | **中文**

---

## 為什麼選 MemXCore？

每次開新的 AI 對話，你都失去上下文。你重複解釋自己的編碼風格、重述專案決策，然後看著 AI 犯你昨天才糾正過的錯誤。

MemXCore 解決這個問題。它作為共享記憶層存在於你和 AI 工具之間 — 自動從對話中提取重要資訊，讓每個未來的 session、每個工具都能使用。

---

## 功能亮點

- 🧠 **AI 自動學習你** — LLM 蒸餾從對話中提取事實、偏好和行為模式。反覆出現的習慣會自動升級為永久記憶。你完全不需要手動教它。

- 📖 **完全透明的 Markdown** — 每條記憶都是可讀的 `.md` 檔案。用任何編輯器打開、`git push` 做版本控制、`grep` 搜尋。不是你無法檢查的黑盒向量庫。

- 🔗 **一份記憶，所有工具** — 早上用 Cursor 寫程式，下午用 Claude Code 做 review，記憶跟著你走。一行 `memxcore setup` 自動配置所有偵測到的工具。

- 🔍 **三層搜尋** — 語意搜尋（ChromaDB + BM25 融合）、LLM 相關性判斷、關鍵字回退。支援實體標籤加權和知識圖譜關係查詢。

- 🕐 **可瀏覽的時間線** — 原始對話永久存檔在每日 journal 檔案中。執行 `memxcore timeline --days 3` 查看這週做了什麼。原始時間戳在蒸餾過程中完整保留。

- ⚡ **一行命令啟動** — `pip install memxcore && memxcore setup`。自動偵測 Claude Code、Cursor、Windsurf、Codex 和 Gemini CLI，Hooks 自動安裝。

---

## 快速開始

```bash
pip install 'memxcore[rag,bm25]'   # 推薦：包含混合搜尋
memxcore setup                      # 自動偵測並配置所有工具
```

從原始碼安裝：
```bash
git clone https://github.com/Danny0218/memxcore.git
cd memxcore && pip install '.[rag,bm25]'
memxcore setup
```

驗證安裝：
```bash
memxcore doctor
```

**環境要求：** Python 3.11+

**LLM API 金鑰**（透過 [litellm](https://docs.litellm.ai/) 支援任何供應商）：
```bash
export ANTHROPIC_API_KEY=sk-ant-...     # 或 OPENAI_API_KEY、GEMINI_API_KEY
# Ollama 也支援 — 不需要金鑰，在 config 設定模型即可
```

沒有 API 金鑰時，蒸餾會退回基礎模式（截斷，不分類）。

---

## 運作原理

```
你跟 AI 對話
      |
  remember()          原始對話 → RECENT.md
      |
  compact()           LLM 提取事實、習慣、決策
      |
  +---+---+
  |       |
journal/ archive/     無損歸檔 + 分類知識庫
  |       |
  |   search()        語意 + 關鍵字 + 知識圖譜
  |       |
  +---+---+
      |
  AI 了解你了
```

記憶流經三個層級：
- **RECENT.md** — 原始寫入緩衝區。每次 `remember()` 都寫在這裡。
- **archive/** — 蒸餾後的事實，分類標注，可搜尋。
- **USER.md** — 永久記憶。經過時間驗證的事實會自動升級到這裡。

在任何蒸餾之前，原始內容會先存到 **journal/** — 一個永久的、無損的每日日誌，永不刪除。你隨時可以回溯。

<details>
<summary><strong>技術架構</strong></summary>

```
RECENT.md          <- L0：僅追加寫入緩衝區
journal/<date>.md  <- L0.5：永久原始歸檔，無損
archive/<cat>.md   <- L1：LLM 蒸餾事實，帶 YAML 前置資料
USER.md            <- L2：永久事實，從 L1 自動升級
chroma/            <- RAG 向量索引（可重建）
knowledge.db       <- SQLite 實體關係三元組
index.json         <- 關鍵字搜尋索引
```

**寫入路徑：**
1. `remember(text)` → 帶時間戳追加到 RECENT.md
2. Token 閾值到達 → 原始內容歸檔到 `journal/<date>.md`
3. LLM 蒸餾為結構化事實，保留原始時間戳（`occurred_at`）
4. 每條事實 → 雙寫到 `archive/<category>.md` + ChromaDB
5. 知識圖譜三元組提取（例如 `Alice → leads → frontend-team`）

**搜尋路徑（3 層）：**
1. **混合搜尋** — RAG 語意（ChromaDB）+ BM25 關鍵字，經倒數排名融合（RRF）
2. **LLM 判斷** — Claude/GPT 閱讀所有事實並挑選相關的
3. **關鍵字回退** — 直接檔案掃描，始終可用

**生命週期管理：**
- `project_state` 超過 90 天 → 自動降級到 `episodic`
- `user_model` 被強化 3 次以上 → 自動升級到 USER.md（L2）
- 重複事實（詞彙重疊 > 65%）→ 寫入時自動去重

</details>

---

## 效能基準

基於 [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) 評測（500 題，每題約 50 個對話 session）。每題將事實分散在數十個對話中，再提問需要從中檢索正確 session 的問題。

### 搜尋引擎上限（Strategy B）

將原始對話文本直接寫入 archive 並建立索引——不經過 LLM 蒸餾。測量搜尋引擎在完美輸入下的**最大召回率**。

| 模式 | R@1 | R@3 | R@5 | R@10 |
|------|----:|----:|----:|-----:|
| **hybrid (RAG + BM25)** | 93.6% | 98.8% | **99.2%** | 100.0% |
| 僅 RAG | 89.8% | 97.0% | 98.8% | 100.0% |
| 僅 BM25 | 92.2% | 98.6% | 99.4% | 99.8% |

500 題中僅 4 題未命中——均為間接措辭或數值推理的邊界情況。

### 端到端流程（Strategy A）

完整 `remember()` → `compact()`（LLM 蒸餾）→ `search()` 流程。測量包含蒸餾資訊損失的**真實召回率**。

> **注意：** Strategy A 需要 LLM API key，完整 500 題需要約 5,000 次 API 呼叫（約 40M 輸入 tokens、20M 輸出 tokens）。使用 Haiku 約需 7 小時、費用約 $100。結果將在跑完後更新。你也可以自己跑——見下方指南。

### 策略說明

```
Strategy B: 對話文本 ──→ archive ──→ RAG/BM25 索引 ──→ search
                          (跳過蒸餾)
                          測量：搜尋引擎上限

Strategy A: 對話文本 ──→ remember() ──→ compact() ──→ archive ──→ search
                          (完整流程)
                          測量：含蒸餾損失的端到端召回率
```

B 與 A 的差距揭示了 LLM 蒸餾過程中的資訊損失——這是調優 compaction prompt 的關鍵指標。

### 自行跑 benchmark

```bash
cd memxcore
python -m venv .bench-venv && source .bench-venv/bin/activate
pip install -r requirements.txt huggingface-hub chromadb sentence-transformers rank-bm25 pyyaml

# Strategy B — 不需要 API key，約 11 分鐘
python -m benchmarks.longmemeval --strategy B

# Strategy A — 需要 LLM API key，使用 Haiku 約 7 小時
export ANTHROPIC_API_KEY=sk-ant-...
python -m benchmarks.longmemeval --strategy A --limit 50   # 快速取樣（約 40 分鐘）
python -m benchmarks.longmemeval --strategy A               # 完整 500 題

# 結果存成 JSON
python -m benchmarks.longmemeval --strategy B --output results_B.json
```

---

## 方案比較

| 功能 | **MemXCore** | Mem0 | Letta | memsearch | mempalace |
|------|:---:|:---:|:---:|:---:|:---:|
| MCP 協議 | Yes | No | No | Yes | Yes |
| 跨工具（Claude/Cursor/Gemini）| Yes | No | No | 部分 | 部分 |
| 透明 Markdown 儲存 | Yes | No | No | No | Yes |
| LLM 蒸餾 | Yes | Yes | Yes | No | No |
| 行為模式識別 | Yes | No | No | No | No |
| 混合搜尋（RAG + BM25）| Yes | 純向量 | 純向量 | 向量 + BM25 | 純向量 |
| 知識圖譜 | Yes | Yes (Neo4j) | No | No | No |
| Journal / 時間線 | Yes | No | No | No | No |
| 自動升級永久記憶 | Yes | No | No | No | No |
| 多租戶 | Yes | Yes | Yes | No | No |
| 完全本地 / 自託管 | Yes | 雲端或自建 | 雲端或自建 | Yes | Yes |
| 一行命令安裝 | Yes | No | No | No | No |

**什麼時候選 MemXCore：** 你希望 AI 工具自動學習你的偏好，你在意能檢查和編輯記憶內容，而且你使用多個 AI 編程工具並希望它們共享同一份上下文。

---

## 配置

位於 `memxcore/config.yaml`。所有欄位都是可選的 — 有合理的預設值。

```yaml
compaction:
  strategy: llm             # 'llm' = LLM 蒸餾為結構化事實（經 litellm）
                            # 'basic' = 截斷前 50 行到 general.md
  threshold_tokens: 2000    # RECENT.md 超過此 token 數時觸發壓縮
  min_entries: 3            # 條目不足 N 條時不壓縮
  check_interval: 5         # 每 N 次 remember() 檢查一次（減少 I/O）
  stale_minutes: 10         # search() 時若 RECENT.md 超過此分鐘數則自動壓縮

llm:
  # model: anthropic/claude-sonnet   # 取消註解並改為你的供應商
  #   範例：openai/gpt-4o, gemini/gemini-2.5-flash, ollama/llama3
  #   省略則根據可用的 API 金鑰自動偵測
  # api_key_env: OPENAI_API_KEY     # 環境變數名稱（省略則自動偵測）
  # base_url:                       # 可選：自訂 API 端點

rag:
  embedding_model: paraphrase-multilingual-MiniLM-L12-v2  # 多語言
  top_k: 10
  rrf_k: 60                # 倒數排名融合常數

memory:
  categories:
    - id: user_model
      description: "使用者身份、偏好、溝通風格、回饋"
    - id: domain
      description: "可遷移的領域/技術知識"
    - id: project_state
      description: "進行中的任務、決策 — 數週後衰減"
    - id: episodic
      description: "有時效的事件和過去的決策"
    - id: references
      description: "URL、文件連結、儀表板、工單追蹤器"
```

---

## MCP 工具

| 工具 | 參數 | 說明 |
|------|------|------|
| `remember` | `text`, `category?`, `permanent?`, `tenant_id?` | 儲存記憶 |
| `search` | `query`, `max_results?`, `after?`, `before?`, `tenant_id?` | 搜尋，支援時間範圍 |
| `compact` | `force?`, `tenant_id?` | 蒸餾 RECENT.md 到分類歸檔 |
| `flush` | `tenant_id?` | 移動 RECENT.md 到 journal（無損，不跑 LLM）|
| `reindex` | `category?`, `tenant_id?` | 重建 RAG + BM25 索引 |
| `purge_journal` | `keep_days?`, `tenant_id?` | 刪除超過 N 天的 journal 檔案 |
| `set_config` | `key`, `value`, `tenant_id?` | 設定配置值（點號表示法）|
| `kg_add` | `subject`, `predicate`, `object`, `valid_from?` | 新增知識圖譜三元組 |
| `kg_query` | `entity`, `as_of?` | 查詢實體關係 |
| `kg_timeline` | `entity` | 實體的時間線歷史 |

---

## CLI 命令

```bash
# 安裝與診斷
memxcore setup                               # 自動配置所有偵測到的工具
memxcore setup --dry-run                     # 預覽不做更改
memxcore doctor                              # 檢查系統就緒狀態

# 配置
memxcore config show                         # 顯示有效配置
memxcore config set llm.model openai/gpt-4o  # 切換 LLM 供應商
memxcore config path                         # 顯示配置檔路徑

# 記憶操作
memxcore compact                             # 強制蒸餾
memxcore flush                               # 移動 RECENT.md 到 journal（不跑 LLM）
memxcore reindex                             # 重建 RAG + BM25 索引

# 搜尋與瀏覽
memxcore search "查詢"                        # 搜尋記憶
memxcore search --after 2026-04-08 "查詢"     # 帶時間範圍搜尋
memxcore timeline                            # 最近 7 天的記憶活動
memxcore timeline --days 3                   # 最近 3 天

# 維護
memxcore purge-journal --keep-days 30        # 清理舊 journal 檔案
memxcore benchmark                           # 搜尋精確度測試
memxcore mine <路徑>                          # 匯入對話記錄

# 多租戶
memxcore --tenant alice doctor
```

---

## 整合

### 一行命令安裝（推薦）

```bash
memxcore setup
```

自動偵測並配置：

| 工具 | 做了什麼 |
|------|---------|
| **Claude Code** | MCP 伺服器 + Hooks（自動記憶、自動壓縮）+ Agent 規則 |
| **Cursor** | MCP 配置 + 規則檔案 |
| **Windsurf** | MCP 配置 |
| **Codex (OpenAI)** | MCP 配置 |
| **Gemini CLI** | MCP 配置 |

### 手動配置

任何工具的 MCP 伺服器配置：
```json
{
  "mcpServers": {
    "memxcore": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "memxcore.mcp_server"],
      "env": { "MEMXCORE_WORKSPACE": "/path/to/workspace" }
    }
  }
}
```

---

## 多租戶

所有工具和命令支援 `tenant_id` 參數，實現隔離的使用者/Agent 儲存：

```bash
memxcore --tenant alice search "偏好"
```

每個租戶擁有獨立的 `tenants/<id>/storage/` 和可選的 `config.yaml` 覆寫。

---

## 儲存結構

```
storage/
+-- RECENT.md              寫入緩衝區（壓縮後清空）
+-- USER.md                永久記憶（自動升級的事實）
+-- journal/               無損每日歸檔（永不自動刪除）
|   +-- 2026-04-10.md
|   +-- 2026-04-11.md
+-- archive/               按分類蒸餾的知識
|   +-- user_model.md
|   +-- domain.md
|   +-- project_state.md
|   +-- episodic.md
|   +-- references.md
+-- chroma/                向量索引（可重建）
+-- knowledge.db           實體關係圖
+-- index.json             關鍵字搜尋索引
```

所有檔案都是純 Markdown 或 SQLite — 人類可讀、可編輯、適合 `git` 管理。

---

## 安全性

- **完全本地** — 所有資料留在你的機器上。沒有雲端，沒有遙測。
- **HTTP 伺服器** 僅綁定 `127.0.0.1`。永遠不要暴露到網路。
- **LLM 提示注入** — 記憶內容會在蒸餾時傳給 LLM。分類名稱已做路徑遍歷防護。
- **儲存權限** — 使用預設 OS 權限建立。共享系統請執行：`chmod 700 storage/`。

---

## 故障排除

先執行 `memxcore doctor` — 它會檢查所有項目並告訴你如何修復。

```bash
memxcore doctor          # 完整診斷
memxcore reindex         # 重建搜尋索引
memxcore config show     # 驗證配置
```

---

## 授權

MIT。見 [LICENSE](LICENSE)。
