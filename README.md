# LLM Stock Advisory System

以 LLM 為核心、以公開資訊為基礎的**單股近期動態摘要**系統，針對台股設計。專案目標是在資訊不足或超出範圍時**明確拒答**，而不是硬湊出一份看起來完整但不可信的摘要。

---

## Part 1 — 產品與設計概覽

### 產品定位

`single_stock_digest` 產品線專注於：「**這檔股票的相關問題，但不往前估值、不做預測**」

- **Scope**：單股、可選擇觀測天數、限於 `NEWS` / `ANNOUNCEMENT` / `COMPOSITE` 三類主題、財報深度分析、技術分析
- **非目標**：多股比較、估值與目標價、預測漲跌、投資建議、財報深度分析(毛利率、營業利益率、EPS、現金流，需要FinMind 付費API的部份)
- **輸出契約**：摘要 + 事實條列 + 可能影響 + 風險提醒 + 來源清單 + 信任度紅綠燈
- 完整規格見 [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md)；Refusal Boundary 規格見 [`digest_refusal_boundary_v1_1.md`](digest_refusal_boundary_v1_1.md)

### 設計原則

| # | 原則 | 實作位置 |
|---|------|----------|
| 1 | **Refusal 先於 Validation**：先判斷「能不能答」，再判斷「答得有多好」 | `digest/input/scope_guard.py`、`validation_layer` |
| 2 | **拒絕優先於猜測**：資訊不足時退場，不勉強生成 | `ValidationLayer._apply_digest_profile_gate` |
| 3 | **Scope 必須封閉**：所有超出範圍的查詢走 refusal boundary | `OUT_OF_SCOPE_KEYWORDS`（v1.1 封閉清單） |
| 4 | **可觀測性優先**：refusal / degraded / normal 三條路徑都寫 query log | `QueryLog` schema |
| 5 | **LLM-first + 規則 floor**：InputLayer 對每個欄位獨立驗證 LLM 輸出；失敗沿用規則 | `InputLayer._classify_semantics` |

### 架構：Six-Layer Pipeline

```
Input → Retrieval → Data Governance → Generation → Validation → Presentation
```

- **Input Layer**：把自然語言轉成 `StructuredQuery(intent, controlled_tags, time_range, stance_bias, ...)`。支援 pure-rule 與 LLM-first (`OpenAIQueryClassifier`) 兩種分類路徑，LLM 輸出逐欄位用白名單驗證、失敗 fallback 到規則
- **Retrieval Layer**：PostgreSQL + pgvector；結構化篩選 → 向量召回 → rerank
- **Data Governance Layer**：來源 tier 分級、corroboration、freshness 標記
- **Generation Layer**：rule-based 或 OpenAI Responses；兩條路都走同一個 `SynthesisStrategy` 契約
- **Validation Layer**：評估 sufficiency / consistency / freshness，決定紅綠燈
- **Presentation Layer**：組出面向 API 的 `QueryResponse`

更多細節見 [`docs/architecture.md`](docs/architecture.md)、[`docs/responsibility-distribution.md`](docs/responsibility-distribution.md)。

### 路由契約（Wave 4 後的最終狀態）

PolicyRegistry 與所有下游 layer 的 primary key 是 `(Intent, frozenset[TopicTag])`，不再依賴任何字串式的 `question_type` 推導：

- `Intent`：`NEWS_DIGEST` / `EARNINGS_REVIEW` / `VALUATION_CHECK` / `DIVIDEND_ANALYSIS` / `FINANCIAL_HEALTH` / `TECHNICAL_VIEW` / `INVESTMENT_ASSESSMENT`
- `TopicTag`：受控詞彙表（航運、電價、AI、月營收…），共 31 個 tag
- `resolve_by_tags(intent, controlled_tags)` 回傳單一 `QueryPolicy`，帶著預設時間窗、必要／偏好 facets、retrieval profile key、cove_eligible 旗標
- PolicyRegistry 保留全部 27 支 policy；所有 match 走 exact → partial → generic → fallback 四段式降級

### Legacy Sunset 歷程（Wave 1 → Wave 4）

專案經歷四波大型 legacy 下架，目的是把「產品 scope 之外的路徑」整條刪掉，而不是藏起來：

| Wave | 主題 | 動作 |
|------|------|------|
| **Wave 1** | Forecast family sunset | 刪除預測／目標價路徑（`ForecastBlock`、forecast 相關 synthesis strategy） |
| **Wave 2** | Valuation sunset | 刪除估值深度分析（`ValuationCheckStrategy`、`PE/PB` profile、相關 tests） |
| **Wave 3** | Dividend / earnings deep-dive sunset | 拔掉 7 支 dividend / earnings 深度分析 synthesis strategy 與 dual-signal validation profile |
| **Wave 4** | Question-type fallback sunset | 移除 `StructuredQuery.question_type`、`QUESTION_TYPE_TO_INTENT`、`QUESTION_TYPE_FALLBACK_TOPIC_TAGS`；PolicyRegistry 改以 `(Intent, frozenset[TopicTag])` 作主鍵 |

每一波都包含：
1. 畫 blast radius map
2. 刪 source files
3. 改 enums/models/registry
4. 刪或重寫對應的 validation profile、synthesis strategy
5. 刪測試或把測試改寫到新契約
6. pytest 全套驗證

**目前狀態**：219 tests 通過；`src/` 內已無 legacy shim，question_type 只以 private `rule_signature` 的形式存活在 `InputLayer._detect_question_type` 內部，不外洩。

### 技術棧

| 層面 | 選擇 |
|------|------|
| Python | 3.12+ |
| Web framework | FastAPI |
| 資料庫 | PostgreSQL + pgvector |
| LLM | OpenAI Responses API（可關掉改走 rule-based） |
| 股市資料 | FinMind、TWSE |
| 新聞 | Google News RSS（可關） |
| 模型層 | Pydantic v2 |
| 測試 | pytest |

---

## Part 2 — 開發者操作手冊

### 前置需求

- Python 3.12 或以上
- PowerShell 5.1+ / PowerShell 7+
- Docker Desktop（給 local PostgreSQL）
- FinMind API token（需要真實資料時）
- OpenAI API key（需要 LLM generation 時）

**部署平台**

| 層次 | 平台 | 設定檔 |
|------|------|--------|
| 前端 | Streamlit Cloud | `frontend/app.py`；在 App Settings > Secrets 填入 `BACKEND_URL` |
| 後端（生產） | Vercel | `vercel.json` + `api/index.py` |
| 後端（本機） | uvicorn | `啟動後端.bat` 或 `railway.json`（Railway 備用） |

### Quickstart：Local Python + Docker PostgreSQL

開發情境最順的組合。

```powershell
# 1. 虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安裝（含 dev 依賴以便跑 pytest）
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 3. 環境變數
Copy-Item .env.example .env
# 編輯 .env，至少填：
#   FINMIND_API_TOKEN=your_token
#   OPENAI_API_KEY=        # 留空會改走 rule-based synthesis

# 4. 啟動 PostgreSQL
docker compose up -d postgres

# 5. 啟動 API
python -m uvicorn llm_stock_system.api.app:create_app --factory --reload

# 6. Health check
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### Quickstart：全 Docker

```powershell
Copy-Item .env.example .env   # 填 token
docker compose up --build
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### 主要環境變數

| 變數 | 用途 |
|------|------|
| `DATABASE_URL` | PostgreSQL 連線字串（預設讀 `docker-compose.yml` 的 service） |
| `USE_POSTGRES_MARKET_DATA` | 啟用 PostgreSQL-backed retrieval |
| `FALLBACK_TO_SAMPLE_DATA` | PostgreSQL／上游不可用時 fallback 到內建 sample |
| `FINMIND_API_TOKEN` | FinMind 資料存取 |
| `FINMIND_SYNC_ON_QUERY` | 查詢時現抓資料 |
| `NEWS_PIPELINE_ENABLED` | 啟用多來源新聞 |
| `GOOGLE_NEWS_RSS_ENABLED` | 新聞 pipeline 加上 Google News RSS |
| `OPENAI_API_KEY` | 啟用 OpenAI Responses generation |
| `MODEL_NAME` | Generation layer 使用的 OpenAI model |
| `PRELIMINARY_LLM_ANSWERS_ENABLED` | 允許早期草稿生成 |
| `LOW_CONFIDENCE_WARMUP_ENABLED` | 低信心時二次 hydration |
| `MAX_RETRIEVAL_DOCS` | 送進 governance／synthesis 的文件上限 |
| `MIN_GREEN_CONFIDENCE` / `MIN_YELLOW_CONFIDENCE` | 紅綠燈閾值 |

### 執行測試

```powershell
pytest                           # 全套
pytest tests/test_pipeline.py    # 單檔
pytest -k digest                 # 按 keyword 篩選
```

目前 status：**219 passed**。

### DB Schema Migration

完整 schema 變更流程、欄位對照、rollout 順序見 [`docs/SCHEMA_MIGRATION_GUIDE.md`](docs/SCHEMA_MIGRATION_GUIDE.md)。SQL 初始化腳本在 `db/sql/`。

### 資料同步（PowerShell helper）

```powershell
# 股票主檔
.\scripts\sync_finmind.ps1 -StockInfo

# 單一 ticker 近 30 天股價
.\scripts\sync_finmind.ps1 -Ticker 2330 -Days 30

# 完整資料集（股價+財報+股利+新聞）
.\scripts\sync_finmind.ps1 -Ticker 2603 -Days 400 -Fundamentals -Dividend -News

# 融資融券
.\scripts\sync_finmind.ps1 -Ticker 2330 -Days 30 -Margin

# 新聞關鍵字擴充
.\scripts\sync_finmind.ps1 -Ticker 3680 -Days 30 -News -NewsKeyword ASML -NewsKeyword "advanced packaging"
```

### API 端點

- `GET /api/health` — 運行模式、feature flags
- `POST /api/query` — body: `{query, ticker?, topic?, timeRange?}`
- `GET /api/sources/{query_id}` — 查詢當時引用的來源

Health endpoint 會回：

- `mode`：`sample-only` / `sample-fallback` / `postgres+finmind`
- `llmMode`：`rule-based` / `openai-responses`
- `openaiConfigured` / `finmindApiConfigured` / `newsPipelineEnabled`

### 查詢範例

```powershell
# 用 helper
.\scripts\query.ps1 -Query "台積電這週發生了什麼？" -Ticker 2330 -Topic news -TimeRange 7d

# 直接打 API
$body = @{
  query = "長榮 2603 紅海航線受阻的最新情況？"
  ticker = "2603"
  topic = "news"
  timeRange = "7d"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/query `
  -ContentType "application/json" `
  -Body $body
```

### Troubleshooting

**`/api/health` 顯示 `sample-fallback`**
PostgreSQL + FinMind 路徑掛了，退回內建 sample。檢查：
- PostgreSQL 是否真的起來
- `DATABASE_URL` 指向正確
- `FINMIND_API_TOKEN` 有設
- `FALLBACK_TO_SAMPLE_DATA` 是否把 startup error 蓋掉了

**OpenAI generation 沒啟用**
- `OPENAI_API_KEY` 是否在 `.env`
- `MODEL_NAME` 是否有效
- `/api/health` 的 `llmMode` 是否為 `openai-responses`

**PowerShell 擋 script 執行**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\query.ps1 -Query "..."
```

**完全不想跑 PostgreSQL**
```env
USE_POSTGRES_MARKET_DATA=false
FALLBACK_TO_SAMPLE_DATA=true
```

### 專案結構

```text
.
├── api/
│   └── index.py              # Vercel Serverless Function 入口（生產後端）
├── db/sql/                       # PostgreSQL 初始化腳本（001–005）
├── docs/                         # 架構、migration、phase 重構紀錄
│   ├── architecture.md
│   ├── responsibility-distribution.md
│   ├── P0_IMPLEMENTATION_STATUS.md
│   ├── SCHEMA_MIGRATION_GUIDE.md
│   ├── validation-profile-design-decisions.md
│   └── ...（phase2–4 重構紀錄、drawio 架構圖）
├── frontend/                     # Streamlit 前端
│   ├── app.py                    # UI 主體，透過 HTTP 呼叫後端 API
│   └── requirements.txt
├── scripts/                      # 資料同步與查詢 PowerShell helpers
│   ├── sync_finmind.ps1
│   ├── query.ps1
│   ├── get_sources.ps1
│   ├── backfill_embeddings.py
│   └── seed_documents.py
├── src/llm_stock_system/         # 主套件（pip install -e . 安裝）
│   ├── api/                      # FastAPI app 與 routes
│   │   ├── app.py
│   │   └── routes.py
│   ├── adapters/                 # 外部服務與持久化 adapter
│   │   ├── augmentation/         # 各 Intent 的 augmentation 策略
│   │   ├── synthesis/            # 各 Intent 的 synthesis 策略
│   │   ├── finmind.py            # FinMind API client
│   │   ├── twse_financial.py     # TWSE openapi client（月營收、PE/PB）
│   │   ├── news_pipeline.py      # 多來源新聞 pipeline
│   │   ├── openai_classifier.py  # LLM-first query 分類器
│   │   ├── openai_responses.py   # OpenAI Responses API
│   │   ├── postgres_market_data.py  # 市場資料 gateway（sync + query）
│   │   ├── postgres_query_log_store.py
│   │   ├── repositories.py
│   │   └── vector_retrieval.py
│   ├── core/                     # 共用 domain 物件
│   │   ├── config.py             # Settings（pydantic-settings）
│   │   ├── enums.py              # Intent、DataFacet、TopicTag 等
│   │   ├── interfaces.py
│   │   ├── models.py             # StructuredQuery、QueryResponse、FacetSpec…
│   │   ├── query_policy.py       # PolicyRegistry（27 支 policy）
│   │   └── validation_profiles.py
│   ├── digest/                   # Digest 產品線專屬邏輯
│   │   └── policy/               # Refusal boundary、closed_tables、terms
│   ├── layers/                   # 六層 pipeline 實作
│   │   ├── input_layer.py
│   │   ├── digest_input_layer.py
│   │   ├── retrieval_layer.py
│   │   ├── data_governance_layer.py
│   │   ├── augmentation_layer.py
│   │   ├── generation_layer.py
│   │   ├── validation_layer.py
│   │   └── presentation_layer.py
│   ├── orchestrator/
│   │   └── pipeline.py           # 六層組裝與 hydration 觸發
│   ├── services/
│   │   ├── query_data_hydrator.py  # 並行 facet sync（ThreadPoolExecutor）
│   │   ├── document_chunker.py
│   │   └── embedding_service.py
│   ├── workers/
│   │   ├── sync_market_data.py   # CLI sync 工具（--fundamentals、--news…）
│   │   └── ingestion_worker.py
│   ├── prompts/                  # system_prompt.md
│   └── sample_data/              # Fallback 用內建範例資料
├── tests/                        # pytest 測試（219 passed）
│   ├── digest/
│   └── test_*.py
├── .env.example
├── digest_refusal_boundary_v1_1.md  # Refusal boundary 規格 v1.1
├── docker-compose.yml
├── nixpacks.toml                 # Railway 部署（Nixpacks builder）
├── Procfile                      # Railway 部署（start command）
├── pyproject.toml
├── pytest.ini
├── railway.json                  # Railway 部署設定
├── requirements.txt              # 生產依賴（供部署平台讀取）
├── 啟動後端.bat                  # Windows 一鍵啟動後端（uvicorn）
└── 啟動前端.bat                  # Windows 一鍵啟動前端（streamlit）
```
