# LLM Stock Advisory System

This repository turns the product specification into a runnable system scaffold that follows the six-layer architecture:

1. Input Layer
2. Retrieval Layer
3. Data Governance Layer
4. Generation Layer
5. Validation Layer
6. Presentation Layer

## Runtime Services

- API service: FastAPI endpoints for `/api/query` and `/api/sources/{query_id}`
- FinMind integration: stock master, daily prices, financial statements, dividends, and news
- Multi-source news pipeline: FinMind ticker news plus RSS-based theme/news expansion
- PostgreSQL + pgvector: persistence layer for documents, embeddings, stock master data, and market datasets

## Local Run

1. Create a virtual environment.
2. Install dependencies with `pip install -e .`
3. Copy `.env.example` to `.env`
4. Set `FINMIND_API_TOKEN` in `.env`
5. Set `OPENAI_API_KEY` in `.env` if you want the generation layer to use OpenAI Responses API
6. Start PostgreSQL with `docker compose up postgres`
7. Start the API with `uvicorn llm_stock_system.api.app:create_app --factory --reload`

## FinMind + PostgreSQL

When PostgreSQL and FinMind are available, the app now uses real ingestion for:

- `TaiwanStockInfo` for stock name resolution
- `TaiwanStockPrice` for daily OHLCV price history
- `TaiwanStockFinancialStatements` for EPS and earnings-related answers
- `TaiwanStockDividend` for dividend and announcement-related answers
- `TaiwanStockNews` for market news when the current FinMind plan allows access
- `Google News RSS` as a first-version backup/expansion source for theme-driven news retrieval

In `postgres+finmind` mode, query-time retrieval no longer falls back to sample documents. If real data is unavailable, the system returns an insufficient-data answer instead of mixing in mock content.

## Manual Sync

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 -StockInfo
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 -Ticker 2344 -Days 30
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 `
  -Ticker 2603 `
  -Days 400 `
  -Fundamentals `
  -Dividend `
  -News
```

You can also expand news sync with theme keywords:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 `
  -Ticker 3680 `
  -Days 30 `
  -News `
  -NewsKeyword ASML `
  -NewsKeyword 半導體設備
```

## Query the System

Start the API first:

```powershell
python -m uvicorn llm_stock_system.api.app:create_app --factory --reload
```

Check runtime mode:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

When PostgreSQL and FinMind are connected, `mode` should be `postgres+finmind`.
When `OPENAI_API_KEY` is configured, `llmMode` should be `openai-responses`.
When the first-version news pipeline is enabled, `/api/health` also reports `newsPipelineEnabled` and `newsProviders`.

Then query it from another PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\query.ps1 -Query "2330 最近 7 天有什麼重點？"
```

Fetch the evidence list for a previous query:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\get_sources.ps1 -QueryId "<query_id>"
```
