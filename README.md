# LLM Stock Advisory System

A FastAPI-based LLM stock advisory scaffold for Taiwan equities. The project combines a six-layer query pipeline with market-data ingestion, source filtering, and grounded answer generation.

## What This Repository Includes

- FastAPI endpoints for health checks, grounded stock queries, and source lookup
- A six-layer architecture: Input, Retrieval, Data Governance, Generation, Validation, Presentation
- PostgreSQL + pgvector storage for documents, market datasets, and retrieval
- FinMind integration for stock master, price history, financial statements, dividends, margin data, and news
- TWSE integration for company financial and monthly revenue data
- Multi-source news expansion with Google News RSS
- Optional OpenAI Responses API integration for the generation layer
- PowerShell helper scripts for sync and query workflows

## Architecture Overview

High-level architecture notes live in [docs/architecture.md](docs/architecture.md), and the product-level requirements live in [PRODUCT_SPEC.md](PRODUCT_SPEC.md).

Pipeline flow:

1. Input Layer
2. Retrieval Layer
3. Data Governance Layer
4. Generation Layer
5. Validation Layer
6. Presentation Layer

## Project Structure

```text
.
|-- db/
|   `-- sql/                 # PostgreSQL initialization scripts
|-- docs/
|   `-- architecture.md      # Architecture notes
|-- scripts/
|   |-- get_sources.ps1      # Fetch sources for a previous query
|   |-- query.ps1            # Send a query to the API
|   `-- sync_finmind.ps1     # Sync market data into PostgreSQL
|-- src/llm_stock_system/
|   |-- api/                 # FastAPI app and routes
|   |-- adapters/            # External services and persistence adapters
|   |-- layers/              # Six-layer pipeline implementation
|   |-- orchestrator/        # Pipeline wiring
|   `-- workers/             # Data sync worker entrypoints
|-- tests/                   # Pytest test suite
|-- .env.example             # Environment variable template
|-- docker-compose.yml       # Local PostgreSQL and API container setup
|-- pyproject.toml           # Package metadata and dependencies
`-- README.md
```

## Requirements

Before running the project locally, make sure you have:

- Python 3.12 or newer
- PowerShell 5.1+ or PowerShell 7+
- Docker Desktop or another Docker runtime
- A FinMind API token if you want real market data sync
- An OpenAI API key if you want the generation layer to use OpenAI Responses

## Quick Start

### Option 1: Local Python Environment + Docker PostgreSQL

This is the most convenient setup for development.

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install the package in editable mode.

```powershell
python -m pip install --upgrade pip
pip install -e .
```

3. Copy the environment template.

```powershell
Copy-Item .env.example .env
```

4. Edit `.env`.

Minimum recommended settings:

```env
FINMIND_API_TOKEN=your_finmind_token
OPENAI_API_KEY=
```

Notes:

- `FINMIND_API_TOKEN` is needed for real market-data sync.
- `OPENAI_API_KEY` is optional. If left empty, the app uses the rule-based synthesis client.
- `DATABASE_URL` defaults to the PostgreSQL service defined in `docker-compose.yml`.

5. Start PostgreSQL.

```powershell
docker compose up -d postgres
```

6. Start the API.

```powershell
python -m uvicorn llm_stock_system.api.app:create_app --factory --reload
```

7. Verify the service.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### Option 2: Run the Full Stack with Docker Compose

If you want both PostgreSQL and the API inside containers:

1. Copy the environment template.

```powershell
Copy-Item .env.example .env
```

2. Edit `.env` with your token settings.

3. Start everything.

```powershell
docker compose up --build
```

4. Check health.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

## Configuration Guide

The project loads environment variables from `.env`. The template is documented in [.env.example](.env.example).

Important settings:

- `DATABASE_URL`: PostgreSQL connection string
- `USE_POSTGRES_MARKET_DATA`: Enables PostgreSQL-backed market retrieval
- `FALLBACK_TO_SAMPLE_DATA`: Falls back to in-memory sample documents if PostgreSQL or upstream data is unavailable
- `FINMIND_API_TOKEN`: Token for FinMind data access
- `FINMIND_SYNC_ON_QUERY`: Syncs fresh market data on query when possible
- `NEWS_PIPELINE_ENABLED`: Enables multi-source news retrieval
- `GOOGLE_NEWS_RSS_ENABLED`: Adds Google News RSS to the news pipeline
- `OPENAI_API_KEY`: Enables the OpenAI Responses generation client
- `MODEL_NAME`: OpenAI model name used by the generation layer
- `PRELIMINARY_LLM_ANSWERS_ENABLED`: Enables early draft generation when OpenAI is configured
- `LOW_CONFIDENCE_WARMUP_ENABLED`: Enables follow-up hydration for low-confidence queries
- `MAX_RETRIEVAL_DOCS`: Maximum number of retrieved documents passed into governance and synthesis
- `MIN_GREEN_CONFIDENCE` and `MIN_YELLOW_CONFIDENCE`: Thresholds for validation confidence lights

## Runtime Modes

The `/api/health` endpoint reports the current runtime mode.

- `sample-only`: PostgreSQL market data is disabled
- `sample-fallback`: PostgreSQL market data is enabled, but the app fell back to sample documents because the real data path was unavailable
- `postgres+finmind`: PostgreSQL is available and the real market-data path is active

The same endpoint also reports:

- `llmMode`: `rule-based` or `openai-responses`
- `openaiConfigured`: whether `OPENAI_API_KEY` is set
- `finmindApiConfigured`: whether `FINMIND_API_TOKEN` is set
- `newsPipelineEnabled`: whether news enrichment is active
- `newsProviders`: which news providers are enabled

## Sync Market Data Manually

The repository includes a helper script at [scripts/sync_finmind.ps1](scripts/sync_finmind.ps1). It wraps the Python worker in `llm_stock_system.workers.sync_market_data`.

### Refresh stock master data

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 -StockInfo
```

### Sync recent price history for a ticker

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 -Ticker 2330 -Days 30
```

If you pass `-Ticker` without other non-price switches, price history sync runs by default.

### Sync a broader dataset for one ticker

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 `
  -Ticker 2603 `
  -Days 400 `
  -Fundamentals `
  -Dividend `
  -News
```

### Sync TWSE monthly revenue and valuation data

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 `
  -Ticker 2330 `
  -MonthlyRevenue `
  -Valuation
```

### Sync margin data

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 `
  -Ticker 2330 `
  -Days 30 `
  -Margin
```

### Expand news retrieval with extra keywords

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_finmind.ps1 `
  -Ticker 3680 `
  -Days 30 `
  -News `
  -NewsKeyword ASML `
  -NewsKeyword "advanced packaging"
```

## Query the API

### Health Check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### Query with the PowerShell helper

Use [scripts/query.ps1](scripts/query.ps1) to send queries to `/api/query`.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\query.ps1 `
  -Query "What changed for TSMC this week?" `
  -Ticker 2330 `
  -Topic news `
  -TimeRange 7d
```

Supported helper parameters:

- `-Query`: Required natural-language user question
- `-Ticker`: Optional stock ticker override
- `-Topic`: One of `news`, `earnings`, `announcement`, `composite`
- `-TimeRange`: Optional string label such as `7d`, `30d`, or another range the pipeline can interpret
- `-TimeoutSec`: Request timeout in seconds
- `-Raw`: Prints the raw JSON response

### Query directly with `Invoke-RestMethod`

```powershell
$body = @{
  query = "What changed for 2330 this week?"
  ticker = "2330"
  topic = "news"
  timeRange = "7d"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/query `
  -ContentType "application/json" `
  -Body $body
```

### Fetch sources for a previous query

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\get_sources.ps1 -QueryId "<query_id>"
```

You can also call the endpoint directly:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/sources/<query_id>
```

## API Endpoints

- `GET /api/health`: Runtime mode and feature flags
- `POST /api/query`: Accepts `query`, optional `ticker`, optional `topic`, and optional `timeRange`
- `GET /api/sources/{query_id}`: Returns source citations for a previous query

## Testing

Run the test suite with:

```powershell
pytest
```

If you only want to run a single test module:

```powershell
pytest tests/test_pipeline.py
```

## Troubleshooting

### `/api/health` shows `sample-fallback`

This means the app could not activate the PostgreSQL + FinMind path.

Check:

- PostgreSQL is running
- `DATABASE_URL` points to the correct database
- `FINMIND_API_TOKEN` is configured if you expect real upstream data
- `FALLBACK_TO_SAMPLE_DATA` is not masking a startup error you want to surface

### OpenAI generation is not enabled

Check:

- `OPENAI_API_KEY` is set in `.env`
- `MODEL_NAME` is a valid model for your account
- `/api/health` reports `llmMode` as `openai-responses`

### PowerShell blocks script execution

Run the helper script with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\query.ps1 -Query "What changed for 2330 this week?"
```

### You want to run without PostgreSQL

Set the following in `.env`:

```env
USE_POSTGRES_MARKET_DATA=false
FALLBACK_TO_SAMPLE_DATA=true
```

This will run the project in `sample-only` mode.

## License

This repository includes a [LICENSE](LICENSE) file. Review it before redistributing or reusing the code.
