from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Request

from llm_stock_system.core.models import (
    QueryLogDetail,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)

router = APIRouter()


@router.get("/health")
def healthcheck(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "mode": getattr(request.app.state, "runtime_mode", "unknown"),
        "llmMode": getattr(request.app.state, "llm_mode", "unknown"),
        "openaiConfigured": getattr(request.app.state, "openai_configured", False),
        "finmindApiConfigured": getattr(request.app.state, "finmind_api_configured", False),
        "queryHydrationEnabled": getattr(request.app.state, "runtime_mode", "") == "postgres+finmind",
        "preliminaryLlmEnabled": getattr(request.app.state, "preliminary_llm_enabled", False),
        "lowConfidenceWarmupEnabled": getattr(request.app.state, "low_confidence_warmup_enabled", False),
        "newsPipelineEnabled": getattr(request.app.state, "news_pipeline_enabled", False),
        "newsProviders": getattr(request.app.state, "news_provider_names", []),
        "embeddingEnabled": getattr(request.app.state, "embedding_enabled", False),
        "retrievalMode": "hybrid" if getattr(request.app.state, "embedding_enabled", False) else "metadata_only",
        "digestEnabled": getattr(request.app.state, "digest_enabled", False),
        "digestClassifierEnabled": getattr(request.app.state, "digest_classifier_enabled", False),
    }


@router.post("/query", response_model=QueryResponse)
def query_stock(request_body: QueryRequest, request: Request) -> QueryResponse:
    pipeline = request.app.state.pipeline
    return pipeline.handle_query(request_body)


@router.get("/price/{ticker}")
def get_price(ticker: str, days: int = 30, request: Request = None) -> dict:
    """回傳某支股票近 N 天的 OHLCV 資料，供前端股價圖使用。"""
    market_gateway = getattr(request.app.state, "market_gateway", None)
    if market_gateway is None:
        raise HTTPException(status_code=503, detail="Market gateway not available")
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        bars = market_gateway.get_price_bars(ticker, start_date, end_date)
        return {
            "ticker": ticker,
            "bars": [
                {
                    "date": b.trading_date.isoformat(),
                    "open": float(b.open_price),
                    "high": float(b.high_price),
                    "low": float(b.low_price),
                    "close": float(b.close_price),
                    "volume": b.trading_volume,
                }
                for b in sorted(bars, key=lambda x: x.trading_date)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/digest/query", response_model=QueryResponse)
def digest_query(request_body: QueryRequest, request: Request) -> QueryResponse:
    """Single Stock Digest 產品線進入點。

    與 /query 使用同一組 retrieval/generation/validation layer，差異在於：
      * InputLayer 使用 DigestInputLayer（強制 query_profile=SINGLE_STOCK_DIGEST、
        未指定時間窗時預設 7d）
      * classifier 主 + rule fallback 路徑解析語意
    若 digest_enabled=False 或初始化失敗，回 503 讓呼叫端明確降級。
    """
    digest_pipeline = getattr(request.app.state, "digest_pipeline", None)
    if digest_pipeline is None:
        raise HTTPException(status_code=503, detail="Digest pipeline is not enabled")
    return digest_pipeline.handle_query(request_body)


@router.get("/sources/{query_id}", response_model=SourceResponse)
def query_sources(query_id: str, request: Request) -> SourceResponse:
    store = request.app.state.query_log_store
    response = store.get_sources(query_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Query sources not found")
    return response


@router.get("/query-log/{query_id}", response_model=QueryLogDetail)
def query_log(query_id: str, request: Request) -> QueryLogDetail:
    """可追溯閉環：回查某個 query_id 的 structured_query + response snapshot + warnings。

    供 QA、digest 產品前端、回歸分析用。
    """
    store = request.app.state.query_log_store
    detail = store.get_query_log(query_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Query log not found")
    return detail
