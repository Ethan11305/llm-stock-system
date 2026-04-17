from fastapi import APIRouter, HTTPException, Request

from llm_stock_system.core.models import QueryRequest, QueryResponse, SourceResponse

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
    }


@router.post("/query", response_model=QueryResponse)
def query_stock(request_body: QueryRequest, request: Request) -> QueryResponse:
    pipeline = request.app.state.pipeline
    return pipeline.handle_query(request_body)


@router.get("/sources/{query_id}", response_model=SourceResponse)
def query_sources(query_id: str, request: Request) -> SourceResponse:
    pipeline = request.app.state.pipeline
    response = pipeline.get_sources(query_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Query sources not found")
    return response
