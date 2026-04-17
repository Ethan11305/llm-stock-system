from fastapi import FastAPI

from llm_stock_system.adapters.finmind import FinMindClient
from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.news_pipeline import (
    FinMindNewsProvider,
    GoogleNewsRssProvider,
    MultiSourceNewsPipeline,
)
from llm_stock_system.adapters.openai_responses import OpenAIResponsesSynthesisClient
from llm_stock_system.adapters.postgres_market_data import (
    FinMindPostgresGateway,
    PostgresMarketDocumentRepository,
    PostgresStockResolver,
)
from llm_stock_system.adapters.repositories import InMemoryDocumentRepository, InMemoryQueryLogStore
from llm_stock_system.adapters.twse_financial import TwseCompanyFinancialClient
from llm_stock_system.core.config import get_settings
from llm_stock_system.layers.data_governance_layer import DataGovernanceLayer
from llm_stock_system.layers.generation_layer import GenerationLayer
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.retrieval_layer import HybridRetrievalLayer, RetrievalLayer
from llm_stock_system.layers.validation_layer import ValidationLayer
from llm_stock_system.orchestrator.pipeline import QueryPipeline
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator
from llm_stock_system.sample_data.documents import SAMPLE_DOCUMENTS

from .routes import router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.state.openai_configured = bool(settings.openai_api_key)
    app.state.finmind_api_configured = bool(settings.finmind_api_token)
    app.state.preliminary_llm_enabled = bool(settings.preliminary_llm_answers_enabled and settings.openai_api_key)
    app.state.low_confidence_warmup_enabled = False
    app.state.llm_mode = "rule-based"
    app.state.news_pipeline_enabled = False
    app.state.news_provider_names = []

    sample_repository = InMemoryDocumentRepository(SAMPLE_DOCUMENTS)
    document_repository = sample_repository
    stock_resolver = None
    llm_client = RuleBasedSynthesisClient()
    query_hydrator = None
    vector_adapter = None
    embedding_service = None

    if settings.openai_api_key:
        llm_client = OpenAIResponsesSynthesisClient(
            api_key=settings.openai_api_key,
            model_name=settings.model_name,
            base_url=settings.openai_base_url,
            preliminary_answers_enabled=settings.preliminary_llm_answers_enabled,
        )
        app.state.llm_mode = "openai-responses"

    if settings.use_postgres_market_data:
        try:
            finmind_client = FinMindClient(
                base_url=settings.finmind_base_url,
                api_token=settings.finmind_api_token,
            )
            twse_financial_client = TwseCompanyFinancialClient(
                base_url=settings.twse_company_financial_url,
                monthly_revenue_url=settings.twse_monthly_revenue_url,
            )
            news_pipeline = None
            if settings.news_pipeline_enabled:
                news_providers = [FinMindNewsProvider(finmind_client)]
                if settings.google_news_rss_enabled:
                    news_providers.append(GoogleNewsRssProvider(base_url=settings.google_news_rss_base_url))
                news_pipeline = MultiSourceNewsPipeline(news_providers)
            market_gateway = FinMindPostgresGateway(
                database_url=settings.database_url,
                finmind_client=finmind_client,
                twse_financial_client=twse_financial_client,
                news_pipeline=news_pipeline,
                sync_on_query=settings.finmind_sync_on_query,
                stock_info_refresh_hours=settings.stock_info_refresh_hours,
            )
            market_gateway.ping()
            document_repository = PostgresMarketDocumentRepository(market_gateway)
            stock_resolver = PostgresStockResolver(market_gateway)
            # P0 Embedding Pipeline（可選，需 openai_api_key + embedding_enabled）
            if settings.embedding_enabled and settings.openai_api_key:
                try:
                    from sqlalchemy import create_engine as _create_engine
                    from llm_stock_system.services.embedding_service import EmbeddingService
                    from llm_stock_system.adapters.vector_retrieval import VectorRetrievalAdapter

                    _db_engine = _create_engine(settings.database_url)
                    embedding_service = EmbeddingService(
                        openai_api_key=settings.openai_api_key,
                        model=settings.embedding_model,
                        batch_size=settings.embedding_batch_size,
                        db_engine=_db_engine,
                    )
                    vector_adapter = VectorRetrievalAdapter(
                        embedding_service=embedding_service,
                        db_engine=_db_engine,
                    )
                    app.state.embedding_enabled = True
                except Exception:
                    # Embedding 初始化失敗不阻擋主流程
                    embedding_service = None
                    vector_adapter = None
                    app.state.embedding_enabled = False
            else:
                app.state.embedding_enabled = False

            query_hydrator = QueryDataHydrator(
                market_gateway,
                low_confidence_warmup_enabled=settings.low_confidence_warmup_enabled,
                low_confidence_warmup_threshold=settings.low_confidence_warmup_threshold,
                follow_up_cooldown_hours=settings.low_confidence_warmup_cooldown_hours,
                embedding_service=embedding_service,
            )
            # P0: 注入 document_repository，讓 hydrator 在 embed 前能先 upsert documents
            query_hydrator._document_repository = document_repository
            app.state.low_confidence_warmup_enabled = bool(settings.low_confidence_warmup_enabled)
            app.state.news_pipeline_enabled = bool(news_pipeline and news_pipeline.provider_names)
            app.state.news_provider_names = news_pipeline.provider_names if news_pipeline else []
            app.state.market_gateway = market_gateway
            app.state.runtime_mode = "postgres+finmind"
        except Exception:
            if not settings.fallback_to_sample_data:
                raise
            document_repository = sample_repository
            app.state.runtime_mode = "sample-fallback"
    else:
        app.state.runtime_mode = "sample-only"

    # P0：若 vector_adapter 已初始化，使用 HybridRetrievalLayer；否則使用原有 RetrievalLayer
    if vector_adapter is not None:
        retrieval_layer = HybridRetrievalLayer(
            document_repository=document_repository,
            vector_adapter=vector_adapter,
            max_documents=settings.max_retrieval_docs,
            semantic_weight=settings.hybrid_retrieval_semantic_weight,
            metadata_weight=settings.hybrid_retrieval_metadata_weight,
        )
    else:
        retrieval_layer = RetrievalLayer(
            document_repository=document_repository,
            max_documents=settings.max_retrieval_docs,
        )

    pipeline = QueryPipeline(
        input_layer=InputLayer(stock_resolver=stock_resolver),
        retrieval_layer=retrieval_layer,
        data_governance_layer=DataGovernanceLayer(),
        generation_layer=GenerationLayer(
            llm_client=llm_client,
            prompt_path=settings.prompt_path,
        ),
        validation_layer=ValidationLayer(
            min_green_confidence=settings.min_green_confidence,
            min_yellow_confidence=settings.min_yellow_confidence,
        ),
        presentation_layer=PresentationLayer(),
        query_log_store=InMemoryQueryLogStore(),
        query_hydrator=query_hydrator,
    )

    app.state.pipeline = pipeline
    app.include_router(router, prefix=settings.api_prefix)
    return app
