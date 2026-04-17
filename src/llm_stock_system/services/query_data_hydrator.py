from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import time
from threading import Lock, Thread
from typing import TYPE_CHECKING

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import HydrationResult, StructuredQuery, ValidationResult

if TYPE_CHECKING:
    from llm_stock_system.core.interfaces import DocumentRepository
    from llm_stock_system.services.document_chunker import DocumentChunker
    from llm_stock_system.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FacetSyncRule:
    gateway_method: str
    requires_dates: bool = False
    default_window_days: int | None = None
    default_window_years: int | None = None
    min_window_days: int | None = None
    min_window_years: int | None = None
    fallback_method: str | None = None
    priority: str = "medium"


@dataclass
class FacetSyncResult:
    facet: DataFacet
    success: bool
    exception: Exception | None = None
    duration_ms: float = 0.0
    rows_synced: int | None = None


class QueryDataHydrator:
    """Fetches missing market data based on intent and facet requirements."""

    FACET_SYNC_MAP: dict[DataFacet, FacetSyncRule] = {
        DataFacet.PRICE_HISTORY: FacetSyncRule(
            gateway_method="sync_price_history",
            requires_dates=True,
            default_window_days=120,
            min_window_days=60,
            priority="high",
        ),
        DataFacet.FINANCIAL_STATEMENTS: FacetSyncRule(
            gateway_method="sync_financial_statements",
            requires_dates=True,
            default_window_years=2,
            min_window_years=1,
            priority="high",
        ),
        DataFacet.MONTHLY_REVENUE: FacetSyncRule(
            gateway_method="sync_monthly_revenue_points",
            priority="medium",
        ),
        DataFacet.DIVIDEND: FacetSyncRule(
            gateway_method="sync_dividend_policies",
            requires_dates=True,
            default_window_years=2,
            priority="high",
        ),
        DataFacet.BALANCE_SHEET: FacetSyncRule(
            gateway_method="sync_balance_sheet_items",
            requires_dates=True,
            default_window_years=3,
            priority="medium",
        ),
        DataFacet.CASH_FLOW: FacetSyncRule(
            gateway_method="sync_cash_flow_statements",
            requires_dates=True,
            default_window_years=3,
            priority="medium",
        ),
        DataFacet.PE_VALUATION: FacetSyncRule(
            gateway_method="sync_pe_valuation_points",
            priority="high",
        ),
        DataFacet.MARGIN_DATA: FacetSyncRule(
            gateway_method="sync_margin_purchase_short_sale",
            requires_dates=True,
            default_window_days=180,
            priority="low",
        ),
        DataFacet.NEWS: FacetSyncRule(
            gateway_method="sync_stock_news",
            requires_dates=True,
            default_window_days=30,
            fallback_method="sync_query_news",
            priority="high",
        ),
    }

    # P1：ThreadPoolExecutor 並行 facet 拉取的預設 worker 數
    # 設定為 6 是因為通常有 9 個 facet，6 個 worker 可以有效並行
    # 上限建議不超過 gateway 的連線池大小（避免 DB 連線耗盡）
    DEFAULT_PARALLEL_WORKERS: int = 6

    def __init__(
        self,
        gateway,
        low_confidence_warmup_enabled: bool = True,
        low_confidence_warmup_threshold: float = 0.80,
        follow_up_cooldown_hours: int = 12,
        run_follow_up_async: bool = True,
        embedding_service: EmbeddingService | None = None,
        chunker: DocumentChunker | None = None,
        parallel_hydration_workers: int = DEFAULT_PARALLEL_WORKERS,
    ) -> None:
        self._gateway = gateway
        self._low_confidence_warmup_enabled = low_confidence_warmup_enabled
        self._low_confidence_warmup_threshold = low_confidence_warmup_threshold
        self._follow_up_cooldown = timedelta(hours=max(follow_up_cooldown_hours, 1))
        self._run_follow_up_async = run_follow_up_async
        self._warmup_lock = Lock()
        self._active_follow_up_tickers: set[str] = set()
        self._last_follow_up_at: dict[str, datetime] = {}
        # P0 Embedding Pipeline：可選，若未設定則跳過 embedding 生成
        self._embedding_service = embedding_service
        self._chunker = chunker
        self._document_repository: DocumentRepository | None = None  # 由 app.py 注入
        # P1 並行 hydration：worker 數（0 或 1 = 退回串行模式）
        self._parallel_workers = max(1, parallel_hydration_workers)

    def hydrate(self, query: StructuredQuery) -> HydrationResult:
        """執行資料補水（Hydration）主流程。

        P1 改造：用 ThreadPoolExecutor 將所有 (ticker × facet) 任務並行執行。
        - 預設 6 個 worker（可透過 parallel_hydration_workers 調整）
        - 任何單一 facet 失敗不阻擋其他 facet
        - 結果聚合方式與串行版本完全一致（向後相容）
        - parallel_hydration_workers=1 可退回串行行為（便於測試）
        """
        result = HydrationResult()
        if not query.ticker:
            return result

        today = datetime.now(timezone.utc).date()
        started_at = time.perf_counter()
        required_facets = set(query.required_facets)
        preferred_facets = set(query.preferred_facets)

        # sync_stock_info 必須先跑完（其他 facet 可能依賴最新的股票清單）
        self._safe_call(getattr(self._gateway, "sync_stock_info", None))

        # 建立所有 (ticker, facet) 任務清單
        tasks: list[tuple[str, DataFacet]] = [
            (ticker, facet)
            for ticker in self._iter_tickers(query)
            for facet in self._ordered_facets(query)
        ]

        if not tasks:
            result.total_duration_ms = (time.perf_counter() - started_at) * 1000
            self._trigger_embedding_async(query)
            return result

        # P1：並行執行所有 facet 拉取
        # max_workers = min(任務數, 設定的 worker 上限)，避免為少量任務開過多 thread
        max_workers = min(len(tasks), self._parallel_workers)
        facet_results: list[FacetSyncResult] = []

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hydrator") as executor:
            future_to_facet = {
                executor.submit(self._sync_facet, facet, ticker, query, today): facet
                for ticker, facet in tasks
            }
            for future in as_completed(future_to_facet):
                try:
                    facet_results.append(future.result())
                except Exception as exc:
                    # 防禦：_sync_facet 內部已有 try/except，這裡是雙重保護
                    facet = future_to_facet[future]
                    facet_results.append(
                        FacetSyncResult(facet=facet, success=False, exception=exc)
                    )

        # 聚合結果（邏輯與原串行版本相同）
        for facet_result in facet_results:
            if facet_result.success:
                result.synced_facets.add(facet_result.facet)
                continue

            facet = facet_result.facet
            result.failed_facets.setdefault(facet, self._format_exception(facet_result.exception))
            if facet in required_facets and facet.value not in result.facet_miss_list:
                result.facet_miss_list.append(facet.value)
            elif facet in preferred_facets and facet.value not in result.preferred_miss_list:
                result.preferred_miss_list.append(facet.value)

        result.total_duration_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            "hydrate 完成：%d 個 facet 並行（workers=%d），耗時 %.0fms",
            len(tasks), max_workers, result.total_duration_ms,
        )

        # P0 Embedding Pipeline：hydration 完成後非同步觸發 embedding 生成
        self._trigger_embedding_async(query)

        return result

    def _trigger_embedding_async(self, query: StructuredQuery) -> None:
        """在背景執行緒中，對本次查詢的文件執行「upsert → embed」完整流程。

        流程：
        1. 從 gateway.build_documents() 取得本次查詢合成的 Document 物件
        2. 先呼叫 document_repository.upsert_documents()，將 Document 持久化到
           documents 表（解決 FK constraint：document_embeddings.document_id
           REFERENCES documents.id）
        3. 再呼叫 embedding_service.embed_and_store()，生成並寫入向量

        設計決策：
        - 只在 embedding_service 已設定時觸發
        - daemon thread 執行，不阻擋主查詢回應
        - 任何失敗只記錄 warning，不影響主流程
        """
        if self._embedding_service is None:
            return

        def _background_embed() -> None:
            try:
                # Step 1: 取得本次查詢合成的文件
                build_documents = getattr(self._gateway, "build_documents", None)
                if not callable(build_documents):
                    return
                documents = build_documents(query)
                if not documents:
                    return

                # Step 2: 先將文件 upsert 到 documents 表（解決 FK constraint）
                # document_embeddings.document_id REFERENCES documents.id，
                # 所以 embed 之前文件必須先存在於 documents 表。
                if self._document_repository is not None:
                    upsert_fn = getattr(self._document_repository, "upsert_documents", None)
                    if callable(upsert_fn):
                        upserted = upsert_fn(documents)
                        logger.debug(
                            "_trigger_embedding_async: upsert %d 篇文件到 documents 表",
                            upserted,
                        )

                # Step 3: 生成並寫入 embedding
                written = self._embedding_service.embed_and_store(
                    documents, self._chunker
                )
                if written:
                    logger.info(
                        "_trigger_embedding_async: ticker=%s 新增 %d 個 chunk embedding",
                        query.ticker,
                        written,
                    )
            except Exception as exc:
                # exc_info=True 確保完整 traceback 被記錄，方便追查 FK constraint 失敗、
                # OpenAI API 錯誤、或 documents/document_embeddings 不一致等問題
                logger.warning(
                    "_trigger_embedding_async 失敗（ticker=%s）：%s",
                    query.ticker,
                    exc,
                    exc_info=True,
                )

        Thread(target=_background_embed, daemon=True).start()

    def _ordered_facets(self, query: StructuredQuery) -> list[DataFacet]:
        requested_facets = set(query.required_facets) | set(query.preferred_facets)
        configured_facets = [facet for facet in self.FACET_SYNC_MAP if facet in requested_facets]
        remaining_facets = sorted(requested_facets - set(configured_facets), key=lambda facet: facet.value)
        return configured_facets + remaining_facets

    def _sync_facet(
        self,
        facet: DataFacet,
        ticker: str,
        query: StructuredQuery,
        today: date,
    ) -> FacetSyncResult:
        rule = self.FACET_SYNC_MAP.get(facet)
        if rule is None:
            return FacetSyncResult(
                facet=facet,
                success=False,
                exception=ValueError(f"Unknown facet: {facet.value}"),
            )

        started_at = time.perf_counter()
        try:
            rows_synced = self._dispatch_facet_sync(rule, facet, ticker, query, today)
            return FacetSyncResult(
                facet=facet,
                success=True,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                rows_synced=rows_synced if isinstance(rows_synced, int) else None,
            )
        except Exception as exc:
            return FacetSyncResult(
                facet=facet,
                success=False,
                exception=exc,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )

    def _dispatch_facet_sync(
        self,
        rule: FacetSyncRule,
        facet: DataFacet,
        ticker: str,
        query: StructuredQuery,
        today: date,
    ) -> int | None:
        if facet == DataFacet.NEWS and query.intent == Intent.NEWS_DIGEST and rule.fallback_method:
            query_news_sync = getattr(self._gateway, rule.fallback_method, None)
            if callable(query_news_sync):
                return query_news_sync(query)

        sync_method = getattr(self._gateway, rule.gateway_method)
        if rule.requires_dates:
            date_range = self._compute_facet_window(facet, query, today)
            if date_range is None:
                raise ValueError(f"Facet {facet.value} requires a date window.")
            start_date, end_date = date_range
            return sync_method(ticker, start_date, end_date)
        return sync_method(ticker)

    def _compute_facet_window(
        self,
        facet: DataFacet,
        query: StructuredQuery,
        today: date,
    ) -> tuple[date, date] | None:
        history_years = max((query.time_range_days + 364) // 365, 2)

        if facet == DataFacet.PRICE_HISTORY:
            baseline_days = 180 if self._is_technical_indicator_query(query) else 120
            return today - timedelta(days=max(query.time_range_days, baseline_days)), today

        if facet == DataFacet.FINANCIAL_STATEMENTS:
            return date(today.year - history_years, 1, 1), today

        if facet == DataFacet.DIVIDEND:
            start_date = max(date(today.year - history_years, 1, 1), today - timedelta(days=730))
            return start_date, today

        if facet in {DataFacet.BALANCE_SHEET, DataFacet.CASH_FLOW}:
            return date(today.year - 3, 1, 1), today

        if facet == DataFacet.MARGIN_DATA:
            return today - timedelta(days=180), today

        if facet == DataFacet.NEWS:
            if self._is_profitability_stability_query(query):
                return date(today.year - 3, 1, 1), today
            return today - timedelta(days=max(query.time_range_days, 30)), today

        if facet in {DataFacet.MONTHLY_REVENUE, DataFacet.PE_VALUATION}:
            return None

        return None

    def _topic_tags(self, query: StructuredQuery) -> set[str]:
        return set(query.topic_tags or [])

    def _is_technical_indicator_query(self, query: StructuredQuery) -> bool:
        tags = self._topic_tags(query)
        return query.intent == Intent.TECHNICAL_VIEW and not bool(tags & {"季線", "籌碼"})

    def _is_profitability_stability_query(self, query: StructuredQuery) -> bool:
        tags = self._topic_tags(query)
        return query.intent == Intent.FINANCIAL_HEALTH and bool(tags & {"獲利", "穩定性"})

    def schedule_follow_up(self, query: StructuredQuery, validation_result: ValidationResult) -> bool:
        if not self._should_schedule_follow_up(query, validation_result):
            return False

        today = datetime.now(timezone.utc).date()
        scheduled = False
        for ticker in self._iter_tickers(query):
            if not self._mark_follow_up_started(ticker):
                continue

            scheduled = True
            if self._run_follow_up_async:
                Thread(
                    target=self._run_follow_up_bundle,
                    args=(ticker, today),
                    daemon=True,
                ).start()
            else:
                self._run_follow_up_bundle(ticker, today)
        return scheduled

    def _safe_call(self, fn, *args) -> None:
        if not callable(fn):
            return
        try:
            fn(*args)
        except Exception:
            pass

    def _format_exception(self, exception: Exception | None) -> str:
        if exception is None:
            return "Unknown sync failure."
        message = str(exception).strip()
        return message or exception.__class__.__name__

    def _should_schedule_follow_up(
        self,
        query: StructuredQuery,
        validation_result: ValidationResult,
    ) -> bool:
        if not self._low_confidence_warmup_enabled or not query.ticker:
            return False

        if validation_result.confidence_score < self._low_confidence_warmup_threshold:
            return True

        return any(
            warning in validation_result.warnings
            for warning in (
                "No supporting evidence retrieved.",
                "Answer indicates insufficient data.",
                "Preliminary LLM answer returned without grounded local evidence.",
            )
        )

    def _iter_tickers(self, query: StructuredQuery) -> list[str]:
        tickers = [query.ticker] if query.ticker else []
        if query.comparison_ticker and query.comparison_ticker not in tickers:
            tickers.append(query.comparison_ticker)
        return tickers

    def _mark_follow_up_started(self, ticker: str) -> bool:
        now = datetime.now(timezone.utc)
        with self._warmup_lock:
            if ticker in self._active_follow_up_tickers:
                return False

            last_follow_up_at = self._last_follow_up_at.get(ticker)
            if last_follow_up_at is not None and now - last_follow_up_at < self._follow_up_cooldown:
                return False

            self._active_follow_up_tickers.add(ticker)
            return True

    def _run_follow_up_bundle(self, ticker: str, today: date) -> None:
        financial_start = date(today.year - 5, 1, 1)
        price_start = today - timedelta(days=730)
        margin_start = today - timedelta(days=365)
        news_start = today - timedelta(days=365)

        try:
            self._safe_call(getattr(self._gateway, "sync_stock_info", None))
            self._safe_call(getattr(self._gateway, "sync_price_history", None), ticker, price_start, today)
            self._safe_call(getattr(self._gateway, "sync_financial_statements", None), ticker, financial_start, today)
            self._safe_call(getattr(self._gateway, "sync_balance_sheet_items", None), ticker, financial_start, today)
            self._safe_call(getattr(self._gateway, "sync_cash_flow_statements", None), ticker, financial_start, today)
            self._safe_call(getattr(self._gateway, "sync_dividend_policies", None), ticker, financial_start, today)
            self._safe_call(getattr(self._gateway, "sync_monthly_revenue_points", None), ticker)
            self._safe_call(getattr(self._gateway, "sync_pe_valuation_points", None), ticker)
            self._safe_call(getattr(self._gateway, "sync_margin_purchase_short_sale", None), ticker, margin_start, today)
            self._safe_call(getattr(self._gateway, "sync_stock_news", None), ticker, news_start, today)
        finally:
            with self._warmup_lock:
                self._active_follow_up_tickers.discard(ticker)
                self._last_follow_up_at[ticker] = datetime.now(timezone.utc)
