from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import time
from threading import Lock, Thread

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import HydrationResult, StructuredQuery, ValidationResult


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

    def __init__(
        self,
        gateway,
        low_confidence_warmup_enabled: bool = True,
        low_confidence_warmup_threshold: float = 0.80,
        follow_up_cooldown_hours: int = 12,
        run_follow_up_async: bool = True,
    ) -> None:
        self._gateway = gateway
        self._low_confidence_warmup_enabled = low_confidence_warmup_enabled
        self._low_confidence_warmup_threshold = low_confidence_warmup_threshold
        self._follow_up_cooldown = timedelta(hours=max(follow_up_cooldown_hours, 1))
        self._run_follow_up_async = run_follow_up_async
        self._warmup_lock = Lock()
        self._active_follow_up_tickers: set[str] = set()
        self._last_follow_up_at: dict[str, datetime] = {}

    def hydrate(self, query: StructuredQuery) -> HydrationResult:
        result = HydrationResult()
        if not query.ticker:
            return result

        today = datetime.now(timezone.utc).date()
        started_at = time.perf_counter()
        required_facets = set(query.required_facets)
        preferred_facets = set(query.preferred_facets)

        self._safe_call(getattr(self._gateway, "sync_stock_info", None))
        for ticker in self._iter_tickers(query):
            for facet in self._ordered_facets(query):
                facet_result = self._sync_facet(facet, ticker, query, today)
                if facet_result.success:
                    result.synced_facets.add(facet)
                    continue

                result.failed_facets.setdefault(facet, self._format_exception(facet_result.exception))
                if facet in required_facets and facet.value not in result.facet_miss_list:
                    result.facet_miss_list.append(facet.value)
                elif facet in preferred_facets and facet.value not in result.preferred_miss_list:
                    result.preferred_miss_list.append(facet.value)

        result.total_duration_ms = (time.perf_counter() - started_at) * 1000
        return result

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
