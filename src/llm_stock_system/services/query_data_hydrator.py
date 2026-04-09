from datetime import date, datetime, timedelta, timezone
from threading import Lock, Thread

from llm_stock_system.core.models import StructuredQuery, ValidationResult


class QueryDataHydrator:
    """Fetches missing market/fundamental data before retrieval runs."""

    PRICE_RELATED_QUESTION_TYPES = {
        "price_range",
        "price_outlook",
        "technical_indicator_review",
        "season_line_margin_review",
        "dividend_yield_review",
        "ex_dividend_performance",
    }
    FUNDAMENTAL_QUESTION_TYPES = {
        "earnings_summary",
        "eps_dividend_review",
        "margin_turnaround_review",
        "gross_margin_comparison_review",
        "profitability_stability_review",
        "debt_dividend_safety_review",
        "fcf_dividend_sustainability_review",
    }
    NEWS_RELATED_QUESTION_TYPES = {
        "market_summary",
        "announcement_summary",
        "theme_impact_review",
        "shipping_rate_impact_review",
        "electricity_cost_impact_review",
        "macro_yield_sentiment_review",
        "revenue_growth_review",
        "monthly_revenue_yoy_review",
        "listing_revenue_review",
        "guidance_reaction_review",
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

    def hydrate(self, query: StructuredQuery) -> None:
        if not query.ticker:
            return

        today = datetime.now(timezone.utc).date()
        tickers = [query.ticker]
        if query.comparison_ticker and query.comparison_ticker not in tickers:
            tickers.append(query.comparison_ticker)

        self._safe_call(self._gateway.sync_stock_info)
        if self._should_sync_query_news(query):
            self._sync_query_news(query, today)

        for ticker in tickers:
            self._hydrate_ticker(query, ticker, today)

    def _hydrate_ticker(self, query: StructuredQuery, ticker: str, today: date) -> None:
        history_years = max((query.time_range_days + 364) // 365, 2)
        price_start = today - timedelta(days=max(query.time_range_days, 120))
        fundamentals_start = date(today.year - history_years, 1, 1)
        long_term_start = date(today.year - 3, 1, 1)
        dividend_start = max(date(today.year - history_years, 1, 1), today - timedelta(days=730))

        if query.question_type in self.PRICE_RELATED_QUESTION_TYPES:
            self._safe_call(self._gateway.sync_price_history, ticker, price_start, today)

        if query.question_type == "technical_indicator_review":
            self._safe_call(self._gateway.sync_price_history, ticker, today - timedelta(days=180), today)

        if query.question_type == "season_line_margin_review":
            self._safe_call(self._gateway.sync_margin_purchase_short_sale, ticker, today - timedelta(days=180), today)

        if query.question_type == "monthly_revenue_yoy_review":
            self._safe_call(self._gateway.sync_monthly_revenue_points, ticker)

        if query.question_type == "listing_revenue_review":
            self._safe_call(self._gateway.sync_monthly_revenue_points, ticker)

        if query.question_type == "pe_valuation_review":
            self._safe_call(self._gateway.sync_pe_valuation_points, ticker)

        if query.question_type in self.FUNDAMENTAL_QUESTION_TYPES:
            self._safe_call(self._gateway.sync_financial_statements, ticker, fundamentals_start, today)

        if query.question_type in {"debt_dividend_safety_review", "fcf_dividend_sustainability_review"}:
            self._safe_call(self._gateway.sync_balance_sheet_items, ticker, long_term_start, today)
            self._safe_call(self._gateway.sync_cash_flow_statements, ticker, long_term_start, today)

        if query.question_type in {
            "eps_dividend_review",
            "dividend_yield_review",
            "ex_dividend_performance",
            "debt_dividend_safety_review",
            "fcf_dividend_sustainability_review",
            "announcement_summary",
        }:
            self._safe_call(self._gateway.sync_dividend_policies, ticker, dividend_start, today)

        if query.question_type in {"announcement_summary", "market_summary"}:
            self._safe_call(self._gateway.sync_financial_statements, ticker, fundamentals_start, today)
            self._safe_call(self._gateway.sync_dividend_policies, ticker, dividend_start, today)

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
        try:
            fn(*args)
        except Exception:
            pass

    def _should_sync_query_news(self, query: StructuredQuery) -> bool:
        return query.question_type in self.NEWS_RELATED_QUESTION_TYPES or query.question_type in {
            "season_line_margin_review",
            "profitability_stability_review",
        }

    def _sync_query_news(self, query: StructuredQuery, today: date) -> None:
        sync_query_news = getattr(self._gateway, "sync_query_news", None)
        if callable(sync_query_news):
            self._safe_call(sync_query_news, query)
            return

        news_start = today - timedelta(days=max(query.time_range_days, 30))
        if query.question_type == "profitability_stability_review":
            news_start = date(today.year - 3, 1, 1)
        for ticker in self._iter_tickers(query):
            self._safe_call(self._gateway.sync_stock_news, ticker, news_start, today)

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
            self._safe_call(self._gateway.sync_stock_info)
            self._safe_call(self._gateway.sync_price_history, ticker, price_start, today)
            self._safe_call(self._gateway.sync_financial_statements, ticker, financial_start, today)
            self._safe_call(self._gateway.sync_balance_sheet_items, ticker, financial_start, today)
            self._safe_call(self._gateway.sync_cash_flow_statements, ticker, financial_start, today)
            self._safe_call(self._gateway.sync_dividend_policies, ticker, financial_start, today)
            self._safe_call(self._gateway.sync_monthly_revenue_points, ticker)
            self._safe_call(self._gateway.sync_pe_valuation_points, ticker)
            self._safe_call(self._gateway.sync_margin_purchase_short_sale, ticker, margin_start, today)
            self._safe_call(self._gateway.sync_stock_news, ticker, news_start, today)
        finally:
            with self._warmup_lock:
                self._active_follow_up_tickers.discard(ticker)
                self._last_follow_up_at[ticker] = datetime.now(timezone.utc)
