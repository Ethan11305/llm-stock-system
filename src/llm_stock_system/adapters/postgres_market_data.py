from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from datetime import date, datetime, time, timedelta, timezone
import re
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import create_engine, text

from llm_stock_system.adapters.finmind import FinMindClient
from llm_stock_system.adapters.news_pipeline import MultiSourceNewsPipeline
from llm_stock_system.adapters.twse_financial import TwseCompanyFinancialClient
from llm_stock_system.core.enums import Intent, SourceTier, Topic
from llm_stock_system.core.interfaces import DocumentRepository, StockResolver
from llm_stock_system.core.models import (
    DividendPolicy,
    Document,
    FinancialStatementItem,
    MarginPurchaseShortSale,
    MonthlyRevenuePoint,
    NewsArticle,
    PriceBar,
    StructuredQuery,
    ValuationPoint,
)


_SORT_PRIORITY_DEFAULT: dict[str, int] = {}
_SORT_PRIORITY_FUNDAMENTAL: dict[str, int] = {
    "pe_current": 4,
    "pe_history": 4,
    "pe_assessment": 4,
    "financial_statement": 3,
    "financial_statement_breakdown": 3,
    "financial_statement_latest": 3,
    "dividend_policy": 2,
    "dividend_analysis": 2,
    "news_article": 1,
}
_SORT_PRIORITY_PRICE_OUTLOOK: dict[str, int] = {
    "market_data": 3,
    "technical_indicator": 3,
    "news_article": 1,
}


@dataclass(frozen=True)
class RetrievalProfile:
    key: str
    builder_plan: tuple[str, ...]
    sort_priority: Mapping[str, int] = dataclass_field(default_factory=dict)
    news_search_term_seeds: tuple[str, ...] = ()
    search_term_strategy: str = "static"
    include_comparison: bool = False
    append_primary_label: bool = False


RETRIEVAL_PROFILES: dict[str, RetrievalProfile] = {
    "news_shipping": RetrievalProfile(
        key="news_shipping",
        builder_plan=("_build_shipping_rate_impact_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        news_search_term_seeds=(
            "紅海",
            "紅海航線",
            "SCFI",
            "運價",
            "運價指數",
            "貨櫃",
            "航運",
            "塞港",
            "繞道",
            "目標價",
            "分析師",
            "法人",
            "外資",
            "評等",
            "上修",
            "下修",
        ),
        include_comparison=True,
        append_primary_label=True,
    ),
    "news_electricity": RetrievalProfile(
        key="news_electricity",
        builder_plan=("_build_electricity_cost_impact_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        news_search_term_seeds=(
            "工業電價",
            "電價",
            "調漲",
            "漲價",
            "電費",
            "用電大戶",
            "成本",
            "節能",
            "轉嫁",
            "因應對策",
        ),
        include_comparison=True,
        append_primary_label=True,
    ),
    "news_macro": RetrievalProfile(
        key="news_macro",
        builder_plan=("_build_macro_yield_sentiment_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        news_search_term_seeds=(
            "CPI",
            "通膨",
            "高殖利率",
            "殖利率",
            "金控",
            "金控股",
            "法人",
            "外資",
            "觀點",
            "防禦",
            "美債",
            "利率",
        ),
        append_primary_label=True,
    ),
    "news_guidance": RetrievalProfile(
        key="news_guidance",
        builder_plan=("_build_guidance_reaction_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        news_search_term_seeds=(
            "法說",
            "法說會",
            "營運指引",
            "下半年",
            "展望",
            "財測",
            "法人",
            "外資",
            "目標價",
            "評等",
            "正面",
            "負面",
        ),
    ),
    "news_listing": RetrievalProfile(
        key="news_listing",
        builder_plan=("_build_listing_revenue_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        news_search_term_seeds=(
            "上市",
            "掛牌",
            "IPO",
            "蜜月行情",
            "承銷",
            "股價波動",
            "營收",
            "月增",
            "年增",
        ),
    ),
    "news_theme": RetrievalProfile(
        key="news_theme",
        builder_plan=("_build_theme_impact_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        search_term_strategy="theme",
        include_comparison=True,
    ),
    "news_generic": RetrievalProfile(
        key="news_generic",
        builder_plan=("_build_market_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        search_term_strategy="generic_news",
    ),
    "earnings_monthly_revenue": RetrievalProfile(
        key="earnings_monthly_revenue",
        builder_plan=("_build_monthly_revenue_yoy_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        news_search_term_seeds=("營收", "月增", "年增", "成長", "展望"),
    ),
    "earnings_margin_turnaround": RetrievalProfile(
        key="earnings_margin_turnaround",
        builder_plan=("_build_margin_turnaround_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "earnings_eps_dividend": RetrievalProfile(
        key="earnings_eps_dividend",
        builder_plan=("_build_fundamental_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "earnings_fundamental": RetrievalProfile(
        key="earnings_fundamental",
        builder_plan=("_build_fundamental_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "valuation_price_range": RetrievalProfile(
        key="valuation_price_range",
        builder_plan=("_build_price_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "valuation_price_outlook": RetrievalProfile(
        key="valuation_price_outlook",
        builder_plan=("_build_price_documents", "_build_market_documents"),
        sort_priority=_SORT_PRIORITY_PRICE_OUTLOOK,
    ),
    "valuation_fundamental": RetrievalProfile(
        key="valuation_fundamental",
        builder_plan=(
            "_build_fundamental_documents",
            "_build_pe_valuation_documents",
            "_build_market_documents",
        ),
        sort_priority=_SORT_PRIORITY_FUNDAMENTAL,
    ),
    "valuation_pe_only": RetrievalProfile(
        key="valuation_pe_only",
        builder_plan=("_build_pe_valuation_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "dividend_ex": RetrievalProfile(
        key="dividend_ex",
        builder_plan=("_build_ex_dividend_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "dividend_fcf": RetrievalProfile(
        key="dividend_fcf",
        builder_plan=("_build_fcf_dividend_sustainability_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "dividend_debt": RetrievalProfile(
        key="dividend_debt",
        builder_plan=("_build_debt_dividend_safety_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "dividend_yield": RetrievalProfile(
        key="dividend_yield",
        builder_plan=("_build_dividend_yield_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "health_gross_margin_cmp": RetrievalProfile(
        key="health_gross_margin_cmp",
        builder_plan=("_build_gross_margin_comparison_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "health_profitability": RetrievalProfile(
        key="health_profitability",
        builder_plan=("_build_profitability_stability_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "health_revenue_growth": RetrievalProfile(
        key="health_revenue_growth",
        builder_plan=("_build_revenue_growth_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        news_search_term_seeds=("營收", "月增", "年增", "成長", "展望", "AI", "AI伺服器", "占比"),
    ),
    "technical_margin_flow": RetrievalProfile(
        key="technical_margin_flow",
        builder_plan=("_build_season_line_margin_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "technical_indicators": RetrievalProfile(
        key="technical_indicators",
        builder_plan=("_build_technical_indicator_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
    "investment_announcement": RetrievalProfile(
        key="investment_announcement",
        builder_plan=("_build_announcement_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
        search_term_strategy="generic_news",
    ),
    "investment_support": RetrievalProfile(
        key="investment_support",
        builder_plan=(
            "_build_fundamental_documents",
            "_build_pe_valuation_documents",
            "_build_market_documents",
        ),
        sort_priority=_SORT_PRIORITY_FUNDAMENTAL,
    ),
    "investment_risk": RetrievalProfile(
        key="investment_risk",
        builder_plan=("_build_market_documents",),
        sort_priority=_SORT_PRIORITY_DEFAULT,
    ),
}


class FinMindPostgresGateway:
    def __init__(
        self,
        database_url: str,
        finmind_client: FinMindClient,
        twse_financial_client: TwseCompanyFinancialClient | None = None,
        news_pipeline: MultiSourceNewsPipeline | None = None,
        sync_on_query: bool = True,
        stock_info_refresh_hours: int = 24,
    ) -> None:
        self._engine = create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
        self._finmind_client = finmind_client
        self._twse_financial_client = twse_financial_client
        self._news_pipeline = news_pipeline
        self._sync_on_query = sync_on_query
        self._stock_info_refresh_hours = stock_info_refresh_hours
        self.ensure_schema()

    def ensure_schema(self) -> None:
        statements = [
            "CREATE TABLE IF NOT EXISTS stock_info (stock_id VARCHAR(64) PRIMARY KEY, stock_name TEXT NOT NULL, industry_category TEXT, market_type TEXT, reference_date DATE, synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
            "CREATE INDEX IF NOT EXISTS idx_stock_info_stock_name ON stock_info (stock_name)",
            "CREATE TABLE IF NOT EXISTS daily_price_bars (ticker VARCHAR(64) NOT NULL, trading_date DATE NOT NULL, open_price NUMERIC(12,4) NOT NULL, high_price NUMERIC(12,4) NOT NULL, low_price NUMERIC(12,4) NOT NULL, close_price NUMERIC(12,4) NOT NULL, trading_volume BIGINT, trading_money BIGINT, spread NUMERIC(12,4), turnover BIGINT, source_name TEXT NOT NULL DEFAULT 'FinMind TaiwanStockPrice', synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, trading_date))",
            "CREATE INDEX IF NOT EXISTS idx_daily_price_bars_ticker_date ON daily_price_bars (ticker, trading_date DESC)",
            "CREATE TABLE IF NOT EXISTS financial_statement_items (ticker VARCHAR(64) NOT NULL, statement_date DATE NOT NULL, item_type TEXT NOT NULL, value NUMERIC(20,6) NOT NULL, origin_name TEXT NOT NULL, synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, statement_date, item_type))",
            "CREATE INDEX IF NOT EXISTS idx_financial_statement_items_lookup ON financial_statement_items (ticker, statement_date DESC, item_type)",
            "CREATE TABLE IF NOT EXISTS balance_sheet_items (ticker VARCHAR(64) NOT NULL, statement_date DATE NOT NULL, item_type TEXT NOT NULL, value NUMERIC(20,6) NOT NULL, origin_name TEXT NOT NULL, synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, statement_date, item_type))",
            "CREATE INDEX IF NOT EXISTS idx_balance_sheet_items_lookup ON balance_sheet_items (ticker, statement_date DESC, item_type)",
            "CREATE TABLE IF NOT EXISTS cash_flow_statement_items (ticker VARCHAR(64) NOT NULL, statement_date DATE NOT NULL, item_type TEXT NOT NULL, value NUMERIC(20,6) NOT NULL, origin_name TEXT NOT NULL, synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, statement_date, item_type))",
            "CREATE INDEX IF NOT EXISTS idx_cash_flow_statement_items_lookup ON cash_flow_statement_items (ticker, statement_date DESC, item_type)",
            "CREATE TABLE IF NOT EXISTS monthly_revenue_points (ticker VARCHAR(64) NOT NULL, revenue_month DATE NOT NULL, revenue NUMERIC(20,6) NOT NULL, prior_year_month_revenue NUMERIC(20,6), month_over_month_pct NUMERIC(20,6), year_over_year_pct NUMERIC(20,6), cumulative_revenue NUMERIC(20,6), prior_year_cumulative_revenue NUMERIC(20,6), cumulative_yoy_pct NUMERIC(20,6), report_date DATE, notes TEXT, synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, revenue_month))",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS prior_year_month_revenue NUMERIC(20,6)",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS month_over_month_pct NUMERIC(20,6)",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS year_over_year_pct NUMERIC(20,6)",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS cumulative_revenue NUMERIC(20,6)",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS prior_year_cumulative_revenue NUMERIC(20,6)",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS cumulative_yoy_pct NUMERIC(20,6)",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS report_date DATE",
            "ALTER TABLE monthly_revenue_points ADD COLUMN IF NOT EXISTS notes TEXT",
            "CREATE INDEX IF NOT EXISTS idx_monthly_revenue_points_lookup ON monthly_revenue_points (ticker, revenue_month DESC)",
            "CREATE TABLE IF NOT EXISTS pe_valuation_points (ticker VARCHAR(64) NOT NULL, valuation_month DATE NOT NULL, pe_ratio NUMERIC(20,6), peer_pe_ratio NUMERIC(20,6), pb_ratio NUMERIC(20,6), peer_pb_ratio NUMERIC(20,6), synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, valuation_month))",
            "CREATE INDEX IF NOT EXISTS idx_pe_valuation_points_lookup ON pe_valuation_points (ticker, valuation_month DESC)",
            "CREATE TABLE IF NOT EXISTS dividend_policies (ticker VARCHAR(64) NOT NULL, base_date DATE NOT NULL, year_label TEXT NOT NULL, cash_earnings_distribution NUMERIC(20,6), cash_statutory_surplus NUMERIC(20,6), stock_earnings_distribution NUMERIC(20,6), stock_statutory_surplus NUMERIC(20,6), participate_distribution_of_total_shares NUMERIC(20,6), announcement_date DATE, announcement_time TEXT, cash_ex_dividend_trading_date DATE, cash_dividend_payment_date DATE, synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, base_date, year_label))",
            "ALTER TABLE dividend_policies ADD COLUMN IF NOT EXISTS participate_distribution_of_total_shares NUMERIC(20,6)",
            "CREATE INDEX IF NOT EXISTS idx_dividend_policies_lookup ON dividend_policies (ticker, base_date DESC)",
            "CREATE TABLE IF NOT EXISTS stock_news_articles (ticker VARCHAR(64) NOT NULL, published_at TIMESTAMP NOT NULL, title TEXT NOT NULL, summary TEXT, source_name TEXT NOT NULL, url TEXT NOT NULL, source_tier TEXT NOT NULL DEFAULT 'medium', source_type TEXT NOT NULL DEFAULT 'news_article', provider_name TEXT NOT NULL DEFAULT 'finmind', tags TEXT, synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, url))",
            "ALTER TABLE stock_news_articles ADD COLUMN IF NOT EXISTS source_tier TEXT NOT NULL DEFAULT 'medium'",
            "ALTER TABLE stock_news_articles ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'news_article'",
            "ALTER TABLE stock_news_articles ADD COLUMN IF NOT EXISTS provider_name TEXT NOT NULL DEFAULT 'finmind'",
            "ALTER TABLE stock_news_articles ADD COLUMN IF NOT EXISTS tags TEXT",
            "CREATE INDEX IF NOT EXISTS idx_stock_news_articles_lookup ON stock_news_articles (ticker, published_at DESC)",
            "CREATE TABLE IF NOT EXISTS margin_purchase_short_sale_bars (ticker VARCHAR(64) NOT NULL, trading_date DATE NOT NULL, margin_purchase_buy BIGINT, margin_purchase_cash_repayment BIGINT, margin_purchase_limit BIGINT, margin_purchase_sell BIGINT, margin_purchase_today_balance BIGINT, margin_purchase_yesterday_balance BIGINT, offset_loan_and_short BIGINT, short_sale_buy BIGINT, short_sale_cash_repayment BIGINT, short_sale_limit BIGINT, short_sale_sell BIGINT, short_sale_today_balance BIGINT, short_sale_yesterday_balance BIGINT, note TEXT, source_name TEXT NOT NULL DEFAULT 'FinMind TaiwanStockMarginPurchaseShortSale', synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (ticker, trading_date))",
            "CREATE INDEX IF NOT EXISTS idx_margin_purchase_short_sale_lookup ON margin_purchase_short_sale_bars (ticker, trading_date DESC)",
        ]
        with self._engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

    def ping(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def sync_stock_info(self, force: bool = False) -> int:
        if not force and self._stock_info_is_fresh():
            return 0
        items = self._finmind_client.fetch_stock_info()
        if not items:
            return 0
        payload = [
            {
                "stock_id": item.stock_id,
                "stock_name": item.stock_name,
                "industry_category": item.industry_category,
                "market_type": item.market_type,
                "reference_date": item.reference_date.date() if item.reference_date else None,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO stock_info (stock_id, stock_name, industry_category, market_type, reference_date, synced_at) "
            "VALUES (:stock_id, :stock_name, :industry_category, :market_type, :reference_date, CURRENT_TIMESTAMP) "
            "ON CONFLICT (stock_id) DO UPDATE SET "
            "stock_name = EXCLUDED.stock_name, industry_category = EXCLUDED.industry_category, market_type = EXCLUDED.market_type, "
            "reference_date = EXCLUDED.reference_date, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def resolve_company(self, query_text: str) -> tuple[str | None, str | None]:
        if not query_text.strip():
            return None, None
        normalized_query = self._normalize_lookup_text(query_text)
        if self._sync_on_query:
            self.sync_stock_info()
        with self._engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT stock_id, stock_name FROM stock_info "
                    "WHERE POSITION(stock_name IN :query_text) > 0 "
                    "OR POSITION(REPLACE(stock_name, ' ', '') IN :normalized_query) > 0 "
                    "ORDER BY CHAR_LENGTH(REPLACE(stock_name, ' ', '')) DESC LIMIT 1"
                ),
                {"query_text": query_text, "normalized_query": normalized_query},
            ).mappings().first()
        if row is None:
            return None, None
        return str(row["stock_id"]), str(row["stock_name"])

    def sync_price_history(self, ticker: str, start_date: date, end_date: date) -> int:
        items = self._finmind_client.fetch_stock_price(ticker, start_date, end_date)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "trading_date": item.trading_date.date(),
                "open_price": item.open_price,
                "high_price": item.high_price,
                "low_price": item.low_price,
                "close_price": item.close_price,
                "trading_volume": item.trading_volume,
                "trading_money": item.trading_money,
                "spread": item.spread,
                "turnover": item.turnover,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO daily_price_bars (ticker, trading_date, open_price, high_price, low_price, close_price, trading_volume, trading_money, spread, turnover, synced_at) "
            "VALUES (:ticker, :trading_date, :open_price, :high_price, :low_price, :close_price, :trading_volume, :trading_money, :spread, :turnover, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, trading_date) DO UPDATE SET "
            "open_price = EXCLUDED.open_price, high_price = EXCLUDED.high_price, low_price = EXCLUDED.low_price, close_price = EXCLUDED.close_price, "
            "trading_volume = EXCLUDED.trading_volume, trading_money = EXCLUDED.trading_money, spread = EXCLUDED.spread, turnover = EXCLUDED.turnover, "
            "synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_financial_statements(self, ticker: str, start_date: date, end_date: date) -> int:
        items = self._finmind_client.fetch_financial_statements(ticker, start_date, end_date)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "statement_date": item.statement_date.date(),
                "item_type": item.item_type,
                "value": item.value,
                "origin_name": item.origin_name,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO financial_statement_items (ticker, statement_date, item_type, value, origin_name, synced_at) "
            "VALUES (:ticker, :statement_date, :item_type, :value, :origin_name, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, statement_date, item_type) DO UPDATE SET value = EXCLUDED.value, origin_name = EXCLUDED.origin_name, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_balance_sheet_items(self, ticker: str, start_date: date, end_date: date) -> int:
        items = self._finmind_client.fetch_balance_sheet_items(ticker, start_date, end_date)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "statement_date": item.statement_date.date(),
                "item_type": item.item_type,
                "value": item.value,
                "origin_name": item.origin_name,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO balance_sheet_items (ticker, statement_date, item_type, value, origin_name, synced_at) "
            "VALUES (:ticker, :statement_date, :item_type, :value, :origin_name, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, statement_date, item_type) DO UPDATE SET value = EXCLUDED.value, origin_name = EXCLUDED.origin_name, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_cash_flow_statements(self, ticker: str, start_date: date, end_date: date) -> int:
        items = self._finmind_client.fetch_cash_flow_statements(ticker, start_date, end_date)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "statement_date": item.statement_date.date(),
                "item_type": item.item_type,
                "value": item.value,
                "origin_name": item.origin_name,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO cash_flow_statement_items (ticker, statement_date, item_type, value, origin_name, synced_at) "
            "VALUES (:ticker, :statement_date, :item_type, :value, :origin_name, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, statement_date, item_type) DO UPDATE SET value = EXCLUDED.value, origin_name = EXCLUDED.origin_name, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_monthly_revenue_points(self, ticker: str) -> int:
        if self._twse_financial_client is None:
            return 0
        items = self._twse_financial_client.fetch_monthly_revenue(ticker)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "revenue_month": item.revenue_month.date(),
                "revenue": item.revenue,
                "prior_year_month_revenue": item.prior_year_month_revenue,
                "month_over_month_pct": item.month_over_month_pct,
                "year_over_year_pct": item.year_over_year_pct,
                "cumulative_revenue": item.cumulative_revenue,
                "prior_year_cumulative_revenue": item.prior_year_cumulative_revenue,
                "cumulative_yoy_pct": item.cumulative_yoy_pct,
                "report_date": item.report_date.date() if item.report_date else None,
                "notes": item.notes,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO monthly_revenue_points (ticker, revenue_month, revenue, prior_year_month_revenue, month_over_month_pct, year_over_year_pct, cumulative_revenue, prior_year_cumulative_revenue, cumulative_yoy_pct, report_date, notes, synced_at) "
            "VALUES (:ticker, :revenue_month, :revenue, :prior_year_month_revenue, :month_over_month_pct, :year_over_year_pct, :cumulative_revenue, :prior_year_cumulative_revenue, :cumulative_yoy_pct, :report_date, :notes, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, revenue_month) DO UPDATE SET "
            "revenue = EXCLUDED.revenue, prior_year_month_revenue = EXCLUDED.prior_year_month_revenue, "
            "month_over_month_pct = EXCLUDED.month_over_month_pct, year_over_year_pct = EXCLUDED.year_over_year_pct, "
            "cumulative_revenue = EXCLUDED.cumulative_revenue, prior_year_cumulative_revenue = EXCLUDED.prior_year_cumulative_revenue, "
            "cumulative_yoy_pct = EXCLUDED.cumulative_yoy_pct, report_date = EXCLUDED.report_date, notes = EXCLUDED.notes, "
            "synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_pe_valuation_points(self, ticker: str) -> int:
        if self._twse_financial_client is None:
            return 0
        items = self._twse_financial_client.fetch_valuation_points(ticker)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "valuation_month": item.valuation_month.date(),
                "pe_ratio": item.pe_ratio,
                "peer_pe_ratio": item.peer_pe_ratio,
                "pb_ratio": item.pb_ratio,
                "peer_pb_ratio": item.peer_pb_ratio,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO pe_valuation_points (ticker, valuation_month, pe_ratio, peer_pe_ratio, pb_ratio, peer_pb_ratio, synced_at) "
            "VALUES (:ticker, :valuation_month, :pe_ratio, :peer_pe_ratio, :pb_ratio, :peer_pb_ratio, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, valuation_month) DO UPDATE SET "
            "pe_ratio = EXCLUDED.pe_ratio, peer_pe_ratio = EXCLUDED.peer_pe_ratio, "
            "pb_ratio = EXCLUDED.pb_ratio, peer_pb_ratio = EXCLUDED.peer_pb_ratio, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_dividend_policies(self, ticker: str, start_date: date, end_date: date) -> int:
        items = self._finmind_client.fetch_dividend_policies(ticker, start_date, end_date)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "base_date": item.date.date(),
                "year_label": item.year_label,
                "cash_earnings_distribution": item.cash_earnings_distribution,
                "cash_statutory_surplus": item.cash_statutory_surplus,
                "stock_earnings_distribution": item.stock_earnings_distribution,
                "stock_statutory_surplus": item.stock_statutory_surplus,
                "participate_distribution_of_total_shares": item.participate_distribution_of_total_shares,
                "announcement_date": item.announcement_date.date() if item.announcement_date else None,
                "announcement_time": item.announcement_time,
                "cash_ex_dividend_trading_date": item.cash_ex_dividend_trading_date.date() if item.cash_ex_dividend_trading_date else None,
                "cash_dividend_payment_date": item.cash_dividend_payment_date.date() if item.cash_dividend_payment_date else None,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO dividend_policies (ticker, base_date, year_label, cash_earnings_distribution, cash_statutory_surplus, stock_earnings_distribution, stock_statutory_surplus, participate_distribution_of_total_shares, announcement_date, announcement_time, cash_ex_dividend_trading_date, cash_dividend_payment_date, synced_at) "
            "VALUES (:ticker, :base_date, :year_label, :cash_earnings_distribution, :cash_statutory_surplus, :stock_earnings_distribution, :stock_statutory_surplus, :participate_distribution_of_total_shares, :announcement_date, :announcement_time, :cash_ex_dividend_trading_date, :cash_dividend_payment_date, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, base_date, year_label) DO UPDATE SET "
            "cash_earnings_distribution = EXCLUDED.cash_earnings_distribution, cash_statutory_surplus = EXCLUDED.cash_statutory_surplus, "
            "stock_earnings_distribution = EXCLUDED.stock_earnings_distribution, stock_statutory_surplus = EXCLUDED.stock_statutory_surplus, "
            "participate_distribution_of_total_shares = EXCLUDED.participate_distribution_of_total_shares, "
            "announcement_date = EXCLUDED.announcement_date, announcement_time = EXCLUDED.announcement_time, "
            "cash_ex_dividend_trading_date = EXCLUDED.cash_ex_dividend_trading_date, cash_dividend_payment_date = EXCLUDED.cash_dividend_payment_date, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_stock_news(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        search_terms: tuple[str, ...] | None = None,
    ) -> int:
        company_name = self._lookup_stock_name(ticker)
        items = self._fetch_news_articles(
            ticker=ticker,
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            search_terms=search_terms or (),
        )
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "published_at": item.published_at,
                "title": item.title,
                "summary": item.summary,
                "source_name": item.source_name,
                "url": item.url,
                "source_tier": self._resolve_news_source_tier(item).value,
                "source_type": item.source_type,
                "provider_name": item.provider_name or "unknown",
                "tags": self._serialize_news_tags(item.tags),
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO stock_news_articles (ticker, published_at, title, summary, source_name, url, source_tier, source_type, provider_name, tags, synced_at) "
            "VALUES (:ticker, :published_at, :title, :summary, :source_name, :url, :source_tier, :source_type, :provider_name, :tags, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, url) DO UPDATE SET published_at = EXCLUDED.published_at, title = EXCLUDED.title, summary = EXCLUDED.summary, "
            "source_name = EXCLUDED.source_name, source_tier = EXCLUDED.source_tier, source_type = EXCLUDED.source_type, "
            "provider_name = EXCLUDED.provider_name, tags = EXCLUDED.tags, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def sync_query_news(self, query: StructuredQuery) -> int:
        if not query.ticker:
            return 0

        profile = self._resolve_retrieval_profile(query)
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=max(query.time_range_days, 30))
        sync_plan = [(query.ticker, query.company_name)]
        if (
            profile.include_comparison
            and query.comparison_ticker
            and query.comparison_ticker not in {query.ticker}
        ):
            sync_plan.append((query.comparison_ticker, query.comparison_company_name))

        total_rows = 0
        for ticker, label in sync_plan:
            search_terms = self._build_news_search_terms(query, label)
            total_rows += self.sync_stock_news(
                ticker=ticker,
                start_date=start_date,
                end_date=today,
                search_terms=search_terms,
            )
        return total_rows

    def sync_margin_purchase_short_sale(self, ticker: str, start_date: date, end_date: date) -> int:
        items = self._finmind_client.fetch_margin_purchase_short_sale(ticker, start_date, end_date)
        if not items:
            return 0
        payload = [
            {
                "ticker": item.ticker,
                "trading_date": item.trading_date.date(),
                "margin_purchase_buy": item.margin_purchase_buy,
                "margin_purchase_cash_repayment": item.margin_purchase_cash_repayment,
                "margin_purchase_limit": item.margin_purchase_limit,
                "margin_purchase_sell": item.margin_purchase_sell,
                "margin_purchase_today_balance": item.margin_purchase_today_balance,
                "margin_purchase_yesterday_balance": item.margin_purchase_yesterday_balance,
                "offset_loan_and_short": item.offset_loan_and_short,
                "short_sale_buy": item.short_sale_buy,
                "short_sale_cash_repayment": item.short_sale_cash_repayment,
                "short_sale_limit": item.short_sale_limit,
                "short_sale_sell": item.short_sale_sell,
                "short_sale_today_balance": item.short_sale_today_balance,
                "short_sale_yesterday_balance": item.short_sale_yesterday_balance,
                "note": item.note,
            }
            for item in items
        ]
        sql = text(
            "INSERT INTO margin_purchase_short_sale_bars (ticker, trading_date, margin_purchase_buy, margin_purchase_cash_repayment, margin_purchase_limit, margin_purchase_sell, margin_purchase_today_balance, margin_purchase_yesterday_balance, offset_loan_and_short, short_sale_buy, short_sale_cash_repayment, short_sale_limit, short_sale_sell, short_sale_today_balance, short_sale_yesterday_balance, note, synced_at) "
            "VALUES (:ticker, :trading_date, :margin_purchase_buy, :margin_purchase_cash_repayment, :margin_purchase_limit, :margin_purchase_sell, :margin_purchase_today_balance, :margin_purchase_yesterday_balance, :offset_loan_and_short, :short_sale_buy, :short_sale_cash_repayment, :short_sale_limit, :short_sale_sell, :short_sale_today_balance, :short_sale_yesterday_balance, :note, CURRENT_TIMESTAMP) "
            "ON CONFLICT (ticker, trading_date) DO UPDATE SET "
            "margin_purchase_buy = EXCLUDED.margin_purchase_buy, margin_purchase_cash_repayment = EXCLUDED.margin_purchase_cash_repayment, "
            "margin_purchase_limit = EXCLUDED.margin_purchase_limit, margin_purchase_sell = EXCLUDED.margin_purchase_sell, "
            "margin_purchase_today_balance = EXCLUDED.margin_purchase_today_balance, margin_purchase_yesterday_balance = EXCLUDED.margin_purchase_yesterday_balance, "
            "offset_loan_and_short = EXCLUDED.offset_loan_and_short, short_sale_buy = EXCLUDED.short_sale_buy, "
            "short_sale_cash_repayment = EXCLUDED.short_sale_cash_repayment, short_sale_limit = EXCLUDED.short_sale_limit, "
            "short_sale_sell = EXCLUDED.short_sale_sell, short_sale_today_balance = EXCLUDED.short_sale_today_balance, "
            "short_sale_yesterday_balance = EXCLUDED.short_sale_yesterday_balance, note = EXCLUDED.note, synced_at = CURRENT_TIMESTAMP"
        )
        with self._engine.begin() as connection:
            connection.execute(sql, payload)
        return len(payload)

    def get_price_bars(self, ticker: str, start_date: date, end_date: date) -> list[PriceBar]:
        if self._sync_on_query and self._price_data_needs_refresh(ticker, end_date):
            try:
                self.sync_price_history(ticker, start_date, end_date)
            except Exception:
                pass
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT ticker, trading_date, open_price, high_price, low_price, close_price, trading_volume, trading_money, spread, turnover "
                    "FROM daily_price_bars WHERE ticker = :ticker AND trading_date BETWEEN :start_date AND :end_date ORDER BY trading_date DESC"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()
        return [
            PriceBar(
                ticker=str(row["ticker"]),
                trading_date=self._to_utc_datetime(row["trading_date"]),
                open_price=float(row["open_price"]),
                high_price=float(row["high_price"]),
                low_price=float(row["low_price"]),
                close_price=float(row["close_price"]),
                trading_volume=int(row["trading_volume"]) if row["trading_volume"] is not None else None,
                trading_money=int(row["trading_money"]) if row["trading_money"] is not None else None,
                spread=float(row["spread"]) if row["spread"] is not None else None,
                turnover=int(row["turnover"]) if row["turnover"] is not None else None,
            )
            for row in rows
        ]

    def get_financial_statement_items(self, ticker: str, start_date: date, end_date: date) -> list[FinancialStatementItem]:
        rows = self._query_financial_rows(ticker, start_date, end_date)
        if not rows and self._sync_on_query:
            try:
                self.sync_financial_statements(ticker, start_date, end_date)
            except Exception:
                pass
            rows = self._query_financial_rows(ticker, start_date, end_date)
        return [
            FinancialStatementItem(
                ticker=str(row["ticker"]),
                statement_date=self._to_utc_datetime(row["statement_date"]),
                item_type=str(row["item_type"]),
                value=float(row["value"]),
                origin_name=str(row["origin_name"]),
            )
            for row in rows
        ]

    def get_balance_sheet_items(self, ticker: str, start_date: date, end_date: date) -> list[FinancialStatementItem]:
        rows = self._query_balance_sheet_rows(ticker, start_date, end_date)
        if not rows and self._sync_on_query:
            try:
                self.sync_balance_sheet_items(ticker, start_date, end_date)
            except Exception:
                pass
            rows = self._query_balance_sheet_rows(ticker, start_date, end_date)
        return [
            FinancialStatementItem(
                ticker=str(row["ticker"]),
                statement_date=self._to_utc_datetime(row["statement_date"]),
                item_type=str(row["item_type"]),
                value=float(row["value"]),
                origin_name=str(row["origin_name"]),
            )
            for row in rows
        ]

    def get_cash_flow_statement_items(self, ticker: str, start_date: date, end_date: date) -> list[FinancialStatementItem]:
        rows = self._query_cash_flow_rows(ticker, start_date, end_date)
        if not rows and self._sync_on_query:
            try:
                self.sync_cash_flow_statements(ticker, start_date, end_date)
            except Exception:
                pass
            rows = self._query_cash_flow_rows(ticker, start_date, end_date)
        return [
            FinancialStatementItem(
                ticker=str(row["ticker"]),
                statement_date=self._to_utc_datetime(row["statement_date"]),
                item_type=str(row["item_type"]),
                value=float(row["value"]),
                origin_name=str(row["origin_name"]),
            )
            for row in rows
        ]

    def get_monthly_revenue_points(self, ticker: str, start_date: date, end_date: date) -> list[MonthlyRevenuePoint]:
        if self._sync_on_query and self._monthly_revenue_needs_refresh(ticker):
            try:
                self.sync_monthly_revenue_points(ticker)
            except Exception:
                pass
        rows = self._query_monthly_revenue_rows(ticker, start_date, end_date)
        return [
            MonthlyRevenuePoint(
                ticker=str(row["ticker"]),
                revenue_month=self._to_utc_datetime(row["revenue_month"]),
                revenue=float(row["revenue"]),
                prior_year_month_revenue=float(row["prior_year_month_revenue"])
                if row["prior_year_month_revenue"] is not None
                else None,
                month_over_month_pct=float(row["month_over_month_pct"])
                if row["month_over_month_pct"] is not None
                else None,
                year_over_year_pct=float(row["year_over_year_pct"])
                if row["year_over_year_pct"] is not None
                else None,
                cumulative_revenue=float(row["cumulative_revenue"])
                if row["cumulative_revenue"] is not None
                else None,
                prior_year_cumulative_revenue=float(row["prior_year_cumulative_revenue"])
                if row["prior_year_cumulative_revenue"] is not None
                else None,
                cumulative_yoy_pct=float(row["cumulative_yoy_pct"])
                if row["cumulative_yoy_pct"] is not None
                else None,
                report_date=self._to_utc_datetime(row["report_date"]) if row["report_date"] is not None else None,
                notes=row["notes"],
            )
            for row in rows
        ]

    def get_pe_valuation_points(self, ticker: str, start_date: date, end_date: date) -> list[ValuationPoint]:
        if self._sync_on_query and self._pe_valuation_needs_refresh(ticker):
            try:
                self.sync_pe_valuation_points(ticker)
            except Exception:
                pass
        rows = self._query_pe_valuation_rows(ticker, start_date, end_date)
        return [
            ValuationPoint(
                ticker=str(row["ticker"]),
                valuation_month=self._to_utc_datetime(row["valuation_month"]),
                pe_ratio=float(row["pe_ratio"]) if row["pe_ratio"] is not None else None,
                peer_pe_ratio=float(row["peer_pe_ratio"]) if row["peer_pe_ratio"] is not None else None,
                pb_ratio=float(row["pb_ratio"]) if row["pb_ratio"] is not None else None,
                peer_pb_ratio=float(row["peer_pb_ratio"]) if row["peer_pb_ratio"] is not None else None,
            )
            for row in rows
        ]

    def get_dividend_policies(self, ticker: str, start_date: date, end_date: date) -> list[DividendPolicy]:
        rows = self._query_dividend_rows(ticker, start_date, end_date)
        if not rows and self._sync_on_query:
            try:
                self.sync_dividend_policies(ticker, start_date, end_date)
            except Exception:
                pass
            rows = self._query_dividend_rows(ticker, start_date, end_date)
        return [
            DividendPolicy(
                ticker=str(row["ticker"]),
                date=self._to_utc_datetime(row["base_date"]),
                year_label=str(row["year_label"]),
                cash_earnings_distribution=float(row["cash_earnings_distribution"]) if row["cash_earnings_distribution"] is not None else None,
                cash_statutory_surplus=float(row["cash_statutory_surplus"]) if row["cash_statutory_surplus"] is not None else None,
                stock_earnings_distribution=float(row["stock_earnings_distribution"]) if row["stock_earnings_distribution"] is not None else None,
                stock_statutory_surplus=float(row["stock_statutory_surplus"]) if row["stock_statutory_surplus"] is not None else None,
                participate_distribution_of_total_shares=float(row["participate_distribution_of_total_shares"])
                if row["participate_distribution_of_total_shares"] is not None
                else None,
                announcement_date=self._to_utc_datetime(row["announcement_date"]) if row["announcement_date"] is not None else None,
                announcement_time=row["announcement_time"],
                cash_ex_dividend_trading_date=self._to_utc_datetime(row["cash_ex_dividend_trading_date"]) if row["cash_ex_dividend_trading_date"] is not None else None,
                cash_dividend_payment_date=self._to_utc_datetime(row["cash_dividend_payment_date"]) if row["cash_dividend_payment_date"] is not None else None,
            )
            for row in rows
        ]

    def get_stock_news(self, ticker: str, start_date: date, end_date: date) -> list[NewsArticle]:
        rows = self._query_news_rows(ticker, start_date, end_date)
        if not rows and self._sync_on_query:
            try:
                self.sync_stock_news(ticker, start_date, end_date)
            except Exception:
                pass
            rows = self._query_news_rows(ticker, start_date, end_date)
        return [
            NewsArticle(
                ticker=str(row["ticker"]),
                published_at=self._coerce_datetime(row["published_at"]),
                title=str(row["title"]),
                summary=row["summary"],
                source_name=str(row["source_name"]),
                url=str(row["url"]),
                source_tier=self._coerce_source_tier(row.get("source_tier")),
                source_type=str(row.get("source_type") or "news_article"),
                provider_name=str(row.get("provider_name") or "unknown"),
                tags=self._deserialize_news_tags(row.get("tags")),
            )
            for row in rows
        ]

    def get_margin_purchase_short_sale(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[MarginPurchaseShortSale]:
        rows = self._query_margin_rows(ticker, start_date, end_date)
        if not rows and self._sync_on_query:
            try:
                self.sync_margin_purchase_short_sale(ticker, start_date, end_date)
            except Exception:
                pass
            rows = self._query_margin_rows(ticker, start_date, end_date)
        return [
            MarginPurchaseShortSale(
                ticker=str(row["ticker"]),
                trading_date=self._to_utc_datetime(row["trading_date"]),
                margin_purchase_buy=int(row["margin_purchase_buy"]) if row["margin_purchase_buy"] is not None else None,
                margin_purchase_cash_repayment=int(row["margin_purchase_cash_repayment"]) if row["margin_purchase_cash_repayment"] is not None else None,
                margin_purchase_limit=int(row["margin_purchase_limit"]) if row["margin_purchase_limit"] is not None else None,
                margin_purchase_sell=int(row["margin_purchase_sell"]) if row["margin_purchase_sell"] is not None else None,
                margin_purchase_today_balance=int(row["margin_purchase_today_balance"]) if row["margin_purchase_today_balance"] is not None else None,
                margin_purchase_yesterday_balance=int(row["margin_purchase_yesterday_balance"]) if row["margin_purchase_yesterday_balance"] is not None else None,
                offset_loan_and_short=int(row["offset_loan_and_short"]) if row["offset_loan_and_short"] is not None else None,
                short_sale_buy=int(row["short_sale_buy"]) if row["short_sale_buy"] is not None else None,
                short_sale_cash_repayment=int(row["short_sale_cash_repayment"]) if row["short_sale_cash_repayment"] is not None else None,
                short_sale_limit=int(row["short_sale_limit"]) if row["short_sale_limit"] is not None else None,
                short_sale_sell=int(row["short_sale_sell"]) if row["short_sale_sell"] is not None else None,
                short_sale_today_balance=int(row["short_sale_today_balance"]) if row["short_sale_today_balance"] is not None else None,
                short_sale_yesterday_balance=int(row["short_sale_yesterday_balance"]) if row["short_sale_yesterday_balance"] is not None else None,
                note=row["note"],
            )
            for row in rows
        ]

    def build_documents(self, query: StructuredQuery) -> list[Document]:
        if not query.ticker:
            return []
        profile = self._resolve_retrieval_profile(query)
        documents: list[Document] = []
        for builder_name in profile.builder_plan:
            documents.extend(getattr(self, builder_name)(query))
        return self._sorted(documents, profile)

    def _build_price_documents(self, query: StructuredQuery) -> list[Document]:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=query.time_range_days)
        bars = self.get_price_bars(query.ticker, start_date, end_date)
        if not bars:
            return []
        label = query.company_name or query.ticker
        latest = bars[0]
        earliest = bars[-1]
        high_price = max(item.high_price for item in bars)
        low_price = min(item.low_price for item in bars)
        average_close = sum(item.close_price for item in bars) / len(bars)
        change = latest.close_price - earliest.close_price
        url = self._build_finmind_url("TaiwanStockPrice", query.ticker, start_date, end_date)
        return [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{url}:range")),
                ticker=query.ticker,
                title=f"{label} \u8fd1 {query.time_range_days} \u5929\u80a1\u50f9\u5340\u9593",
                content=(
                    f"\u6839\u64da FinMind TaiwanStockPrice {start_date.isoformat()} \u81f3 {end_date.isoformat()} \u8cc7\u6599\uff0c"
                    f"{label} \u8fd1 {query.time_range_days} \u5929\u6700\u9ad8\u50f9\u70ba {high_price:.2f} \u5143\uff0c"
                    f"\u6700\u4f4e\u50f9\u70ba {low_price:.2f} \u5143\u3002"
                ),
                source_name="FinMind TaiwanStockPrice",
                source_type="market_data",
                source_tier=SourceTier.HIGH,
                url=url,
                published_at=latest.trading_date,
                topics=[Topic.NEWS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{url}:summary")),
                ticker=query.ticker,
                title=f"{label} \u8fd1 {query.time_range_days} \u5929\u50f9\u683c\u6458\u8981",
                content=(
                    f"\u6700\u65b0\u4ea4\u6613\u65e5 {latest.trading_date:%Y-%m-%d} \u6536\u76e4\u50f9\u7d04 {latest.close_price:.2f} \u5143\u3002"
                    f"\u8fd1 {query.time_range_days} \u5929\u5e73\u5747\u6536\u76e4\u50f9\u7d04 {average_close:.2f} \u5143\uff0c"
                    f"\u671f\u9593\u6536\u76e4\u8b8a\u52d5\u7d04 {change:.2f} \u5143\u3002"
                ),
                source_name="FinMind TaiwanStockPrice",
                source_type="market_data_summary",
                source_tier=SourceTier.HIGH,
                url=f"{url}#summary",
                published_at=latest.trading_date,
                topics=[Topic.NEWS],
            ),
        ]

    def _build_technical_indicator_documents(self, query: StructuredQuery) -> list[Document]:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(query.time_range_days, 90))
        bars = self.get_price_bars(query.ticker, start_date, end_date)
        if len(bars) < 35:
            return []

        ordered_bars = sorted(bars, key=lambda item: item.trading_date)
        latest_bar = ordered_bars[-1]
        previous_bar = ordered_bars[-2] if len(ordered_bars) >= 2 else ordered_bars[-1]
        label = query.company_name or query.ticker
        closes = [item.close_price for item in ordered_bars]

        rsi_value = self._calculate_rsi(closes, period=14)
        kd_values = self._calculate_kd(ordered_bars, period=9)
        macd_values = self._calculate_macd(closes, short_period=12, long_period=26, signal_period=9)
        bollinger_values = self._calculate_bollinger_bands(closes, period=20, standard_deviations=2)
        moving_average_values = self._calculate_moving_average_bias(closes, short_period=5, long_period=20)
        if (
            rsi_value is None
            or kd_values is None
            or macd_values is None
            or bollinger_values is None
            or moving_average_values is None
        ):
            return []
        k_value, d_value = kd_values
        macd_line, signal_line, histogram = macd_values
        upper_band, middle_band, lower_band = bollinger_values
        ma5_value, ma20_value, ma5_bias, ma20_bias = moving_average_values
        overbought_status = self._technical_overbought_status(rsi_value, k_value, d_value)
        macd_trend = self._describe_macd_trend(macd_line, signal_line, histogram)
        bollinger_position = self._describe_bollinger_position(latest_bar.close_price, upper_band, middle_band, lower_band)
        bias_status = self._describe_bias_status(ma5_bias, ma20_bias)
        price_url = self._build_finmind_url("TaiwanStockPrice", query.ticker, start_date, end_date)

        snapshot_document = Document(
            id=str(uuid5(NAMESPACE_URL, f"{price_url}:technical-snapshot")),
            ticker=query.ticker,
            title=f"{label} 技術指標快照",
            content=(
                f"最新收盤價約 {latest_bar.close_price:.2f} 元。"
                f"RSI14 約 {rsi_value:.2f}。"
                f"K 值約 {k_value:.2f}，D 值約 {d_value:.2f}。"
                f"MACD 線約 {macd_line:.2f}，Signal 線約 {signal_line:.2f}，Histogram 約 {histogram:.2f}。"
                f"布林通道上軌約 {upper_band:.2f} 元，中軌約 {middle_band:.2f} 元，下軌約 {lower_band:.2f} 元。"
                f"MA5 約 {ma5_value:.2f} 元，MA20 約 {ma20_value:.2f} 元。"
                f"MA5 乖離率約 {ma5_bias:.2f}%，MA20 乖離率約 {ma20_bias:.2f}%。"
                f"{overbought_status}。"
            ),
            source_name="FinMind TaiwanStockPrice",
            source_type="technical_indicator_snapshot",
            source_tier=SourceTier.HIGH,
            url=f"{price_url}#technical-snapshot",
            published_at=latest_bar.trading_date,
            topics=[Topic.NEWS],
        )

        trend_direction = "偏強" if latest_bar.close_price >= previous_bar.close_price else "轉弱"
        assessment_document = Document(
            id=str(uuid5(NAMESPACE_URL, f"{price_url}:technical-assessment")),
            ticker=query.ticker,
            title=f"{label} 超買判讀",
            content=(
                f"最新交易日為 {latest_bar.trading_date:%Y-%m-%d}。"
                f"一般而言 RSI 70 以上、KD 80 以上屬超買觀察區。"
                f"目前 RSI14 約 {rsi_value:.2f}，K 值約 {k_value:.2f}，D 值約 {d_value:.2f}，"
                f"{overbought_status}。"
                f"MACD 動能{macd_trend}。"
                f"股價{bollinger_position}。"
                f"均線乖離維持{bias_status}。"
                f"最新收盤相較前一交易日表現{trend_direction}。"
            ),
            source_name="FinMind TaiwanStockPrice",
            source_type="technical_indicator_assessment",
            source_tier=SourceTier.HIGH,
            url=f"{price_url}#technical-assessment",
            published_at=latest_bar.trading_date,
            topics=[Topic.NEWS],
        )
        return [snapshot_document, assessment_document]

    def _build_season_line_margin_documents(self, query: StructuredQuery) -> list[Document]:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(query.time_range_days, 120))
        label = query.company_name or query.ticker
        bars = self.get_price_bars(query.ticker, start_date, end_date)
        margin_rows = self.get_margin_purchase_short_sale(query.ticker, start_date, end_date)
        news = self.get_stock_news(query.ticker, end_date - timedelta(days=30), end_date)

        documents: list[Document] = []
        season_line_document = self._build_season_line_document(label, query.ticker, bars, start_date, end_date)
        if season_line_document is not None:
            documents.append(season_line_document)

        margin_document = self._build_margin_balance_document(label, query.ticker, margin_rows, start_date, end_date)
        if margin_document is not None:
            documents.append(margin_document)

        documents.extend(
            self._build_news_documents(
                label,
                news,
                limit=2,
                keyword_filter=("\u878d\u8cc7", "\u878d\u5238", "\u7c4c\u78bc", "\u4fe1\u7528\u4ea4\u6613", "\u501f\u5238"),
            )
        )
        return self._sorted(documents)

    def _build_theme_impact_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        comparison_label = query.comparison_company_name or query.comparison_ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=max(query.time_range_days, 30))
        tickers = [(query.ticker, label)]
        if query.comparison_ticker and comparison_label:
            tickers.append((query.comparison_ticker, comparison_label))

        theme_keywords = self._extract_theme_keywords(query.user_query, label, comparison_label)
        documents: list[Document] = []
        per_ticker_limit = 3 if len(tickers) > 1 else 5
        for ticker, ticker_label in tickers:
            news = self.get_stock_news(ticker, since, today)
            documents.extend(
                self._build_news_documents(
                    ticker_label,
                    news,
                    limit=per_ticker_limit,
                    keyword_filter=theme_keywords or None,
                )
            )
        return self._sorted(documents)[:6]

    def _build_shipping_rate_impact_documents(self, query: StructuredQuery) -> list[Document]:
        primary_label = query.company_name or query.ticker
        comparison_label = query.comparison_company_name or query.comparison_ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=max(query.time_range_days, 30))
        labels = [label for label in (primary_label, comparison_label) if label]
        tickers = [(query.ticker, primary_label)]
        if query.comparison_ticker and comparison_label:
            tickers.append((query.comparison_ticker, comparison_label))

        rate_keywords = (
            "紅海",
            "紅海航線",
            "SCFI",
            "運價",
            "運價指數",
            "貨櫃",
            "航運",
            "繞道",
            "塞港",
            "支撐",
        )
        target_keywords = (
            "目標價",
            "分析師",
            "法人",
            "外資",
            "評等",
            "上修",
            "下修",
            "調整",
            "SCFI",
            "運價",
        )

        documents: list[Document] = []
        rate_documents: list[Document] = []
        target_documents: list[Document] = []
        aggregate_label = "、".join(labels)

        for ticker, ticker_label in tickers:
            news = self.get_stock_news(ticker, since, today)
            rate_documents.extend(
                self._build_news_documents(
                    ticker_label,
                    news,
                    limit=3,
                    keyword_filter=rate_keywords,
                    minimum_tier=SourceTier.MEDIUM,
                )
            )
            target_documents.extend(
                self._build_news_documents(
                    ticker_label,
                    news,
                    limit=3,
                    keyword_filter=target_keywords,
                    minimum_tier=SourceTier.MEDIUM,
                )
            )

        if rate_documents:
            latest_published_at = max(item.published_at for item in rate_documents)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"https://local.summary/{query.ticker}/shipping-rate-impact#support")),
                    ticker=query.ticker,
                    title=f"{aggregate_label} 紅海與 SCFI 支撐摘要",
                    content=self._build_shipping_rate_support_excerpt(labels, rate_documents),
                    source_name="Multi-source shipping digest",
                    source_type="shipping_rate_support_digest",
                    source_tier=SourceTier.MEDIUM,
                    url=f"https://local.summary/{query.ticker}/shipping-rate-impact#support",
                    published_at=latest_published_at,
                    topics=[Topic.NEWS],
                )
            )

        if target_documents:
            latest_published_at = max(item.published_at for item in target_documents)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"https://local.summary/{query.ticker}/shipping-rate-impact#target-price")),
                    ticker=query.ticker,
                    title=f"{aggregate_label} 法人目標價調整摘要",
                    content=self._build_shipping_target_price_excerpt(labels, target_documents),
                    source_name="Multi-source shipping digest",
                    source_type="shipping_target_price_digest",
                    source_tier=SourceTier.MEDIUM,
                    url=f"https://local.summary/{query.ticker}/shipping-rate-impact#target-price",
                    published_at=latest_published_at,
                    topics=[Topic.NEWS],
                )
            )

        documents.extend(rate_documents[:4])
        documents.extend(target_documents[:4])
        return self._sorted(documents)[:8]

    def _build_electricity_cost_impact_documents(self, query: StructuredQuery) -> list[Document]:
        primary_label = query.company_name or query.ticker
        comparison_label = query.comparison_company_name or query.comparison_ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=max(query.time_range_days, 30))
        labels = [label for label in (primary_label, comparison_label) if label]
        tickers = [(query.ticker, primary_label)]
        if query.comparison_ticker and comparison_label:
            tickers.append((query.comparison_ticker, comparison_label))

        cost_keywords = (
            "工業電價",
            "電價",
            "漲價",
            "調漲",
            "電費",
            "用電大戶",
            "成本",
            "毛利",
            "壓力",
        )
        response_keywords = (
            "因應",
            "對策",
            "節能",
            "節電",
            "降耗",
            "轉嫁",
            "調價",
            "自發電",
            "綠電",
            "效率",
        )

        documents: list[Document] = []
        cost_documents: list[Document] = []
        response_documents: list[Document] = []
        aggregate_label = "、".join(labels)

        for ticker, ticker_label in tickers:
            news = self.get_stock_news(ticker, since, today)
            cost_documents.extend(
                self._build_news_documents(
                    ticker_label,
                    news,
                    limit=3,
                    keyword_filter=cost_keywords,
                    minimum_tier=SourceTier.MEDIUM,
                )
            )
            response_documents.extend(
                self._build_news_documents(
                    ticker_label,
                    news,
                    limit=3,
                    keyword_filter=response_keywords,
                    minimum_tier=SourceTier.MEDIUM,
                )
            )

        if cost_documents:
            latest_published_at = max(item.published_at for item in cost_documents)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"https://local.summary/{query.ticker}/electricity-cost-impact#cost")),
                    ticker=query.ticker,
                    title=f"{aggregate_label} 電價成本壓力摘要",
                    content=self._build_electricity_cost_excerpt(labels, cost_documents),
                    source_name="Multi-source electricity digest",
                    source_type="electricity_cost_pressure_digest",
                    source_tier=SourceTier.MEDIUM,
                    url=f"https://local.summary/{query.ticker}/electricity-cost-impact#cost",
                    published_at=latest_published_at,
                    topics=[Topic.NEWS],
                )
            )

        if response_documents:
            latest_published_at = max(item.published_at for item in response_documents)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"https://local.summary/{query.ticker}/electricity-cost-impact#response")),
                    ticker=query.ticker,
                    title=f"{aggregate_label} 電價調漲因應對策摘要",
                    content=self._build_electricity_response_excerpt(labels, response_documents),
                    source_name="Multi-source electricity digest",
                    source_type="electricity_response_digest",
                    source_tier=SourceTier.MEDIUM,
                    url=f"https://local.summary/{query.ticker}/electricity-cost-impact#response",
                    published_at=latest_published_at,
                    topics=[Topic.NEWS],
                )
            )

        documents.extend(cost_documents[:4])
        documents.extend(response_documents[:4])
        return self._sorted(documents)[:8]

    def _build_macro_yield_sentiment_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=max(query.time_range_days, 30))
        news = self.get_stock_news(query.ticker, since, today)

        sentiment_keywords = (
            "CPI",
            "cpi",
            "通膨",
            "利率",
            "殖利率",
            "高殖利率",
            "金控",
            "法人",
            "外資",
            "觀點",
            "防禦",
            "美債",
        )
        view_keywords = (
            "法人",
            "外資",
            "觀點",
            "看法",
            "觀望",
            "保守",
            "降息",
            "升息",
            "殖利率",
            "高殖利率",
            "金控",
        )

        sentiment_documents = self._build_news_documents(
            label,
            news,
            limit=4,
            keyword_filter=sentiment_keywords,
            minimum_tier=SourceTier.MEDIUM,
        )
        view_documents = self._build_news_documents(
            label,
            news,
            limit=4,
            keyword_filter=view_keywords,
            minimum_tier=SourceTier.MEDIUM,
        )

        documents: list[Document] = []
        if sentiment_documents:
            latest_published_at = max(item.published_at for item in sentiment_documents)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"https://local.summary/{query.ticker}/macro-yield-sentiment#macro")),
                    ticker=query.ticker,
                    title=f"{label} 高殖利率情緒摘要",
                    content=self._build_macro_yield_sentiment_excerpt(label, sentiment_documents),
                    source_name="Multi-source macro digest",
                    source_type="macro_yield_sentiment_digest",
                    source_tier=SourceTier.MEDIUM,
                    url=f"https://local.summary/{query.ticker}/macro-yield-sentiment#macro",
                    published_at=latest_published_at,
                    topics=[Topic.NEWS],
                )
            )

        if view_documents:
            latest_published_at = max(item.published_at for item in view_documents)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"https://local.summary/{query.ticker}/macro-yield-sentiment#view")),
                    ticker=query.ticker,
                    title=f"{label} 法人最新觀點摘要",
                    content=self._build_macro_yield_view_excerpt(label, view_documents),
                    source_name="Multi-source macro digest",
                    source_type="macro_yield_view_digest",
                    source_tier=SourceTier.MEDIUM,
                    url=f"https://local.summary/{query.ticker}/macro-yield-sentiment#view",
                    published_at=latest_published_at,
                    topics=[Topic.NEWS],
                )
            )

        documents.extend(sentiment_documents[:4])
        documents.extend(view_documents[:4])
        return self._sorted(documents)[:8]

    def _build_revenue_growth_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=max(query.time_range_days, 90))
        news = self.get_stock_news(query.ticker, since, today)
        documents = self._build_news_documents(
            label,
            news,
            limit=5,
            keyword_filter=(
                "AI伺服器",
                "AI 伺服器",
                "伺服器",
                "AI",
                "營收",
                "占比",
                "比重",
                "成長",
                "2026",
                "倍增",
            ),
        )
        return self._sorted(documents)

    def _build_listing_revenue_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=max(query.time_range_days, 30))
        documents: list[Document] = []

        revenue_points = self.get_monthly_revenue_points(query.ticker, today - timedelta(days=400), today)
        if revenue_points:
            latest_point = max(revenue_points, key=lambda item: item.revenue_month)
            latest_report_date = latest_point.report_date or latest_point.revenue_month
            monthly_revenue_url = self._twse_financial_client._monthly_revenue_url if self._twse_financial_client else ""
            signal_text = self._describe_revenue_signal_strength(latest_point)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{monthly_revenue_url}:listing-revenue:{query.ticker}:{latest_point.revenue_month:%Y-%m}")),
                    ticker=query.ticker,
                    title=f"{label} 最新月營收增長訊號",
                    content=self._build_latest_monthly_revenue_excerpt(label, latest_point) + signal_text,
                    source_name="TWSE OpenAPI Monthly Revenue",
                    source_type="listing_revenue_signal",
                    source_tier=SourceTier.HIGH,
                    url=f"{monthly_revenue_url}#listing-revenue-signal",
                    published_at=latest_report_date,
                    topics=[Topic.EARNINGS, Topic.NEWS],
                )
            )

        news = self.get_stock_news(query.ticker, since, today)
        filtered_news_documents = self._build_news_documents(
            label,
            news,
            limit=4,
            keyword_filter=(
                "IPO",
                "上市",
                "掛牌",
                "蜜月行情",
                "承銷",
                "轉上市",
                "股價",
                "波動",
                "營收",
                "年增",
                "月增",
                "航線",
                "燃油附加費",
                "旅展",
            ),
            minimum_tier=SourceTier.MEDIUM,
        )
        if not filtered_news_documents:
            filtered_news_documents = self._build_news_documents(
                label,
                news,
                limit=3,
                minimum_tier=SourceTier.MEDIUM,
            )
        documents.extend(filtered_news_documents)
        return self._sorted(documents)

    def _build_guidance_reaction_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=max(query.time_range_days, 30))
        news = self.get_stock_news(query.ticker, since, today)
        guidance_news = self._build_news_documents(
            label,
            news,
            limit=6,
            keyword_filter=(
                "法說",
                "法說會",
                "營運指引",
                "指引",
                "下半年",
                "展望",
                "財測",
                "法人",
                "外資",
                "目標價",
                "評等",
                "會後",
            ),
            minimum_tier=SourceTier.MEDIUM,
        )
        if not guidance_news:
            return []

        positive_items: list[Document] = []
        negative_items: list[Document] = []
        for document in guidance_news:
            sentiment = self._classify_guidance_document(document)
            if sentiment == "positive":
                positive_items.append(document)
            elif sentiment == "negative":
                negative_items.append(document)

        documents: list[Document] = []
        aggregate_url_base = f"https://local.summary/{query.ticker}/guidance-reaction"
        published_at = max(item.published_at for item in guidance_news)

        if positive_items:
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{aggregate_url_base}:positive")),
                    ticker=query.ticker,
                    title=f"{label} 法說後正面反應整理",
                    content=self._build_guidance_reaction_excerpt(label, positive_items, "positive"),
                    source_name="Multi-source guidance digest",
                    source_type="guidance_reaction_positive",
                    source_tier=SourceTier.MEDIUM,
                    url=f"{aggregate_url_base}#positive",
                    published_at=published_at,
                    topics=[Topic.NEWS, Topic.EARNINGS],
                )
            )

        if negative_items:
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{aggregate_url_base}:negative")),
                    ticker=query.ticker,
                    title=f"{label} 法說後負面反應整理",
                    content=self._build_guidance_reaction_excerpt(label, negative_items, "negative"),
                    source_name="Multi-source guidance digest",
                    source_type="guidance_reaction_negative",
                    source_tier=SourceTier.MEDIUM,
                    url=f"{aggregate_url_base}#negative",
                    published_at=published_at,
                    topics=[Topic.NEWS, Topic.EARNINGS],
                )
            )

        documents.extend(guidance_news)
        return self._sorted(documents)

    def _build_season_line_document(
        self,
        label: str,
        ticker: str,
        bars: list[PriceBar],
        start_date: date,
        end_date: date,
    ) -> Document | None:
        ordered_bars = sorted(bars, key=lambda item: item.trading_date)
        closes = [item.close_price for item in ordered_bars]
        if len(closes) < 60:
            return None

        latest_bar = ordered_bars[-1]
        latest_close = latest_bar.close_price
        season_line = self._calculate_simple_moving_average(closes, period=60)
        if season_line is None:
            return None

        previous_close = ordered_bars[-2].close_price if len(ordered_bars) >= 2 else latest_close
        previous_season_line = self._calculate_simple_moving_average(closes[:-1], period=60) or season_line
        season_line_status = self._describe_season_line_status(
            latest_close,
            season_line,
            previous_close,
            previous_season_line,
        )
        season_line_bias = ((latest_close - season_line) / season_line) * 100 if season_line else 0.0
        price_url = self._build_finmind_url("TaiwanStockPrice", ticker, start_date, end_date)
        return Document(
            id=str(uuid5(NAMESPACE_URL, f"{price_url}:season-line-status")),
            ticker=ticker,
            title=f"{label} 季線位置觀察",
            content=(
                f"最新交易日為 {latest_bar.trading_date:%Y-%m-%d}。"
                f"最新收盤價約 {latest_close:.2f} 元。"
                f"季線(MA60)約 {season_line:.2f} 元。"
                f"與季線乖離約 {season_line_bias:.2f}%。"
                f"目前{season_line_status}。"
            ),
            source_name="FinMind TaiwanStockPrice",
            source_type="season_line_status",
            source_tier=SourceTier.HIGH,
            url=f"{price_url}#season-line-status",
            published_at=latest_bar.trading_date,
            topics=[Topic.NEWS],
        )

    def _build_margin_balance_document(
        self,
        label: str,
        ticker: str,
        margin_rows: list[MarginPurchaseShortSale],
        start_date: date,
        end_date: date,
    ) -> Document | None:
        ordered_rows = sorted(margin_rows, key=lambda item: item.trading_date)
        usable_rows = [item for item in ordered_rows if item.margin_purchase_today_balance is not None]
        if not usable_rows:
            return None

        latest = usable_rows[-1]
        latest_balance = latest.margin_purchase_today_balance or 0
        latest_limit = latest.margin_purchase_limit or 0
        utilization_pct = ((latest_balance / latest_limit) * 100) if latest_limit else None

        window20 = usable_rows[-20:]
        average_20d = sum(item.margin_purchase_today_balance or 0 for item in window20) / len(window20)
        delta_vs_average_pct = ((latest_balance - average_20d) / average_20d * 100) if average_20d else None

        baseline_index = -5 if len(usable_rows) >= 5 else 0
        baseline_balance = usable_rows[baseline_index].margin_purchase_today_balance or latest_balance
        five_day_change = latest_balance - baseline_balance
        short_sale_balance = latest.short_sale_today_balance
        margin_status = self._describe_margin_balance_status(utilization_pct, delta_vs_average_pct)
        price_url = self._build_finmind_url("TaiwanStockMarginPurchaseShortSale", ticker, start_date, end_date)

        short_sale_sentence = ""
        if short_sale_balance is not None:
            short_sale_sentence = f"最新融券餘額約 {short_sale_balance:,} 張。"

        return Document(
            id=str(uuid5(NAMESPACE_URL, f"{price_url}:margin-balance-status")),
            ticker=ticker,
            title=f"{label} 融資餘額觀察",
            content=(
                f"最新資料日為 {latest.trading_date:%Y-%m-%d}。"
                f"最新融資餘額約 {latest_balance} 張。"
                f"融資限額約 {latest_limit} 張。"
                f"融資使用率約 {(utilization_pct or 0.0):.2f}%。"
                f"近 20 個交易日平均融資餘額約 {average_20d:.0f} 張。"
                f"相較近 20 日平均變動約 {(delta_vs_average_pct or 0.0):.2f}%。"
                f"近 5 個交易日融資餘額變動約 {five_day_change:+d} 張。"
                f"若以融資使用率與近 20 日均值推估，籌碼面屬{margin_status}。"
                f"{short_sale_sentence}"
            ),
            source_name="FinMind TaiwanStockMarginPurchaseShortSale",
            source_type="margin_balance_status",
            source_tier=SourceTier.HIGH,
            url=f"{price_url}#margin-balance-status",
            published_at=latest.trading_date,
            topics=[Topic.NEWS],
        )

    def _build_monthly_revenue_yoy_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone(timedelta(hours=8))).date()
        target_year = today.year
        requested_month_count = self._detect_cumulative_month_count(query.user_query, today)
        revenue_start = date(target_year - 1, 1, 1)
        revenues = self.get_monthly_revenue_points(query.ticker, revenue_start, today)
        if not revenues:
            return []

        latest_point = max(revenues, key=lambda item: item.revenue_month)
        requested_month = self._detect_requested_revenue_month(query.user_query)
        if requested_month is not None:
            requested_year = self._detect_requested_revenue_year(query.user_query, latest_point.revenue_month.year)
            month_specific_documents = self._build_monthly_revenue_mom_documents(
                query=query,
                revenues=revenues,
                latest_point=latest_point,
                today=today,
                requested_year=requested_year,
                requested_month=requested_month,
            )
            if month_specific_documents:
                return self._sorted(month_specific_documents)

        target_year = latest_point.revenue_month.year
        available_month_count = latest_point.revenue_month.month
        current_total = latest_point.cumulative_revenue
        previous_total = latest_point.prior_year_cumulative_revenue
        yoy_pct = latest_point.cumulative_yoy_pct
        availability_note = ""
        if requested_month_count > available_month_count:
            availability_note = (
                f"截至 {latest_point.revenue_month:%Y-%m}，官方月營收資料僅更新到今年前 {available_month_count} 個月。"
            )
        if current_total is None or previous_total is None or yoy_pct is None:
            revenue_map = {(item.revenue_month.year, item.revenue_month.month): item for item in revenues}
            current_points = [
                revenue_map[(target_year, month)]
                for month in range(1, min(requested_month_count, available_month_count) + 1)
                if (target_year, month) in revenue_map
            ]
            previous_points = [
                revenue_map[(target_year - 1, month)]
                for month in range(1, min(requested_month_count, available_month_count) + 1)
                if (target_year - 1, month) in revenue_map
            ]
            available_month_count = min(len(current_points), len(previous_points))
            if available_month_count <= 0:
                return []
            current_points = sorted(current_points, key=lambda item: item.revenue_month)[:available_month_count]
            previous_points = sorted(previous_points, key=lambda item: item.revenue_month)[:available_month_count]
            current_total = sum(item.revenue for item in current_points)
            previous_total = sum(item.revenue for item in previous_points)
            yoy_pct = ((current_total - previous_total) / previous_total * 100) if previous_total else None

        if current_total is None or previous_total is None or yoy_pct is None:
            return []

        monthly_revenue_url = (
            self._twse_financial_client._monthly_revenue_url if self._twse_financial_client else ""
        )
        report_date = latest_point.report_date or latest_point.revenue_month
        current_total_100m = self._to_hundred_million_revenue(current_total)
        previous_total_100m = self._to_hundred_million_revenue(previous_total)
        monthly_revenue_100m = self._to_hundred_million_revenue(latest_point.revenue)
        single_month_yoy = (
            f"單月年增率約 {latest_point.year_over_year_pct:.2f}%。"
            if latest_point.year_over_year_pct is not None
            else ""
        )
        notes_segment = f"備註：{latest_point.notes}。" if latest_point.notes else ""

        documents = [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{monthly_revenue_url}:revenue-yoy:{query.ticker}:{target_year}:{available_month_count}")),
                ticker=query.ticker,
                title=f"{label} {target_year} 年前 {available_month_count} 個月累計營收",
                content=(
                    f"{availability_note}"
                    f"{label} {target_year} 年前 {available_month_count} 個月累計營收約 {current_total_100m:.2f} 億元；"
                    f"{target_year - 1} 年同期約 {previous_total_100m:.2f} 億元；"
                    f"年增率約 {yoy_pct:.2f}%。"
                ),
                source_name="TWSE OpenAPI Monthly Revenue",
                source_type="monthly_revenue_yoy",
                source_tier=SourceTier.HIGH,
                url=f"{monthly_revenue_url}#revenue-yoy",
                published_at=report_date,
                topics=[Topic.EARNINGS, Topic.NEWS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{monthly_revenue_url}:revenue-detail:{query.ticker}:{target_year}:{available_month_count}")),
                ticker=query.ticker,
                title=f"{label} 最新月營收出表摘要",
                content=(
                    f"TWSE 最新出表日期為 {report_date:%Y-%m-%d}，資料年月為 {latest_point.revenue_month:%Y-%m}。"
                    f"單月營收約 {monthly_revenue_100m:.2f} 億元。"
                    f"{single_month_yoy}"
                    f"{notes_segment}"
                ),
                source_name="TWSE OpenAPI Monthly Revenue",
                source_type="monthly_revenue_report",
                source_tier=SourceTier.HIGH,
                url=f"{monthly_revenue_url}#revenue-report",
                published_at=report_date,
                topics=[Topic.EARNINGS, Topic.NEWS],
            ),
        ]
        return self._sorted(documents)

    def _build_monthly_revenue_mom_documents(
        self,
        query: StructuredQuery,
        revenues: list[MonthlyRevenuePoint],
        latest_point: MonthlyRevenuePoint,
        today: date,
        requested_year: int,
        requested_month: int,
    ) -> list[Document]:
        label = query.company_name or query.ticker
        monthly_revenue_url = (
            self._twse_financial_client._monthly_revenue_url if self._twse_financial_client else ""
        )
        latest_report_date = latest_point.report_date or latest_point.revenue_month
        revenue_map = {
            (item.revenue_month.year, item.revenue_month.month): item
            for item in revenues
        }
        requested_point = revenue_map.get((requested_year, requested_month))
        recent_news = self.get_stock_news(query.ticker, today - timedelta(days=120), today)

        if requested_point is None:
            documents = [
                Document(
                    id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{monthly_revenue_url}:revenue-unavailable:{query.ticker}:{requested_year}-{requested_month:02d}",
                        )
                    ),
                    ticker=query.ticker,
                    title=f"{label} {requested_year}-{requested_month:02d} 月營收尚未出表",
                    content=(
                        f"截至 {today:%Y-%m-%d}，官方月營收資料最新僅到 {latest_point.revenue_month:%Y-%m}，"
                        f"尚未公布 {requested_year}-{requested_month:02d} 月營收。"
                        "因此目前無法判定該月月增率是否超過 20%，也無法判定是否創下近一年新高。"
                    ),
                    source_name="TWSE OpenAPI Monthly Revenue",
                    source_type="monthly_revenue_availability",
                    source_tier=SourceTier.HIGH,
                    url=f"{monthly_revenue_url}#revenue-availability",
                    published_at=latest_report_date,
                    topics=[Topic.EARNINGS, Topic.NEWS],
                ),
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{monthly_revenue_url}:latest-revenue:{query.ticker}:{latest_point.revenue_month:%Y-%m}")),
                    ticker=query.ticker,
                    title=f"{label} 最新已公布月營收摘要",
                    content=self._build_latest_monthly_revenue_excerpt(label, latest_point),
                    source_name="TWSE OpenAPI Monthly Revenue",
                    source_type="monthly_revenue_report",
                    source_tier=SourceTier.HIGH,
                    url=f"{monthly_revenue_url}#latest-revenue",
                    published_at=latest_report_date,
                    topics=[Topic.EARNINGS, Topic.NEWS],
                ),
            ]
            documents.extend(
                self._build_news_documents(
                    label,
                    recent_news,
                    limit=3,
                    keyword_filter=("\u71df\u6536", "\u6708\u71df\u6536", "\u5e74\u589e", "\u6708\u589e", "\u51fa\u8ca8", "\u8a02\u55ae"),
                )
            )
            return documents

        monthly_revenue_100m = self._to_hundred_million_revenue(requested_point.revenue)
        mom_pct = requested_point.month_over_month_pct
        yoy_pct = requested_point.year_over_year_pct
        if mom_pct is None:
            previous_month_point = self._find_previous_month_revenue_point(revenues, requested_point)
            if previous_month_point is not None and previous_month_point.revenue:
                mom_pct = ((requested_point.revenue - previous_month_point.revenue) / previous_month_point.revenue) * 100

        if yoy_pct is None:
            previous_year_point = revenue_map.get((requested_point.revenue_month.year - 1, requested_point.revenue_month.month))
            if previous_year_point is not None and previous_year_point.revenue:
                yoy_pct = ((requested_point.revenue - previous_year_point.revenue) / previous_year_point.revenue) * 100

        high_status = self._describe_monthly_revenue_high_status(revenues, requested_point)
        market_view = self._describe_monthly_revenue_market_view(recent_news, requested_point)
        month_status = "月增率已超過 20%" if mom_pct is not None and mom_pct > 20 else "月增率未達 20%"
        report_date = requested_point.report_date or requested_point.revenue_month

        documents = [
            Document(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{monthly_revenue_url}:mom-review:{query.ticker}:{requested_point.revenue_month:%Y-%m}",
                    )
                ),
                ticker=query.ticker,
                title=f"{label} {requested_point.revenue_month:%Y-%m} 月營收月增觀察",
                content=(
                    f"{label} {requested_point.revenue_month:%Y-%m} 單月營收約 {monthly_revenue_100m:.2f} 億元。"
                    f"{self._format_monthly_revenue_pct_sentence('月增率', mom_pct)}"
                    f"{self._format_monthly_revenue_pct_sentence('年增率', yoy_pct)}"
                    f"{month_status}。"
                ),
                source_name="TWSE OpenAPI Monthly Revenue",
                source_type="monthly_revenue_mom",
                source_tier=SourceTier.HIGH,
                url=f"{monthly_revenue_url}#mom-review",
                published_at=report_date,
                topics=[Topic.EARNINGS, Topic.NEWS],
            ),
            Document(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{monthly_revenue_url}:one-year-high:{query.ticker}:{requested_point.revenue_month:%Y-%m}",
                    )
                ),
                ticker=query.ticker,
                title=f"{label} 近一年月營收高點觀察",
                content=high_status,
                source_name="TWSE OpenAPI Monthly Revenue",
                source_type="monthly_revenue_high_status",
                source_tier=SourceTier.HIGH,
                url=f"{monthly_revenue_url}#one-year-high",
                published_at=report_date,
                topics=[Topic.EARNINGS, Topic.NEWS],
            ),
        ]
        if market_view:
            documents.append(
                Document(
                    id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{monthly_revenue_url}:market-view:{query.ticker}:{requested_point.revenue_month:%Y-%m}",
                        )
                    ),
                    ticker=query.ticker,
                    title=f"{label} 月營收市場解讀摘要",
                    content=market_view,
                    source_name="TWSE OpenAPI Monthly Revenue x News",
                    source_type="monthly_revenue_market_view",
                    source_tier=SourceTier.MEDIUM,
                    url=f"{monthly_revenue_url}#market-view",
                    published_at=max(
                        [report_date] + [article.published_at for article in recent_news],
                    ),
                    topics=[Topic.EARNINGS, Topic.NEWS],
                )
            )
        documents.extend(
            self._build_news_documents(
                label,
                recent_news,
                limit=3,
                keyword_filter=("\u71df\u6536", "\u6708\u71df\u6536", "\u5e74\u589e", "\u6708\u589e", "\u5275\u9ad8", "\u8a02\u55ae", "\u51fa\u8ca8"),
            )
        )
        return documents

    def _build_pe_valuation_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=430)
        valuations = self.get_pe_valuation_points(query.ticker, start_date, today)
        valid_points = [item for item in valuations if item.pe_ratio is not None]
        if len(valid_points) < 6:
            return []

        ordered = sorted(valid_points, key=lambda item: item.valuation_month)
        latest_point = ordered[-1]
        pe_values = [item.pe_ratio for item in ordered if item.pe_ratio is not None]
        if not pe_values:
            return []

        low_pe = min(pe_values)
        high_pe = max(pe_values)
        median_pe = self._calculate_percentile(pe_values, 0.5)
        p25_pe = self._calculate_percentile(pe_values, 0.25)
        p75_pe = self._calculate_percentile(pe_values, 0.75)
        percentile = self._calculate_percentile_rank(pe_values, latest_point.pe_ratio or 0.0)
        valuation_zone = self._describe_pe_valuation_zone(percentile)
        entry_view = self._describe_pe_entry_view(percentile)
        peer_segment = ""
        if latest_point.peer_pe_ratio is not None:
            peer_segment = f"同業本益比約 {latest_point.peer_pe_ratio:.2f} 倍。"
        twse_url = f"{self._twse_financial_client._base_url}?code={query.ticker}" if self._twse_financial_client else ""

        documents = [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{twse_url}:pe-current")),
                ticker=query.ticker,
                title=f"{label} 目前本益比",
                content=(
                    f"截至 {latest_point.valuation_month:%Y-%m}，{label} 本益比約 {latest_point.pe_ratio:.2f} 倍。"
                    f"{peer_segment}"
                ),
                source_name="TWSE IIH Company Financial",
                source_type="pe_current",
                source_tier=SourceTier.HIGH,
                url=f"{twse_url}#pe-current",
                published_at=latest_point.valuation_month,
                topics=[Topic.NEWS, Topic.EARNINGS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{twse_url}:pe-history")),
                ticker=query.ticker,
                title=f"{label} 近 13 個月本益比區間",
                content=(
                    f"若以近 13 個月月資料觀察，本益比區間約 {low_pe:.2f} 至 {high_pe:.2f} 倍，"
                    f"中位數約 {median_pe:.2f} 倍，25 分位約 {p25_pe:.2f} 倍，75 分位約 {p75_pe:.2f} 倍。"
                    f"目前約落在近 13 個月歷史分位 {percentile * 100:.1f}% 左右，屬{valuation_zone}。"
                ),
                source_name="TWSE IIH Company Financial",
                source_type="pe_history",
                source_tier=SourceTier.HIGH,
                url=f"{twse_url}#pe-history",
                published_at=latest_point.valuation_month,
                topics=[Topic.NEWS, Topic.EARNINGS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{twse_url}:pe-entry-view")),
                ticker=query.ticker,
                title=f"{label} 估值進場評估",
                content=(
                    f"若以近 13 個月歷史本益比區間衡量，{label} 目前估值屬{valuation_zone}，"
                    f"對長期投資來說，{entry_view}。"
                ),
                source_name="TWSE IIH Company Financial",
                source_type="pe_assessment",
                source_tier=SourceTier.HIGH,
                url=f"{twse_url}#pe-assessment",
                published_at=latest_point.valuation_month,
                topics=[Topic.NEWS, Topic.EARNINGS],
            ),
        ]
        return self._sorted(documents)

    def _build_margin_turnaround_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        start_date = max(date(today.year - 2, 1, 1), today - timedelta(days=730))
        statements = self.get_financial_statement_items(query.ticker, start_date, today)
        snapshots = self._extract_gross_margin_snapshots(statements)
        if len(snapshots) < 2:
            return []

        latest_snapshot = snapshots[-1]
        previous_snapshot = snapshots[-2]
        latest_date = latest_snapshot["statement_date"]
        previous_date = previous_snapshot["statement_date"]
        latest_margin = float(latest_snapshot["gross_margin_pct"])
        previous_margin = float(previous_snapshot["gross_margin_pct"])
        latest_operating_income = float(latest_snapshot["operating_income_100m"])
        previous_operating_income = float(previous_snapshot["operating_income_100m"])
        latest_operating_margin = float(latest_snapshot["operating_margin_pct"])
        previous_operating_margin = float(previous_snapshot["operating_margin_pct"])

        gross_margin_status = self._describe_gross_margin_turnaround_status(previous_margin, latest_margin)
        operating_status = self._describe_operating_turnaround_status(
            previous_operating_income,
            latest_operating_income,
        )
        profitability_view = self._describe_profitability_improvement_view(
            previous_margin,
            latest_margin,
            previous_operating_income,
            latest_operating_income,
        )

        finmind_url = self._build_finmind_url(
            "TaiwanStockFinancialStatements",
            query.ticker,
            previous_date.date(),
            latest_date.date(),
        )

        documents = [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{finmind_url}:gross-margin-turnaround-latest")),
                ticker=query.ticker,
                title=f"{label} 最新季毛利率觀察",
                content=(
                    f"截至 {latest_date:%Y-%m-%d}，{label} 最新季營收約 {latest_snapshot['revenue_100m']:.2f} 億元，"
                    f"營業毛利約 {latest_snapshot['gross_profit_100m']:.2f} 億元，毛利率約 {latest_margin:.2f}%。"
                    f"上一季（{previous_date:%Y-%m-%d}）毛利率約 {previous_margin:.2f}%。"
                    f"{gross_margin_status}"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="gross_margin_turnaround_latest",
                source_tier=SourceTier.HIGH,
                url=f"{finmind_url}#gross-margin-turnaround-latest",
                published_at=latest_date,
                topics=[Topic.EARNINGS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{finmind_url}:operating-income-turnaround-latest")),
                ticker=query.ticker,
                title=f"{label} 最新季營業利益觀察",
                content=(
                    f"截至 {latest_date:%Y-%m-%d}，{label} 最新季營業利益約 {latest_operating_income:.2f} 億元，"
                    f"營業利益率約 {latest_operating_margin:.2f}%。"
                    f"上一季營業利益約 {previous_operating_income:.2f} 億元，"
                    f"營業利益率約 {previous_operating_margin:.2f}%。"
                    f"{operating_status}"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="operating_income_turnaround_latest",
                source_tier=SourceTier.HIGH,
                url=f"{finmind_url}#operating-income-turnaround-latest",
                published_at=latest_date,
                topics=[Topic.EARNINGS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{finmind_url}:turnaround-quarter-compare")),
                ticker=query.ticker,
                title=f"{label} 最新季與前季本業比較",
                content=(
                    f"若把最新季 {latest_date:%Y-%m-%d} 與前一季 {previous_date:%Y-%m-%d} 相比，"
                    f"毛利率由 {previous_margin:.2f}% 變動至 {latest_margin:.2f}%，"
                    f"營業利益由 {previous_operating_income:.2f} 億元變動至 {latest_operating_income:.2f} 億元。"
                    f"{gross_margin_status}{operating_status}"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="turnaround_quarter_compare",
                source_tier=SourceTier.HIGH,
                url=f"{finmind_url}#turnaround-quarter-compare",
                published_at=latest_date,
                topics=[Topic.EARNINGS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{finmind_url}:profitability-improvement-view")),
                ticker=query.ticker,
                title=f"{label} 實質獲利改善判讀",
                content=(
                    f"若同時觀察毛利率與營業利益，{profitability_view}"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="profitability_improvement_view",
                source_tier=SourceTier.HIGH,
                url=f"{finmind_url}#profitability-improvement-view",
                published_at=latest_date,
                topics=[Topic.EARNINGS],
            ),
        ]
        return self._sorted(documents)

    def _build_gross_margin_comparison_documents(self, query: StructuredQuery) -> list[Document]:
        if not query.comparison_ticker:
            return []

        primary_label = query.company_name or query.ticker
        comparison_label = query.comparison_company_name or query.comparison_ticker
        today = datetime.now(timezone.utc).date()
        start_date = max(date(today.year - 2, 1, 1), today - timedelta(days=730))
        primary_items = self.get_financial_statement_items(query.ticker, start_date, today)
        comparison_items = self.get_financial_statement_items(query.comparison_ticker, start_date, today)
        primary_snapshots = self._extract_gross_margin_snapshots(primary_items)
        comparison_snapshots = self._extract_gross_margin_snapshots(comparison_items)
        if not primary_snapshots or not comparison_snapshots:
            return []

        primary_by_date = {snapshot["statement_date"]: snapshot for snapshot in primary_snapshots}
        comparison_by_date = {snapshot["statement_date"]: snapshot for snapshot in comparison_snapshots}
        common_dates = sorted(set(primary_by_date).intersection(comparison_by_date))

        if common_dates:
            comparison_date = common_dates[-1]
            primary_snapshot = primary_by_date[comparison_date]
            comparison_snapshot = comparison_by_date[comparison_date]
            comparable_statement = True
        else:
            primary_snapshot = primary_snapshots[-1]
            comparison_snapshot = comparison_snapshots[-1]
            comparable_statement = False

        primary_margin = primary_snapshot["gross_margin_pct"]
        comparison_margin = comparison_snapshot["gross_margin_pct"]
        higher_label = primary_label if primary_margin >= comparison_margin else comparison_label
        margin_gap = abs(primary_margin - comparison_margin)
        efficiency_view = self._describe_gross_margin_efficiency_view(margin_gap)

        primary_url = self._build_finmind_url(
            "TaiwanStockFinancialStatements",
            query.ticker,
            primary_snapshot["statement_date"].date(),
            primary_snapshot["statement_date"].date(),
        )
        comparison_url = self._build_finmind_url(
            "TaiwanStockFinancialStatements",
            query.comparison_ticker,
            comparison_snapshot["statement_date"].date(),
            comparison_snapshot["statement_date"].date(),
        )

        documents = [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{primary_url}:gross-margin-latest")),
                ticker=query.ticker,
                title=f"{primary_label} 最新毛利率",
                content=(
                    f"截至 {primary_snapshot['statement_date']:%Y-%m-%d}，{primary_label} 營業收入約 "
                    f"{primary_snapshot['revenue_100m']:.2f} 億元，營業毛利約 {primary_snapshot['gross_profit_100m']:.2f} 億元，"
                    f"毛利率約 {primary_margin:.2f}%。"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="gross_margin_snapshot",
                source_tier=SourceTier.HIGH,
                url=f"{primary_url}#gross-margin-latest",
                published_at=primary_snapshot["statement_date"],
                topics=[Topic.EARNINGS],
            ),
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{comparison_url}:gross-margin-latest")),
                ticker=query.comparison_ticker,
                title=f"{comparison_label} 最新毛利率",
                content=(
                    f"截至 {comparison_snapshot['statement_date']:%Y-%m-%d}，{comparison_label} 營業收入約 "
                    f"{comparison_snapshot['revenue_100m']:.2f} 億元，營業毛利約 {comparison_snapshot['gross_profit_100m']:.2f} 億元，"
                    f"毛利率約 {comparison_margin:.2f}%。"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="gross_margin_snapshot",
                source_tier=SourceTier.HIGH,
                url=f"{comparison_url}#gross-margin-latest",
                published_at=comparison_snapshot["statement_date"],
                topics=[Topic.EARNINGS],
            ),
        ]

        if comparable_statement:
            comparison_content = (
                f"若以 {primary_snapshot['statement_date']:%Y-%m-%d} 最新可比口徑比較，"
                f"{primary_label} 毛利率約 {primary_margin:.2f}%，{comparison_label} 毛利率約 {comparison_margin:.2f}%，"
                f"由 {higher_label} 較高，高出 {margin_gap:.2f} 個百分點。"
                f"單看毛利率，{higher_label} 在定價能力或成本結構上{efficiency_view}；"
                f"但經營效率仍需搭配營益率、費用率與資產周轉一起看。"
            )
            comparison_published_at = max(
                primary_snapshot["statement_date"],
                comparison_snapshot["statement_date"],
            )
        else:
            comparison_content = (
                f"{primary_label} 最新可得財報日期為 {primary_snapshot['statement_date']:%Y-%m-%d}，"
                f"{comparison_label} 為 {comparison_snapshot['statement_date']:%Y-%m-%d}。"
                f"若以各自最新口徑觀察，{primary_label} 毛利率約 {primary_margin:.2f}%，"
                f"{comparison_label} 毛利率約 {comparison_margin:.2f}%，由 {higher_label} 較高；"
                "但兩者並非完全同一財報時點，解讀時需保守。"
            )
            comparison_published_at = max(
                primary_snapshot["statement_date"],
                comparison_snapshot["statement_date"],
            )

        documents.append(
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{primary_url}:{comparison_url}:gross-margin-comparison")),
                ticker=query.ticker,
                title=f"{primary_label} vs {comparison_label} 毛利率比較",
                content=comparison_content,
                source_name="FinMind TaiwanStockFinancialStatements Comparison",
                source_type="gross_margin_comparison",
                source_tier=SourceTier.HIGH,
                url=f"{primary_url}#gross-margin-comparison",
                published_at=comparison_published_at,
                topics=[Topic.EARNINGS],
            )
        )

        common_recent_dates = common_dates[-4:]
        if len(common_recent_dates) >= 2:
            primary_average_margin = sum(primary_by_date[item]["gross_margin_pct"] for item in common_recent_dates) / len(
                common_recent_dates
            )
            comparison_average_margin = sum(
                comparison_by_date[item]["gross_margin_pct"] for item in common_recent_dates
            ) / len(common_recent_dates)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{primary_url}:{comparison_url}:gross-margin-trend")),
                    ticker=query.ticker,
                    title=f"{primary_label} vs {comparison_label} 近 {len(common_recent_dates)} 季毛利率",
                    content=(
                        f"若看最近 {len(common_recent_dates)} 個可比較季度，{primary_label} 平均毛利率約 "
                        f"{primary_average_margin:.2f}%，{comparison_label} 平均毛利率約 "
                        f"{comparison_average_margin:.2f}%。"
                        "這可協助判斷最新一期毛利率差距是短期波動，還是較長期的成本結構差異。"
                    ),
                    source_name="FinMind TaiwanStockFinancialStatements Comparison",
                    source_type="gross_margin_trend",
                    source_tier=SourceTier.HIGH,
                    url=f"{primary_url}#gross-margin-trend",
                    published_at=common_recent_dates[-1],
                    topics=[Topic.EARNINGS],
                )
            )

        return self._sorted(documents)

    def _build_profitability_stability_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        history_years = max((query.time_range_days + 364) // 365, 5)
        start_date = date(today.year - history_years, 1, 1)
        statements = self.get_financial_statement_items(query.ticker, start_date, today)
        annual_metrics = self._extract_annual_profitability_metrics(statements)
        if not annual_metrics:
            return []

        available_years = sorted(annual_metrics.keys(), reverse=True)
        target_years = available_years[:5]
        target_years.sort()
        if len(target_years) < 3:
            return []

        documents: list[Document] = []
        summary_parts: list[str] = []
        profitable_years = 0
        loss_years: list[int] = []
        for year in target_years:
            metrics = annual_metrics[year]
            parent_net_income = metrics.get("parent_net_income")
            eps_value = metrics.get("eps")
            status = "獲利" if parent_net_income is not None and parent_net_income > 0 else "虧損"
            if status == "獲利":
                profitable_years += 1
            else:
                loss_years.append(year)
            summary_parts.append(
                f"{year} 年 {status}，歸屬母公司淨利約 {metrics['parent_net_income_100m']:.2f} 億元，EPS 約 {eps_value:.2f} 元"
            )

        latest_metrics = annual_metrics[target_years[-1]]
        financial_url = self._build_finmind_url(
            "TaiwanStockFinancialStatements",
            query.ticker,
            latest_metrics["statement_date"].date(),
            latest_metrics["statement_date"].date(),
        )
        documents.append(
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{financial_url}:profitability-history")),
                ticker=query.ticker,
                title=f"{label} 近五年年度獲利",
                content=f"{label} 近五個完整年度的獲利表現為：{'；'.join(summary_parts)}。",
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="profitability_history",
                source_tier=SourceTier.HIGH,
                url=f"{financial_url}#profitability-history",
                published_at=latest_metrics["statement_date"],
                topics=[Topic.EARNINGS],
            )
        )

        stability_view = self._describe_profitability_stability(target_years, annual_metrics)
        loss_segment = "；".join(f"{year} 年轉虧" for year in loss_years) if loss_years else "近五年沒有出現年度轉虧"
        documents.append(
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{financial_url}:profitability-stability")),
                ticker=query.ticker,
                title=f"{label} 獲利穩定性評估",
                content=(
                    f"若以 {target_years[0]} 至 {target_years[-1]} 年的年度財報觀察，{label}"
                    f"{stability_view}。近五年中有 {profitable_years} 年為正獲利，{loss_segment}。"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="profitability_stability_assessment",
                source_tier=SourceTier.HIGH,
                url=f"{financial_url}#profitability-stability",
                published_at=latest_metrics["statement_date"],
                topics=[Topic.EARNINGS],
            )
        )

        worst_year = min(
            target_years,
            key=lambda year: annual_metrics[year].get("parent_net_income", 0.0),
        )
        worst_metrics = annual_metrics[worst_year]
        if worst_metrics.get("parent_net_income", 0.0) < 0:
            previous_year = worst_year - 1 if (worst_year - 1) in annual_metrics else None
            reason_excerpt = self._build_loss_reason_excerpt(label, worst_year, worst_metrics, annual_metrics.get(previous_year))
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{financial_url}:loss-reason:{worst_year}")),
                    ticker=query.ticker,
                    title=f"{label} {worst_year} 年虧損原因",
                    content=reason_excerpt,
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_type="loss_reason_assessment",
                    source_tier=SourceTier.HIGH,
                    url=f"{financial_url}#loss-reason-{worst_year}",
                    published_at=worst_metrics["statement_date"],
                    topics=[Topic.EARNINGS],
                )
            )

            worst_year_news = self.get_stock_news(
                query.ticker,
                date(worst_year, 1, 1),
                min(date(worst_year, 12, 31), today),
            )
            documents.extend(
                self._build_news_documents(
                    label,
                    worst_year_news,
                    limit=2,
                    keyword_filter=(
                        "虧損",
                        "虧",
                        "獲利",
                        "利差",
                        "需求",
                        "價格",
                        "景氣",
                    ),
                )
            )

        return self._sorted(documents)

    def _build_debt_dividend_safety_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        start_date = date(today.year - 3, 1, 1)
        balance_sheet_items = self.get_balance_sheet_items(query.ticker, start_date, today)
        dividends = self.get_dividend_policies(query.ticker, start_date, today)

        balance_snapshots = self._extract_balance_sheet_snapshots(balance_sheet_items)
        if not balance_snapshots:
            return []

        latest_snapshot = balance_snapshots[-1]
        previous_snapshot = balance_snapshots[-2] if len(balance_snapshots) >= 2 else None
        year_ago_snapshot = self._find_year_ago_snapshot(balance_snapshots, latest_snapshot["statement_date"])
        recent_ratios = [item["debt_ratio"] for item in balance_snapshots[-8:] if item["debt_ratio"] is not None]

        debt_ratio_status = self._describe_debt_ratio_status(
            latest_snapshot["debt_ratio"],
            previous_snapshot["debt_ratio"] if previous_snapshot else None,
            year_ago_snapshot["debt_ratio"] if year_ago_snapshot else None,
        )

        documents: list[Document] = []
        balance_url = self._build_finmind_url(
            "TaiwanStockBalanceSheet",
            query.ticker,
            latest_snapshot["statement_date"].date(),
            latest_snapshot["statement_date"].date(),
        )
        documents.append(
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{balance_url}:debt-latest")),
                ticker=query.ticker,
                title=f"{label} 最新資產負債表重點",
                content=(
                    f"截至 {latest_snapshot['statement_date']:%Y-%m-%d}，{label} 資產總額約 {latest_snapshot['assets_100m']:.2f} 億元，"
                    f"負債總額約 {latest_snapshot['liabilities_100m']:.2f} 億元，負債比率約 {latest_snapshot['debt_ratio']:.2f}%。"
                    f"現金及約當現金約 {latest_snapshot['cash_100m']:.2f} 億元。"
                ),
                source_name="FinMind TaiwanStockBalanceSheet",
                source_type="balance_sheet_latest",
                source_tier=SourceTier.HIGH,
                url=f"{balance_url}#latest-balance-sheet",
                published_at=latest_snapshot["statement_date"],
                topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
            )
        )

        if previous_snapshot is not None:
            previous_segment = f"前一季約 {previous_snapshot['debt_ratio']:.2f}%。"
            if year_ago_snapshot is not None:
                previous_segment += f"去年同期約 {year_ago_snapshot['debt_ratio']:.2f}%。"
            ratio_range_segment = ""
            if recent_ratios:
                ratio_range_segment = (
                    f"近 {len(recent_ratios)} 季區間約 {min(recent_ratios):.2f}% 至 {max(recent_ratios):.2f}%。"
                )
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{balance_url}:debt-trend")),
                    ticker=query.ticker,
                    title=f"{label} 負債比率變化",
                    content=(
                        f"{label} 最新負債比率約 {latest_snapshot['debt_ratio']:.2f}%。"
                        f"{previous_segment}"
                        f"{ratio_range_segment}"
                        f"若與最近幾期相比，負債比率{debt_ratio_status}。"
                    ),
                    source_name="FinMind TaiwanStockBalanceSheet",
                    source_type="debt_ratio_trend",
                    source_tier=SourceTier.HIGH,
                    url=f"{balance_url}#debt-ratio-trend",
                    published_at=latest_snapshot["statement_date"],
                    topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
                )
            )

        annual_dividends = self._extract_annual_dividend_totals(dividends)
        latest_dividend_year = max(annual_dividends) if annual_dividends else None
        latest_dividend = annual_dividends.get(latest_dividend_year) if latest_dividend_year else None
        if latest_dividend is not None and latest_dividend["total_payout_100m"] is not None:
            coverage_ratio = latest_snapshot["cash_100m"] / latest_dividend["total_payout_100m"] if latest_dividend["total_payout_100m"] else 0.0
            payout_view = self._describe_cash_dividend_coverage(coverage_ratio)
            dividend_url = self._build_finmind_url(
                "TaiwanStockDividend",
                query.ticker,
                date(max(latest_dividend_year - 1, 2000), 1, 1),
                today,
            )
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{dividend_url}:cash-coverage")),
                    ticker=query.ticker,
                    title=f"{label} 現金股利支應能力",
                    content=(
                        f"{label} 最新可取得的現金股利年度為 {latest_dividend_year} 年，"
                        f"現金股利每股約 {latest_dividend['cash_dividend_per_share']:.3f} 元，"
                        f"現金股利發放總額約 {latest_dividend['total_payout_100m']:.2f} 億元。"
                        f"若以最新資產負債表的現金及約當現金約 {latest_snapshot['cash_100m']:.2f} 億元估算，"
                        f"約可覆蓋 {coverage_ratio:.2f} 倍。{payout_view}。"
                    ),
                    source_name="FinMind TaiwanStockBalanceSheet x TaiwanStockDividend",
                    source_type="cash_dividend_coverage",
                    source_tier=SourceTier.HIGH,
                    url=f"{dividend_url}#cash-dividend-coverage",
                    published_at=max(latest_snapshot["statement_date"], latest_dividend["published_at"]),
                    topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
                )
            )

        return self._sorted(documents)

    def _build_fcf_dividend_sustainability_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        raw_start_year = today.year - 4
        dividends = self.get_dividend_policies(query.ticker, date(raw_start_year, 1, 1), today)
        annual_dividends = self._extract_annual_dividend_totals(dividends)

        if annual_dividends:
            target_years = sorted(annual_dividends.keys(), reverse=True)[:3]
        else:
            target_years = [today.year - 4, today.year - 3, today.year - 2]
        target_years = sorted(target_years)

        cash_flow_start = date(min(target_years), 1, 1)
        cash_flows = self.get_cash_flow_statement_items(query.ticker, cash_flow_start, today)
        annual_cash_flows = self._extract_annual_fcf_metrics(cash_flows)

        documents: list[Document] = []
        for year in target_years:
            cash_flow_metric = annual_cash_flows.get(year)
            if cash_flow_metric is not None:
                cash_flow_url = self._build_finmind_url(
                    "TaiwanStockCashFlowsStatement",
                    query.ticker,
                    date(year, 1, 1),
                    date(year, 12, 31),
                )
                documents.append(
                    Document(
                        id=str(uuid5(NAMESPACE_URL, f"{cash_flow_url}:fcf:{year}")),
                        ticker=query.ticker,
                        title=f"{label} {year} 自由現金流",
                        content=(
                            f"根據 FinMind TaiwanStockCashFlowsStatement，{label} {year} 年營業活動淨現金流入約 "
                            f"{cash_flow_metric['operating_cash_flow_100m']:.2f} 億元，資本支出約 "
                            f"{cash_flow_metric['capex_100m']:.2f} 億元，推估自由現金流約 "
                            f"{cash_flow_metric['fcf_100m']:.2f} 億元。"
                        ),
                        source_name="FinMind TaiwanStockCashFlowsStatement",
                        source_type="cash_flow_statement",
                        source_tier=SourceTier.HIGH,
                        url=f"{cash_flow_url}#fcf",
                        published_at=cash_flow_metric["published_at"],
                        topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
                    )
                )

            dividend_metric = annual_dividends.get(year)
            if dividend_metric is not None:
                dividend_url = self._build_finmind_url(
                    "TaiwanStockDividend",
                    query.ticker,
                    date(max(year - 1, 2000), 1, 1),
                    today,
                )
                content = (
                    f"{label} {year} 年現金股利每股約 {dividend_metric['cash_dividend_per_share']:.3f} 元。"
                )
                if dividend_metric["total_payout_100m"] is not None:
                    content += (
                        f"依參與分派總股數約 {dividend_metric['shares']:.0f} 股估算，"
                        f"現金股利發放總額約 {dividend_metric['total_payout_100m']:.2f} 億元。"
                    )
                else:
                    content += "目前缺少參與分派總股數，無法穩定換算現金股利發放總額。"
                documents.append(
                    Document(
                        id=str(uuid5(NAMESPACE_URL, f"{dividend_url}:cash-total:{year}")),
                        ticker=query.ticker,
                        title=f"{label} {year} 現金股利發放總額",
                        content=content,
                        source_name="FinMind TaiwanStockDividend",
                        source_type="dividend_total",
                        source_tier=SourceTier.HIGH,
                        url=f"{dividend_url}#cash-total-{year}",
                        published_at=dividend_metric["published_at"],
                        topics=[Topic.ANNOUNCEMENT, Topic.EARNINGS],
                    )
                )

        overlapping_years = [
            year
            for year in target_years
            if year in annual_cash_flows and year in annual_dividends and annual_dividends[year]["total_payout_100m"] is not None
        ]
        if overlapping_years:
            coverage_segments: list[str] = []
            coverage_ratios: list[float] = []
            for year in overlapping_years:
                fcf_100m = annual_cash_flows[year]["fcf_100m"]
                payout_100m = annual_dividends[year]["total_payout_100m"]
                coverage_ratio = fcf_100m / payout_100m if payout_100m else 0.0
                coverage_ratios.append(coverage_ratio)
                coverage_segments.append(f"{year} 年約 {coverage_ratio:.2f} 倍")

            latest_year = overlapping_years[-1]
            assessment_url = self._build_finmind_url(
                "TaiwanStockCashFlowsStatement",
                query.ticker,
                date(overlapping_years[0], 1, 1),
                date(latest_year, 12, 31),
            )
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{assessment_url}:sustainability")),
                    ticker=query.ticker,
                    title=f"{label} 股利政策永續性評估",
                    content=(
                        f"以目前已揭露且可對齊的 {overlapping_years[0]} 至 {latest_year} 年資料觀察，"
                        f"自由現金流對現金股利發放總額的覆蓋倍數分別為：{'、'.join(coverage_segments)}。"
                        f"{self._describe_dividend_sustainability(coverage_ratios)}。"
                    ),
                    source_name="FinMind TaiwanStockCashFlowsStatement x TaiwanStockDividend",
                    source_type="dividend_sustainability",
                    source_tier=SourceTier.HIGH,
                    url=f"{assessment_url}#dividend-sustainability",
                    published_at=max(
                        annual_cash_flows[latest_year]["published_at"],
                        annual_dividends[latest_year]["published_at"],
                    ),
                    topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
                )
            )

        return self._sorted(documents)

    def _build_fundamental_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        start_date = max(date(today.year - 2, 1, 1), today - timedelta(days=730))
        statements = self.get_financial_statement_items(query.ticker, start_date, today)
        dividends = self.get_dividend_policies(query.ticker, start_date, today)
        news = self.get_stock_news(query.ticker, today - timedelta(days=30), today)
        documents: list[Document] = []
        documents.extend(self._build_eps_documents(label, query.ticker, statements, today))
        documents.extend(self._build_dividend_documents(label, query.ticker, dividends, today))
        documents.extend(
            self._build_news_documents(
                label,
                news,
                limit=3,
                keyword_filter=(
                    "\u80a1\u5229",
                    "\u914d\u606f",
                    "\u73fe\u91d1\u80a1\u5229",
                    "\u9664\u606f",
                    "\u8ca1\u5831",
                    "\u6cd5\u8aaa",
                    "EPS",
                    "\u71df\u6536",
                ),
            )
        )
        return self._sorted(documents)

    def _build_announcement_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=query.time_range_days)
        statements = self.get_financial_statement_items(query.ticker, today - timedelta(days=365), today)
        dividends = self.get_dividend_policies(query.ticker, today - timedelta(days=365), today)
        news = self.get_stock_news(query.ticker, since, today)
        documents: list[Document] = []
        documents.extend(self._build_dividend_documents(label, query.ticker, dividends, today, since))
        documents.extend(self._build_recent_statement_documents(label, query.ticker, statements, since))
        documents.extend(
            self._build_news_documents(
                label,
                news,
                limit=5,
                keyword_filter=(
                    "\u516c\u544a",
                    "\u516c\u4f48",
                    "\u80a1\u5229",
                    "\u914d\u606f",
                    "\u9664\u606f",
                    "\u6cd5\u8aaa",
                    "\u8ca1\u5831",
                    "\u8655\u5206",
                    "\u571f\u5730",
                    "\u8cc7\u7522",
                    "\u696d\u5916",
                    "\u5165\u5e33",
                    "\u8a8d\u5217",
                ),
            )
        )
        return self._sorted(documents)

    def _build_market_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        since = today - timedelta(days=query.time_range_days)
        statements = self.get_financial_statement_items(query.ticker, today - timedelta(days=365), today)
        dividends = self.get_dividend_policies(query.ticker, today - timedelta(days=365), today)
        news = self.get_stock_news(query.ticker, since, today)
        documents: list[Document] = []
        documents.extend(self._build_news_documents(label, news, limit=5))
        documents.extend(self._build_dividend_documents(label, query.ticker, dividends, today, since))
        documents.extend(self._build_recent_statement_documents(label, query.ticker, statements, since))
        return self._sorted(documents)

    def _build_dividend_yield_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        dividend_start = max(date(today.year - 2, 1, 1), today - timedelta(days=730))
        price_start = today - timedelta(days=45)
        dividends = self.get_dividend_policies(query.ticker, dividend_start, today)
        price_bars = self.get_price_bars(query.ticker, price_start, today)
        if not dividends:
            return []
        latest_dividend = sorted(dividends, key=self._dividend_published_at, reverse=True)[0]
        cash_dividend = self._cash_dividend_amount(latest_dividend)
        stock_dividend = self._stock_dividend_amount(latest_dividend)
        dividend_url = self._build_finmind_url("TaiwanStockDividend", query.ticker, dividend_start, today)
        policy_sentences = [
            f"{label} \u6700\u65b0\u53ef\u53d6\u5f97\u7684\u914d\u606f\u653f\u7b56\u986f\u793a\uff0c{latest_dividend.year_label or latest_dividend.date.year}\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend:.2f} \u5143\uff0c",
            f"\u80a1\u7968\u80a1\u5229\u7d04 {stock_dividend:.2f} \u5143\u3002",
        ]
        if latest_dividend.cash_ex_dividend_trading_date is not None:
            policy_sentences.append(f"\u9664\u606f\u4ea4\u6613\u65e5\u70ba {latest_dividend.cash_ex_dividend_trading_date:%Y-%m-%d}\u3002")
        if latest_dividend.cash_dividend_payment_date is not None:
            policy_sentences.append(f"\u73fe\u91d1\u80a1\u5229\u767c\u653e\u65e5\u70ba {latest_dividend.cash_dividend_payment_date:%Y-%m-%d}\u3002")
        documents = [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{dividend_url}:yield-policy:{latest_dividend.date:%Y-%m-%d}")),
                ticker=query.ticker,
                title=f"{label} \u6700\u65b0\u914d\u606f\u653f\u7b56",
                content="".join(policy_sentences),
                source_name="FinMind TaiwanStockDividend",
                source_type="dividend_policy",
                source_tier=SourceTier.HIGH,
                url=dividend_url,
                published_at=self._dividend_published_at(latest_dividend),
                topics=[Topic.ANNOUNCEMENT, Topic.EARNINGS],
            )
        ]
        if price_bars and cash_dividend > 0:
            latest_bar = price_bars[0]
            if latest_bar.close_price > 0:
                yield_pct = cash_dividend / latest_bar.close_price * 100
                price_url = self._build_finmind_url("TaiwanStockPrice", query.ticker, price_start, today)
                documents.append(
                    Document(
                        id=str(uuid5(NAMESPACE_URL, f"{price_url}:cash-yield:{latest_bar.trading_date:%Y-%m-%d}")),
                        ticker=query.ticker,
                        title=f"{label} \u73fe\u91d1\u6b96\u5229\u7387\u63db\u7b97",
                        content=(
                            f"\u6700\u65b0\u6536\u76e4\u50f9\u7d04 {latest_bar.close_price:.2f} \u5143\u3002"
                            f"\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend:.2f} \u5143\uff0c\u63db\u7b97\u73fe\u91d1\u6b96\u5229\u7387\u7d04 {yield_pct:.2f}%\u3002"
                        ),
                        source_name="FinMind TaiwanStockPrice x TaiwanStockDividend",
                        source_type="dividend_yield",
                        source_tier=SourceTier.HIGH,
                        url=f"{price_url}#cash-yield",
                        published_at=latest_bar.trading_date,
                        topics=[Topic.ANNOUNCEMENT, Topic.NEWS],
                    )
                )
        return self._sorted(documents)

    def _build_ex_dividend_documents(self, query: StructuredQuery) -> list[Document]:
        label = query.company_name or query.ticker
        today = datetime.now(timezone.utc).date()
        dividend_start = max(date(today.year - 2, 1, 1), today - timedelta(days=730))
        dividends = self.get_dividend_policies(query.ticker, dividend_start, today)
        if not dividends:
            return []
        target_policy = next(
            (item for item in sorted(dividends, key=self._dividend_published_at, reverse=True) if item.cash_ex_dividend_trading_date is not None),
            None,
        )
        if target_policy is None:
            target_policy = sorted(dividends, key=self._dividend_published_at, reverse=True)[0]
        ex_date = (target_policy.cash_ex_dividend_trading_date or target_policy.date).date()
        cash_dividend = self._cash_dividend_amount(target_policy)
        dividend_url = self._build_finmind_url("TaiwanStockDividend", query.ticker, dividend_start, today)
        documents: list[Document] = [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{dividend_url}:ex-dividend:{ex_date.isoformat()}")),
                ticker=query.ticker,
                title=f"{label} \u6700\u65b0\u9664\u606f\u4e8b\u4ef6",
                content=self._build_ex_dividend_policy_excerpt(label, target_policy, cash_dividend),
                source_name="FinMind TaiwanStockDividend",
                source_type="dividend_policy",
                source_tier=SourceTier.HIGH,
                url=dividend_url,
                published_at=self._dividend_published_at(target_policy),
                topics=[Topic.ANNOUNCEMENT, Topic.NEWS],
            )
        ]
        price_start = ex_date - timedelta(days=7)
        price_end = ex_date + timedelta(days=7)
        price_bars = self.get_price_bars(query.ticker, price_start, price_end)
        event_document = self._build_ex_dividend_price_document(label, query.ticker, ex_date, cash_dividend, price_bars)
        if event_document is not None:
            documents.append(event_document)
        news = self.get_stock_news(query.ticker, ex_date - timedelta(days=3), ex_date + timedelta(days=3))
        documents.extend(
            self._build_news_documents(
                label,
                news,
                limit=2,
                keyword_filter=("\u9664\u606f", "\u9664\u6b0a", "\u586b\u606f", "\u80a1\u5229", "\u914d\u606f"),
            )
        )
        return self._sorted(documents)

    def _build_eps_documents(
        self,
        label: str,
        ticker: str,
        statements: list[FinancialStatementItem],
        today: date,
    ) -> list[Document]:
        eps_rows = [item for item in statements if self._is_eps_item(item)]
        if not eps_rows:
            return []
        documents: list[Document] = []
        previous_year = today.year - 1
        previous_year_rows = [item for item in eps_rows if item.statement_date.year == previous_year]
        if previous_year_rows:
            annual_eps = sum(item.value for item in previous_year_rows)
            latest = max(previous_year_rows, key=lambda item: item.statement_date)
            url = self._build_finmind_url("TaiwanStockFinancialStatements", ticker, date(previous_year, 1, 1), date(previous_year, 12, 31))
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{url}:annual")),
                    ticker=ticker,
                    title=f"{label} {previous_year} \u5168\u5e74 EPS \u91cd\u9ede",
                    content=(
                        f"\u6839\u64da FinMind TaiwanStockFinancialStatements \u8cc7\u6599\uff0c{label} {previous_year} \u5168\u5e74 EPS \u7d04 {annual_eps:.2f} \u5143\u3002"
                    ),
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_type="financial_statement",
                    source_tier=SourceTier.HIGH,
                    url=url,
                    published_at=latest.statement_date,
                    topics=[Topic.EARNINGS],
                )
            )
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{url}:breakdown")),
                    ticker=ticker,
                    title=f"{label} {previous_year} EPS \u5b63\u5ea6\u62c6\u89e3",
                    content=self._build_eps_breakdown(label, previous_year_rows),
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_type="financial_statement_breakdown",
                    source_tier=SourceTier.HIGH,
                    url=f"{url}#breakdown",
                    published_at=latest.statement_date,
                    topics=[Topic.EARNINGS],
                )
            )
        latest = max(eps_rows, key=lambda item: item.statement_date)
        latest_url = self._build_finmind_url("TaiwanStockFinancialStatements", ticker, latest.statement_date.date(), latest.statement_date.date())
        documents.append(
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{latest_url}:latest")),
                ticker=ticker,
                title=f"{label} \u6700\u65b0\u4e00\u5b63 EPS",
                content=(
                    f"{label} \u6700\u65b0\u4e00\u5b63 EPS \u7d04 {latest.value:.2f} \u5143\uff0c\u8cc7\u6599\u65e5\u671f\u70ba {latest.statement_date:%Y-%m-%d}\u3002"
                ),
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="financial_statement_latest",
                source_tier=SourceTier.HIGH,
                url=f"{latest_url}#latest",
                published_at=latest.statement_date,
                topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
            )
        )
        return documents

    def _build_dividend_documents(
        self,
        label: str,
        ticker: str,
        dividends: list[DividendPolicy],
        today: date,
        since: date | None = None,
    ) -> list[Document]:
        if not dividends:
            return []
        items = sorted(dividends, key=self._dividend_published_at, reverse=True)
        if since is not None:
            items = [item for item in items if self._dividend_published_at(item).date() >= since]
            if not items:
                return []
        latest = items[0]
        latest_cash = self._cash_dividend_amount(latest)
        latest_stock = self._stock_dividend_amount(latest)
        url = self._build_finmind_url("TaiwanStockDividend", ticker, date(today.year - 2, 1, 1), today)
        documents = [
            Document(
                id=str(uuid5(NAMESPACE_URL, f"{url}:latest:{latest.year_label}:{latest.date:%Y-%m-%d}")),
                ticker=ticker,
                title=f"{label} \u6700\u65b0\u80a1\u5229\u653f\u7b56",
                content=(
                    f"{label} \u6700\u65b0\u53ef\u53d6\u5f97\u7684\u80a1\u5229\u8cc7\u6599\u986f\u793a\uff0c{latest.year_label} \u73fe\u91d1\u80a1\u5229\u7d04 {latest_cash:.2f} \u5143\uff0c\u80a1\u7968\u80a1\u5229\u7d04 {latest_stock:.2f} \u5143\u3002"
                ),
                source_name="FinMind TaiwanStockDividend",
                source_type="dividend_policy",
                source_tier=SourceTier.HIGH,
                url=url,
                published_at=self._dividend_published_at(latest),
                topics=[Topic.ANNOUNCEMENT, Topic.EARNINGS],
            )
        ]
        previous = next((item for item in items[1:] if item.year_label and item.year_label != latest.year_label), None)
        if previous is not None:
            previous_cash = self._cash_dividend_amount(previous)
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{url}:previous:{previous.year_label}:{previous.date:%Y-%m-%d}")),
                    ticker=ticker,
                    title=f"{label} {previous.year_label} \u80a1\u5229\u53c3\u8003",
                    content=(
                        f"{label} \u5728 {previous.year_label} \u7684\u73fe\u91d1\u80a1\u5229\u7d04 {previous_cash:.2f} \u5143\u3002"
                    ),
                    source_name="FinMind TaiwanStockDividend",
                    source_type="dividend_policy_history",
                    source_tier=SourceTier.HIGH,
                    url=f"{url}#history-{previous.year_label}",
                    published_at=self._dividend_published_at(previous),
                    topics=[Topic.ANNOUNCEMENT, Topic.EARNINGS],
                )
            )
        return documents

    def _build_recent_statement_documents(
        self,
        label: str,
        ticker: str,
        statements: list[FinancialStatementItem],
        since: date,
    ) -> list[Document]:
        eps_rows = [item for item in statements if self._is_eps_item(item) and item.statement_date.date() >= since]
        if not eps_rows:
            return []
        grouped: dict[datetime, list[FinancialStatementItem]] = defaultdict(list)
        for row in eps_rows:
            grouped[row.statement_date].append(row)
        documents: list[Document] = []
        for statement_date in sorted(grouped.keys(), reverse=True)[:2]:
            item = grouped[statement_date][0]
            url = self._build_finmind_url("TaiwanStockFinancialStatements", ticker, statement_date.date(), statement_date.date())
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, f"{url}:announcement")),
                    ticker=ticker,
                    title=f"{label} {statement_date:%Y-%m-%d} \u8ca1\u5831\u66f4\u65b0",
                    content=(
                        f"{label} \u65bc {statement_date:%Y-%m-%d} \u7684\u8ca1\u5831\u63ed\u9732\u986f\u793a\uff0cEPS \u7d04 {item.value:.2f} \u5143\u3002"
                    ),
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_type="financial_statement_update",
                    source_tier=SourceTier.HIGH,
                    url=f"{url}#announcement",
                    published_at=statement_date,
                    topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
                )
            )
        return documents

    def _build_news_documents(
        self,
        label: str,
        news: list[NewsArticle],
        limit: int,
        keyword_filter: tuple[str, ...] | None = None,
        minimum_tier: SourceTier | None = None,
    ) -> list[Document]:
        if not news:
            return []
        items = news
        if minimum_tier is not None:
            items = [item for item in items if self._tier_rank(item.source_tier) >= self._tier_rank(minimum_tier)]
            if not items:
                return []
        if keyword_filter:
            scored_items = [
                (self._keyword_match_score(item, keyword_filter), item)
                for item in items
            ]
            items = [item for score, item in scored_items if score > 0]
            if not items:
                return []
            items = sorted(
                items,
                key=lambda item: (self._keyword_match_score(item, keyword_filter), item.published_at),
                reverse=True,
            )
        documents: list[Document] = []
        for article in items[:limit]:
            content = f"{label} \u65b0\u805e\uff1a{article.title}\u3002"
            if article.summary:
                content += article.summary
            documents.append(
                Document(
                    id=str(uuid5(NAMESPACE_URL, article.url)),
                    ticker=article.ticker,
                    title=article.title,
                    content=content,
                    source_name=article.source_name,
                    source_type="news_article",
                    source_tier=article.source_tier,
                    url=article.url,
                    published_at=article.published_at,
                    topics=self._infer_news_topics(article),
                )
            )
        return documents

    def _keyword_match_score(self, article: NewsArticle, keywords: tuple[str, ...]) -> int:
        haystack = f"{article.title} {article.summary or ''}".lower()
        score = 0
        for keyword in keywords:
            if keyword.lower() in haystack:
                score += 1
        return score

    def _extract_theme_keywords(
        self,
        query_text: str,
        label: str | None,
        comparison_label: str | None = None,
    ) -> tuple[str, ...]:
        keywords: list[str] = []
        compact_query = self._normalize_lookup_text(query_text)
        compact_label = self._normalize_lookup_text(label or "")
        compact_comparison_label = self._normalize_lookup_text(comparison_label or "")

        if "電動車" in query_text or "ev" in compact_query:
            keywords.extend(["電動車", "EV", "車用"])
        if any(token in query_text for token in ("電池", "正極", "正極材料")):
            keywords.extend(["電池", "正極", "正極材料", "鈷", "鎳", "三元", "城市採礦", "能源轉型"])
        if any(token in query_text for token in ("短線", "需求", "普及率", "放緩")):
            keywords.extend(["需求", "出貨", "訂單", "油價", "鈷價", "鎳價"])
        if any(token in compact_query for token in ("asml", "艾司摩爾", "半導體設備", "設備族群", "設備股", "設備")):
            keywords.extend(
                [
                    "ASML",
                    "艾司摩爾",
                    "半導體設備",
                    "設備",
                    "設備族群",
                    "設備股",
                    "EUV",
                    "曝光機",
                    "資本支出",
                    "擴產",
                    "訂單",
                    "先進製程",
                ]
            )
        if any(token in query_text for token in ("利空", "情緒", "展望", "不如預期", "觀望", "保守")):
            keywords.extend(["利空", "情緒", "展望", "不如預期", "觀望", "保守", "下修", "放緩"])

        if compact_label and compact_label in compact_query:
            keywords.append(label or "")
        if comparison_label and compact_comparison_label and compact_comparison_label in compact_query:
            keywords.append(comparison_label)

        unique_keywords: list[str] = []
        for keyword in keywords:
            if keyword and keyword not in unique_keywords:
                unique_keywords.append(keyword)
        return tuple(unique_keywords)

    def _generic_news_tokens(self, query_text: str) -> tuple[str, ...]:
        search_terms: list[str] = []
        if any(token in query_text for token in ("公告", "法說", "董事會", "股利", "除息", "財報")):
            search_terms.extend(("公告", "法說", "董事會", "股利", "除息", "財報"))
        if any(token in query_text for token in ("處分", "土地", "資產", "業外", "入帳")):
            search_terms.extend(("處分", "土地", "資產", "業外", "EPS", "入帳"))
        if any(token in query_text for token in ("利空", "情緒", "不如預期", "下修", "觀望")):
            search_terms.extend(("利空", "情緒", "不如預期", "下修", "觀望"))
        return self._dedupe_terms(search_terms)

    def _dedupe_terms(self, search_terms: list[str], limit: int = 10) -> tuple[str, ...]:
        unique_terms: list[str] = []
        for term in search_terms:
            if term and term not in unique_terms:
                unique_terms.append(term)
        return tuple(unique_terms[:limit])

    def _finalize_terms(
        self,
        search_terms: list[str],
        pinned_terms: list[str] | None = None,
        limit: int = 10,
    ) -> tuple[str, ...]:
        pinned = list(self._dedupe_terms(pinned_terms or [], limit=limit))
        slots = max(limit - len(pinned), 0)
        base_terms = [term for term in search_terms if term not in pinned]
        return tuple(list(self._dedupe_terms(base_terms, limit=slots)) + pinned)

    def _build_eps_breakdown(self, label: str, items: list[FinancialStatementItem]) -> str:
        ordered = sorted(items, key=lambda item: item.statement_date)
        parts = [f"{item.statement_date:%Y-%m-%d} EPS {item.value:.2f} \u5143" for item in ordered]
        return f"{label} \u5168\u5e74 EPS \u5b63\u5ea6\u62c6\u89e3\u5982\u4e0b\uff1a" + "\u3001".join(parts) + "\u3002"

    def _build_ex_dividend_policy_excerpt(self, label: str, policy: DividendPolicy, cash_dividend: float) -> str:
        parts = [f"{label} \u6700\u65b0\u53ef\u78ba\u8a8d\u7684\u73fe\u91d1\u80a1\u5229\u7d04 {cash_dividend:.2f} \u5143\u3002"]
        if policy.cash_ex_dividend_trading_date is not None:
            parts.append(f"\u9664\u606f\u4ea4\u6613\u65e5\u70ba {policy.cash_ex_dividend_trading_date:%Y-%m-%d}\u3002")
        if policy.cash_dividend_payment_date is not None:
            parts.append(f"\u73fe\u91d1\u80a1\u5229\u767c\u653e\u65e5\u70ba {policy.cash_dividend_payment_date:%Y-%m-%d}\u3002")
        return "".join(parts)

    def _build_ex_dividend_price_document(
        self,
        label: str,
        ticker: str,
        ex_date: date,
        cash_dividend: float,
        price_bars: list[PriceBar],
    ) -> Document | None:
        if cash_dividend <= 0 or not price_bars:
            return None
        ordered = sorted(price_bars, key=lambda item: item.trading_date)
        ex_bar = next((item for item in ordered if item.trading_date.date() == ex_date), None)
        previous_candidates = [item for item in ordered if item.trading_date.date() < ex_date]
        if ex_bar is None or not previous_candidates:
            return None
        previous_bar = previous_candidates[-1]
        reference_price = max(previous_bar.close_price - cash_dividend, 0.0)
        intraday_fill_ratio = max(0.0, (ex_bar.high_price - reference_price) / cash_dividend * 100)
        close_fill_ratio = max(0.0, (ex_bar.close_price - reference_price) / cash_dividend * 100)
        same_day_fill = ex_bar.high_price >= previous_bar.close_price
        reaction = self._describe_market_reaction(ex_bar.close_price, reference_price, close_fill_ratio)
        volume_sentence = ""
        if ex_bar.trading_volume is not None and previous_bar.trading_volume not in (None, 0):
            volume_change = (ex_bar.trading_volume - previous_bar.trading_volume) / previous_bar.trading_volume * 100
            volume_sentence = f"\u7576\u65e5\u6210\u4ea4\u91cf\u8f03\u524d\u4e00\u4ea4\u6613\u65e5\u8b8a\u52d5\u7d04 {volume_change:.2f}%\u3002"
        price_url = self._build_finmind_url("TaiwanStockPrice", ticker, ex_date - timedelta(days=7), ex_date + timedelta(days=7))
        same_day_fill_text = "\u7576\u5929\u76e4\u4e2d\u5b8c\u6210\u586b\u606f" if same_day_fill else "\u7576\u5929\u672a\u5b8c\u6210\u586b\u606f"
        content = (
            f"\u9664\u606f\u524d\u4e00\u4ea4\u6613\u65e5\u6536\u76e4\u50f9\u7d04 {previous_bar.close_price:.2f} \u5143\uff0c"
            f"\u9664\u606f\u7576\u5929\u53c3\u8003\u50f9\u7d04 {reference_price:.2f} \u5143\u3002"
            f"\u9664\u606f\u7576\u5929\u6536\u76e4\u50f9\u7d04 {ex_bar.close_price:.2f} \u5143\uff0c\u76e4\u4e2d\u6700\u9ad8\u50f9\u7d04 {ex_bar.high_price:.2f} \u5143\u3002"
            f"\u76e4\u4e2d\u6700\u9ad8\u586b\u606f\u7387\u7d04 {intraday_fill_ratio:.2f}%\uff0c\u6536\u76e4\u586b\u606f\u7387\u7d04 {close_fill_ratio:.2f}%\u3002"
            f"{same_day_fill_text}\u3002\u5e02\u5834\u53cd\u61c9\u504f{reaction}\u3002{volume_sentence}"
        )
        return Document(
            id=str(uuid5(NAMESPACE_URL, f"{price_url}:ex-dividend-performance:{ex_date.isoformat()}")),
            ticker=ticker,
            title=f"{label} \u9664\u606f\u7576\u5929\u586b\u606f\u8868\u73fe",
            content=content,
            source_name="FinMind TaiwanStockPrice x TaiwanStockDividend",
            source_type="ex_dividend_performance",
            source_tier=SourceTier.HIGH,
            url=f"{price_url}#ex-dividend-performance",
            published_at=ex_bar.trading_date,
            topics=[Topic.ANNOUNCEMENT, Topic.NEWS],
        )

    def _fetch_news_articles(
        self,
        ticker: str,
        company_name: str | None,
        start_date: date,
        end_date: date,
        search_terms: tuple[str, ...] = (),
    ) -> list[NewsArticle]:
        if self._news_pipeline is not None:
            items = self._news_pipeline.fetch_stock_news(
                ticker=ticker,
                company_name=company_name,
                start_date=start_date,
                end_date=end_date,
                search_terms=search_terms,
            )
            if items:
                return items
        return self._finmind_client.fetch_stock_news(ticker, start_date, end_date)

    def _lookup_stock_name(self, ticker: str) -> str | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                text("SELECT stock_name FROM stock_info WHERE stock_id = :ticker LIMIT 1"),
                {"ticker": ticker},
            ).mappings().first()
        if row is None:
            return None
        stock_name = row.get("stock_name")
        return str(stock_name) if stock_name else None

    def _build_news_search_terms(
        self,
        query: StructuredQuery,
        label: str | None,
    ) -> tuple[str, ...]:
        profile = self._resolve_retrieval_profile(query)
        search_terms: list[str] = []
        if profile.search_term_strategy == "theme":
            search_terms.extend(
                self._extract_theme_keywords(
                    query.user_query,
                    label,
                    query.comparison_company_name,
                )
            )
        elif profile.search_term_strategy == "generic_news":
            search_terms.extend(self._generic_news_tokens(query.user_query))
        else:
            search_terms.extend(profile.news_search_term_seeds)
            pinned_terms: list[str] = []
            if profile.append_primary_label and label:
                pinned_terms.append(label)
            if profile.include_comparison and query.comparison_company_name:
                pinned_terms.append(query.comparison_company_name)
            return self._finalize_terms(search_terms, pinned_terms)
        return self._dedupe_terms(search_terms)

    def _build_shipping_rate_support_excerpt(
        self,
        labels: list[str],
        documents: list[Document],
    ) -> str:
        combined_text = " ".join(f"{item.title} {item.content}" for item in documents)
        normalized_text = self._normalize_lookup_text(combined_text)
        mentioned_labels = [
            label
            for label in labels
            if label and self._normalize_lookup_text(label) in normalized_text
        ]
        subject = "、".join(mentioned_labels) if mentioned_labels else "相關航運股"
        has_red_sea = any(token in combined_text for token in ("紅海", "紅海航線", "繞道", "受阻", "危機"))
        has_scfi = "scfi" in combined_text.lower() or any(token in combined_text for token in ("運價指數", "運價"))
        has_support = any(
            token in combined_text
            for token in ("支撐", "反彈", "走揚", "回升", "有戲", "受惠", "吃補", "撐盤", "上行")
        )
        has_caution = any(
            token in combined_text
            for token in ("觀望", "保守", "降溫", "回落", "短暫", "未必", "壓力")
        )

        parts = [f"{subject} 近期與紅海航線及 SCFI 相關的新聞重點顯示，"]
        if has_red_sea and has_scfi:
            parts.append("市場多把紅海繞道、航線受阻與運力吃緊，視為短線支撐 SCFI 與現貨運價的主要因素。")
        elif has_red_sea:
            parts.append("市場多把紅海航線受阻與繞道成本上升，視為短線運價支撐來源。")
        elif has_scfi:
            parts.append("現有報導多聚焦 SCFI 與現貨運價變化，作為航運股短線評價依據。")

        if has_support and has_caution:
            parts.append("不過報導也提醒，支撐力道能否延續，仍要看 SCFI 是否續彈與塞港因素是否持續。")
        elif has_support:
            parts.append("整體解讀偏向運價具短線支撐。")
        else:
            parts.append("目前較像事件題材支撐，後續仍要由實際運價續航驗證。")
        return "".join(parts)

    def _build_shipping_target_price_excerpt(
        self,
        labels: list[str],
        documents: list[Document],
    ) -> str:
        combined_text = " ".join(f"{item.title} {item.content}" for item in documents)
        normalized_text = self._normalize_lookup_text(combined_text)
        mentioned_labels = [
            label
            for label in labels
            if label and self._normalize_lookup_text(label) in normalized_text
        ]
        subject = "、".join(mentioned_labels) if mentioned_labels else "相關航運股"
        positive_count = sum(
            1
            for item in documents
            if any(token in f"{item.title} {item.content}" for token in ("上修", "調高", "買進", "看好", "受惠", "有戲"))
        )
        negative_count = sum(
            1
            for item in documents
            if any(token in f"{item.title} {item.content}" for token in ("下修", "調降", "保守", "觀望", "中立", "壓力"))
        )

        if positive_count > negative_count:
            direction = "偏上修或偏正向"
        elif negative_count > positive_count:
            direction = "偏保守或偏下修"
        else:
            direction = "正負解讀並存"

        return (
            f"{subject} 近期分析師與法人報導多圍繞目標價、評等與運價續航解讀；"
            f"整體反應{direction}。"
            "若後續 SCFI 續彈或紅海繞道時間拉長，目標價調整通常更容易偏正向；反之則可能回到保守評估。"
        )

    def _build_electricity_cost_excerpt(
        self,
        labels: list[str],
        documents: list[Document],
    ) -> str:
        combined_text = " ".join(f"{item.title} {item.content}" for item in documents)
        normalized_text = self._normalize_lookup_text(combined_text)
        mentioned_labels = [
            label
            for label in labels
            if label and self._normalize_lookup_text(label) in normalized_text
        ]
        subject = "、".join(mentioned_labels) if mentioned_labels else "相關用電大戶"
        has_tariff = any(token in combined_text for token in ("工業電價", "電價", "調漲", "漲價", "電費"))
        has_cost = any(token in combined_text for token in ("成本", "毛利", "壓力", "費用", "獲利"))
        has_amount = any(token in combined_text for token in ("億元", "千萬元", "%", "百分比", "增加額度"))

        parts = [f"{subject} 近期與工業電價調整相關的新聞重點顯示，"]
        if has_tariff and has_cost:
            parts.append("市場普遍把電價調漲視為高耗電產業的成本壓力來源，可能影響毛利與獲利彈性。")
        elif has_tariff:
            parts.append("工業電價調漲仍被視為製造業短線成本變數。")
        if has_amount:
            parts.append("不過目前公開報導對單一公司增加額度的揭露仍不完全一致，需以公司正式說明為準。")
        else:
            parts.append("目前多數報導仍停留在方向性壓力描述，未充分揭露單一公司可精算的成本增加額度。")
        return "".join(parts)

    def _build_electricity_response_excerpt(
        self,
        labels: list[str],
        documents: list[Document],
    ) -> str:
        combined_text = " ".join(f"{item.title} {item.content}" for item in documents)
        normalized_text = self._normalize_lookup_text(combined_text)
        mentioned_labels = [
            label
            for label in labels
            if label and self._normalize_lookup_text(label) in normalized_text
        ]
        subject = "、".join(mentioned_labels) if mentioned_labels else "相關公司"
        has_efficiency = any(token in combined_text for token in ("節能", "節電", "降耗", "效率"))
        has_pass_through = any(token in combined_text for token in ("轉嫁", "調價", "售價", "報價"))
        has_green_power = any(token in combined_text for token in ("綠電", "自發電", "能源管理"))

        measures: list[str] = []
        if has_efficiency:
            measures.append("節能與降耗")
        if has_pass_through:
            measures.append("售價或報價轉嫁")
        if has_green_power:
            measures.append("綠電或自發電配置")

        if measures:
            return f"{subject} 目前較常見的因應方向包括" + "、".join(measures) + "，但實際落地幅度仍要看公司正式說明。"
        return f"{subject} 目前公開資訊對因應方案的揭露仍有限，常見方向多半圍繞節能改善、成本轉嫁與能源配置調整。"

    def _build_macro_yield_sentiment_excerpt(
        self,
        label: str,
        documents: list[Document],
    ) -> str:
        combined_text = " ".join(f"{item.title} {item.content}" for item in documents)
        has_cpi = any(token in combined_text for token in ("CPI", "通膨"))
        has_rate = any(token in combined_text for token in ("利率", "美債", "殖利率", "高殖利率"))
        has_financial_sector = any(token in combined_text for token in ("金控", "金控股", "金融股"))
        has_caution = any(token in combined_text for token in ("保守", "觀望", "壓力", "負面", "降溫"))

        parts = [f"{label} 相關的高殖利率與總經新聞重點顯示，"]
        if has_cpi and has_rate:
            parts.append("美國 CPI 若偏熱，市場通常會把它解讀為利率下修延後或債息上行壓力，進而壓抑高殖利率股情緒。")
        elif has_cpi:
            parts.append("市場多把 CPI 變化視為高殖利率標的情緒的重要變數。")
        if has_financial_sector:
            parts.append("這種情緒通常也會延伸到金控股與其他防禦型高殖利率標的。")
        if has_caution:
            parts.append("整體解讀偏保守或觀望。")
        return "".join(parts)

    def _build_macro_yield_view_excerpt(
        self,
        label: str,
        documents: list[Document],
    ) -> str:
        combined_text = " ".join(f"{item.title} {item.content}" for item in documents)
        positive_count = sum(
            1
            for item in documents
            if any(token in f"{item.title} {item.content}" for token in ("看好", "防禦", "現金流", "穩健", "支撐"))
        )
        negative_count = sum(
            1
            for item in documents
            if any(token in f"{item.title} {item.content}" for token in ("保守", "觀望", "壓力", "不利", "負面"))
        )
        if positive_count > negative_count:
            direction = "偏中性偏正向"
        elif negative_count > positive_count:
            direction = "偏保守"
        else:
            direction = "分歧"

        return (
            f"{label} 相關法人與外資最新觀點多圍繞利率走勢、債息變化與防禦型現金流評價；"
            f"目前整體觀點{direction}。"
        )

    def _serialize_news_tags(self, tags: list[str]) -> str | None:
        unique_tags: list[str] = []
        for tag in tags:
            cleaned = tag.strip()
            if cleaned and cleaned not in unique_tags:
                unique_tags.append(cleaned)
        return "|".join(unique_tags) if unique_tags else None

    def _deserialize_news_tags(self, raw_value) -> list[str]:
        if raw_value in (None, "", "None"):
            return []
        return [item for item in str(raw_value).split("|") if item]

    def _coerce_source_tier(self, raw_value) -> SourceTier:
        try:
            return SourceTier(str(raw_value).lower())
        except Exception:
            return SourceTier.MEDIUM

    def _resolve_news_source_tier(self, article: NewsArticle) -> SourceTier:
        inferred_tier = self._infer_news_tier(article.source_name)
        if article.source_tier == SourceTier.MEDIUM and inferred_tier != SourceTier.MEDIUM:
            return inferred_tier
        return article.source_tier

    def _query_financial_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, statement_date, item_type, value, origin_name FROM financial_statement_items "
                    "WHERE ticker = :ticker AND statement_date BETWEEN :start_date AND :end_date ORDER BY statement_date DESC, item_type"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()

    def _query_balance_sheet_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, statement_date, item_type, value, origin_name FROM balance_sheet_items "
                    "WHERE ticker = :ticker AND statement_date BETWEEN :start_date AND :end_date ORDER BY statement_date DESC"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()

    def _query_cash_flow_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, statement_date, item_type, value, origin_name FROM cash_flow_statement_items "
                    "WHERE ticker = :ticker AND statement_date BETWEEN :start_date AND :end_date ORDER BY statement_date DESC, item_type"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()

    def _query_monthly_revenue_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, revenue_month, revenue, prior_year_month_revenue, month_over_month_pct, year_over_year_pct, cumulative_revenue, prior_year_cumulative_revenue, cumulative_yoy_pct, report_date, notes "
                    "FROM monthly_revenue_points "
                    "WHERE ticker = :ticker AND revenue_month BETWEEN :start_date AND :end_date ORDER BY revenue_month DESC"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()

    def _query_pe_valuation_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, valuation_month, pe_ratio, peer_pe_ratio, pb_ratio, peer_pb_ratio FROM pe_valuation_points "
                    "WHERE ticker = :ticker AND valuation_month BETWEEN :start_date AND :end_date ORDER BY valuation_month DESC"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()

    def _query_dividend_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, base_date, year_label, cash_earnings_distribution, cash_statutory_surplus, stock_earnings_distribution, stock_statutory_surplus, participate_distribution_of_total_shares, announcement_date, announcement_time, cash_ex_dividend_trading_date, cash_dividend_payment_date "
                    "FROM dividend_policies WHERE ticker = :ticker AND base_date BETWEEN :start_date AND :end_date ORDER BY base_date DESC"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()

    def _query_news_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, published_at, title, summary, source_name, url, source_tier, source_type, provider_name, tags "
                    "FROM stock_news_articles "
                    "WHERE ticker = :ticker AND published_at BETWEEN :start_ts AND :end_ts ORDER BY published_at DESC"
                ),
                {
                    "ticker": ticker,
                    "start_ts": datetime.combine(start_date, time.min, tzinfo=timezone.utc),
                    "end_ts": datetime.combine(end_date, time.max, tzinfo=timezone.utc),
                },
            ).mappings().all()

    def _query_margin_rows(self, ticker: str, start_date: date, end_date: date):
        with self._engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT ticker, trading_date, margin_purchase_buy, margin_purchase_cash_repayment, margin_purchase_limit, margin_purchase_sell, margin_purchase_today_balance, margin_purchase_yesterday_balance, offset_loan_and_short, short_sale_buy, short_sale_cash_repayment, short_sale_limit, short_sale_sell, short_sale_today_balance, short_sale_yesterday_balance, note "
                    "FROM margin_purchase_short_sale_bars WHERE ticker = :ticker AND trading_date BETWEEN :start_date AND :end_date ORDER BY trading_date DESC"
                ),
                {"ticker": ticker, "start_date": start_date, "end_date": end_date},
            ).mappings().all()

    def _stock_info_is_fresh(self) -> bool:
        with self._engine.connect() as connection:
            row = connection.execute(text("SELECT MAX(synced_at) AS latest_sync FROM stock_info")).mappings().first()
        latest_sync = row["latest_sync"] if row else None
        if latest_sync is None:
            return False
        return self._coerce_datetime(latest_sync) >= datetime.now(timezone.utc) - timedelta(hours=self._stock_info_refresh_hours)

    def _price_data_needs_refresh(self, ticker: str, end_date: date) -> bool:
        with self._engine.connect() as connection:
            row = connection.execute(
                text("SELECT MAX(trading_date) AS latest_date FROM daily_price_bars WHERE ticker = :ticker"),
                {"ticker": ticker},
            ).mappings().first()
        latest_date = row["latest_date"] if row else None
        if latest_date is None:
            return True
        if hasattr(latest_date, "date"):
            latest_date = latest_date.date()
        return latest_date < end_date - timedelta(days=1)

    def _monthly_revenue_needs_refresh(self, ticker: str) -> bool:
        with self._engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT revenue_month AS latest_month, cumulative_revenue "
                    "FROM monthly_revenue_points WHERE ticker = :ticker "
                    "ORDER BY revenue_month DESC LIMIT 1"
                ),
                {"ticker": ticker},
            ).mappings().first()
        latest_month = row["latest_month"] if row else None
        if latest_month is None:
            return True
        if row.get("cumulative_revenue") is None:
            return True
        if hasattr(latest_month, "date"):
            latest_month = latest_month.date()
        expected_latest = (datetime.now(timezone.utc).date().replace(day=1) - timedelta(days=35)).replace(day=1)
        return latest_month < expected_latest

    def _pe_valuation_needs_refresh(self, ticker: str) -> bool:
        with self._engine.connect() as connection:
            row = connection.execute(
                text("SELECT MAX(valuation_month) AS latest_month FROM pe_valuation_points WHERE ticker = :ticker"),
                {"ticker": ticker},
            ).mappings().first()
        latest_month = row["latest_month"] if row else None
        if latest_month is None:
            return True
        if hasattr(latest_month, "date"):
            latest_month = latest_month.date()
        expected_latest = (datetime.now(timezone.utc).date().replace(day=1) - timedelta(days=35)).replace(day=1)
        return latest_month < expected_latest

    def _normalize_lookup_text(self, value: str) -> str:
        return "".join(value.lower().split())

    def _to_utc_datetime(self, raw_value) -> datetime:
        if isinstance(raw_value, datetime):
            return self._coerce_datetime(raw_value)
        return datetime.combine(raw_value, time.min, tzinfo=timezone.utc)

    def _coerce_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _sorted(
        self,
        documents: list[Document],
        profile: RetrievalProfile | None = None,
    ) -> list[Document]:
        deduped: dict[tuple[str, str], Document] = {}
        for document in documents:
            dedupe_key = (document.url, document.source_type)
            existing = deduped.get(dedupe_key)
            if existing is None or self._sort_key(document, profile) > self._sort_key(
                existing,
                profile,
            ):
                deduped[dedupe_key] = document

        ordered = list(deduped.values())
        ordered.sort(key=lambda item: self._sort_key(item, profile), reverse=True)
        return ordered

    def _sort_key(
        self,
        document: Document,
        profile: RetrievalProfile | None,
    ) -> tuple[int, int, datetime]:
        return (
            self._document_priority(document, profile),
            self._tier_rank(document.source_tier),
            document.published_at,
        )

    def _document_priority(self, document: Document, profile: RetrievalProfile | None) -> int:
        if profile is None:
            return 0
        return profile.sort_priority.get(document.source_type, 0)

    def _tier_rank(self, tier: SourceTier) -> int:
        return {SourceTier.HIGH: 3, SourceTier.MEDIUM: 2, SourceTier.LOW: 1}[tier]

    def _resolve_retrieval_profile(self, query: StructuredQuery) -> RetrievalProfile:
        tags = set(query.topic_tags or [])
        user_query = query.user_query or ""
        intent = query.intent

        if intent == Intent.NEWS_DIGEST:
            if tags & {"航運", "SCFI"}:
                return RETRIEVAL_PROFILES["news_shipping"]
            if tags & {"電價", "成本"}:
                return RETRIEVAL_PROFILES["news_electricity"]
            if tags & {"總經", "CPI", "殖利率"}:
                return RETRIEVAL_PROFILES["news_macro"]
            if tags & {"題材", "產業", "AI", "電動車", "半導體設備"}:
                return RETRIEVAL_PROFILES["news_theme"]
            if tags & {"法說", "指引"}:
                return RETRIEVAL_PROFILES["news_guidance"]
            if "上市" in tags:
                return RETRIEVAL_PROFILES["news_listing"]
            if query.topic == Topic.EARNINGS:
                return RETRIEVAL_PROFILES["earnings_fundamental"]
            if query.topic == Topic.ANNOUNCEMENT:
                return RETRIEVAL_PROFILES["investment_announcement"]
            return RETRIEVAL_PROFILES["news_generic"]

        if intent == Intent.EARNINGS_REVIEW:
            if "月營收" in tags:
                return RETRIEVAL_PROFILES["earnings_monthly_revenue"]
            if "毛利率" in tags and ("轉正" in tags or "轉正" in user_query):
                return RETRIEVAL_PROFILES["earnings_margin_turnaround"]
            if tags & {"EPS", "股利"}:
                return RETRIEVAL_PROFILES["earnings_eps_dividend"]
            return RETRIEVAL_PROFILES["earnings_fundamental"]

        if intent == Intent.VALUATION_CHECK:
            if "股價區間" in tags:
                return RETRIEVAL_PROFILES["valuation_price_range"]
            if tags & {"股價", "展望"}:
                return RETRIEVAL_PROFILES["valuation_price_outlook"]
            if "基本面" in tags and "本益比" in tags:
                return RETRIEVAL_PROFILES["valuation_fundamental"]
            return RETRIEVAL_PROFILES["valuation_pe_only"]

        if intent == Intent.DIVIDEND_ANALYSIS:
            if tags & {"除息", "填息"}:
                return RETRIEVAL_PROFILES["dividend_ex"]
            if "現金流" in tags:
                return RETRIEVAL_PROFILES["dividend_fcf"]
            if "負債" in tags:
                return RETRIEVAL_PROFILES["dividend_debt"]
            return RETRIEVAL_PROFILES["dividend_yield"]

        if intent == Intent.FINANCIAL_HEALTH:
            if "毛利率" in tags and query.comparison_ticker:
                return RETRIEVAL_PROFILES["health_gross_margin_cmp"]
            if tags & {"獲利", "穩定性"}:
                return RETRIEVAL_PROFILES["health_profitability"]
            return RETRIEVAL_PROFILES["health_revenue_growth"]

        if intent == Intent.TECHNICAL_VIEW:
            if tags & {"季線", "籌碼"}:
                return RETRIEVAL_PROFILES["technical_margin_flow"]
            return RETRIEVAL_PROFILES["technical_indicators"]

        if intent == Intent.INVESTMENT_ASSESSMENT:
            if "公告" in tags or query.topic == Topic.ANNOUNCEMENT:
                return RETRIEVAL_PROFILES["investment_announcement"]
            if "基本面" in tags and "本益比" in tags:
                return RETRIEVAL_PROFILES["investment_support"]
            if "風險" in tags:
                return RETRIEVAL_PROFILES["investment_risk"]
            return RETRIEVAL_PROFILES["investment_support"]

        return RETRIEVAL_PROFILES["news_generic"]

    def _calculate_rsi(self, closes: list[float], period: int = 14) -> float | None:
        if len(closes) < period + 1:
            return None
        deltas = [current - previous for previous, current in zip(closes, closes[1:])]
        gains = [max(delta, 0.0) for delta in deltas]
        losses = [max(-delta, 0.0) for delta in deltas]
        average_gain = sum(gains[:period]) / period
        average_loss = sum(losses[:period]) / period

        for index in range(period, len(deltas)):
            average_gain = ((average_gain * (period - 1)) + gains[index]) / period
            average_loss = ((average_loss * (period - 1)) + losses[index]) / period

        if average_loss == 0:
            return 100.0
        relative_strength = average_gain / average_loss
        return 100 - (100 / (1 + relative_strength))

    def _calculate_kd(self, bars: list[PriceBar], period: int = 9) -> tuple[float, float] | None:
        if len(bars) < period:
            return None

        k_value = 50.0
        d_value = 50.0
        for index in range(period - 1, len(bars)):
            window = bars[index - period + 1 : index + 1]
            lowest_low = min(item.low_price for item in window)
            highest_high = max(item.high_price for item in window)
            if highest_high == lowest_low:
                rsv = 50.0
            else:
                rsv = ((bars[index].close_price - lowest_low) / (highest_high - lowest_low)) * 100
            k_value = (2 / 3) * k_value + (1 / 3) * rsv
            d_value = (2 / 3) * d_value + (1 / 3) * k_value
        return k_value, d_value

    def _calculate_simple_moving_average(self, values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        window = values[-period:]
        return sum(window) / period

    def _calculate_macd(
        self,
        closes: list[float],
        short_period: int = 12,
        long_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[float, float, float] | None:
        short_series = self._calculate_ema_series(closes, short_period)
        long_series = self._calculate_ema_series(closes, long_period)
        if not short_series or not long_series:
            return None

        macd_series: list[float] = []
        for index in range(long_period - 1, len(closes)):
            short_index = index - (short_period - 1)
            long_index = index - (long_period - 1)
            macd_series.append(short_series[short_index] - long_series[long_index])

        signal_series = self._calculate_ema_series(macd_series, signal_period)
        if not signal_series:
            return None

        macd_line = macd_series[-1]
        signal_line = signal_series[-1]
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_ema_series(self, values: list[float], period: int) -> list[float]:
        if len(values) < period:
            return []

        multiplier = 2 / (period + 1)
        ema_value = sum(values[:period]) / period
        series = [ema_value]
        for value in values[period:]:
            ema_value = ((value - ema_value) * multiplier) + ema_value
            series.append(ema_value)
        return series

    def _calculate_bollinger_bands(
        self,
        closes: list[float],
        period: int = 20,
        standard_deviations: float = 2.0,
    ) -> tuple[float, float, float] | None:
        if len(closes) < period:
            return None

        window = closes[-period:]
        middle_band = sum(window) / period
        variance = sum((value - middle_band) ** 2 for value in window) / period
        standard_deviation = variance ** 0.5
        upper_band = middle_band + (standard_deviation * standard_deviations)
        lower_band = middle_band - (standard_deviation * standard_deviations)
        return upper_band, middle_band, lower_band

    def _calculate_moving_average_bias(
        self,
        closes: list[float],
        short_period: int = 5,
        long_period: int = 20,
    ) -> tuple[float, float, float, float] | None:
        if len(closes) < long_period:
            return None

        latest_close = closes[-1]
        ma5_value = sum(closes[-short_period:]) / short_period
        ma20_value = sum(closes[-long_period:]) / long_period
        ma5_bias = ((latest_close - ma5_value) / ma5_value) * 100 if ma5_value else 0.0
        ma20_bias = ((latest_close - ma20_value) / ma20_value) * 100 if ma20_value else 0.0
        return ma5_value, ma20_value, ma5_bias, ma20_bias

    def _detect_cumulative_month_count(self, query_text: str, today: date) -> int:
        normalized = self._normalize_lookup_text(query_text)
        if "前三個月" in query_text or "前3個月" in normalized or "前三月" in query_text:
            return 3
        match = re.search(r"前\s*(\d{1,2})\s*個?月", query_text)
        if match:
            return max(1, min(int(match.group(1)), 12))
        if "今年" in query_text:
            return min(today.month - 1 if today.day == 1 else today.month, 12)
        return 3

    def _detect_requested_revenue_month(self, query_text: str) -> int | None:
        match = re.search(r"(?<!\d)(1[0-2]|0?[1-9])\s*月", query_text)
        if not match:
            return None
        return int(match.group(1))

    def _detect_requested_revenue_year(self, query_text: str, fallback_year: int) -> int:
        western_match = re.search(r"(?<!\d)(20\d{2})\s*年", query_text)
        if western_match:
            return int(western_match.group(1))

        roc_match = re.search(r"(?<!\d)(1\d{2})\s*年", query_text)
        if roc_match:
            return int(roc_match.group(1)) + 1911

        if "去年" in query_text:
            return fallback_year - 1
        return fallback_year

    def _build_latest_monthly_revenue_excerpt(self, label: str, point: MonthlyRevenuePoint) -> str:
        revenue_100m = self._to_hundred_million_revenue(point.revenue)
        parts = [
            f"{label} 目前最新已公布月營收為 {point.revenue_month:%Y-%m}，單月營收約 {revenue_100m:.2f} 億元。"
        ]
        if point.month_over_month_pct is not None:
            parts.append(f"月增率約 {point.month_over_month_pct:.2f}%。")
        if point.year_over_year_pct is not None:
            parts.append(f"年增率約 {point.year_over_year_pct:.2f}%。")
        if point.notes:
            parts.append(f"公司備註：{point.notes}。")
        return "".join(parts)

    def _describe_revenue_signal_strength(self, point: MonthlyRevenuePoint) -> str:
        yoy_pct = point.year_over_year_pct
        mom_pct = point.month_over_month_pct
        if yoy_pct is not None and yoy_pct >= 30:
            return "若以年增率觀察，這屬偏強的營收成長訊號。"
        if mom_pct is not None and mom_pct >= 20:
            return "若以月增率觀察，這屬短線偏強的營收增長訊號。"
        if yoy_pct is not None and yoy_pct >= 15:
            return "若以年增率觀察，營收表現屬溫和偏強。"
        return "現有月營收數據未顯示特別極端的單月爆發成長。"

    def _classify_guidance_document(self, document: Document) -> str:
        haystack = f"{document.title} {document.content}".lower()
        positive_keywords = (
            "上修",
            "調高",
            "看好",
            "優於預期",
            "正面",
            "樂觀",
            "成長",
            "增溫",
            "回升",
            "買進",
            "調升",
            "有撐",
        )
        negative_keywords = (
            "下修",
            "調降",
            "保守",
            "不如預期",
            "負面",
            "隱憂",
            "逆風",
            "壓力",
            "疲弱",
            "風險",
            "砍單",
            "降評",
            "下滑",
        )
        positive_score = sum(1 for keyword in positive_keywords if keyword in haystack)
        negative_score = sum(1 for keyword in negative_keywords if keyword in haystack)
        if positive_score > negative_score and positive_score > 0:
            return "positive"
        if negative_score > positive_score and negative_score > 0:
            return "negative"
        return "neutral"

    def _build_guidance_reaction_excerpt(
        self,
        label: str,
        items: list[Document],
        sentiment: str,
    ) -> str:
        sentiment_text = "正面" if sentiment == "positive" else "負面"
        lead_titles = "；".join(item.title for item in items[:3])
        return (
            f"{label} 法說後與下半年營運指引相關的{sentiment_text}解讀共有 {len(items)} 則中高可信來源。"
            f"主要觀點包括：{lead_titles}。"
        )

    def _find_previous_month_revenue_point(
        self,
        revenues: list[MonthlyRevenuePoint],
        current_point: MonthlyRevenuePoint,
    ) -> MonthlyRevenuePoint | None:
        ordered = sorted(revenues, key=lambda item: item.revenue_month)
        for index, item in enumerate(ordered):
            if item.revenue_month == current_point.revenue_month and index > 0:
                return ordered[index - 1]
        return None

    def _describe_monthly_revenue_high_status(
        self,
        revenues: list[MonthlyRevenuePoint],
        target_point: MonthlyRevenuePoint,
    ) -> str:
        lower_bound = target_point.revenue_month - timedelta(days=365)
        candidates = [
            item
            for item in revenues
            if lower_bound <= item.revenue_month <= target_point.revenue_month
        ]
        if len(candidates) < 6:
            return (
                f"近一年月營收歷史不足，目前僅能確認 {target_point.revenue_month:%Y-%m} 已出表，"
                "暫時無法穩定判定是否創下近一年新高。"
            )

        highest_point = max(candidates, key=lambda item: item.revenue)
        if highest_point.revenue_month == target_point.revenue_month:
            return (
                f"{target_point.revenue_month:%Y-%m} 單月營收在目前本地可比的近一年資料中，"
                "創下近一年新高。"
            )

        highest_100m = self._to_hundred_million_revenue(highest_point.revenue)
        return (
            f"{target_point.revenue_month:%Y-%m} 單月營收尚未創下近一年新高；"
            f"近一年高點仍為 {highest_point.revenue_month:%Y-%m}，約 {highest_100m:.2f} 億元。"
        )

    def _describe_monthly_revenue_market_view(
        self,
        news: list[NewsArticle],
        target_point: MonthlyRevenuePoint,
    ) -> str | None:
        if not news:
            return None

        filtered_news = [
            article
            for article in news
            if self._keyword_match_score(
                article,
                ("\u71df\u6536", "\u6708\u71df\u6536", "\u5e74\u589e", "\u6708\u589e", "\u51fa\u8ca8", "\u8a02\u55ae", "\u9700\u6c42"),
            )
            > 0
        ]
        if not filtered_news:
            return None

        latest_article = max(filtered_news, key=lambda item: item.published_at)
        summary = (latest_article.summary or "").strip()
        summary_segment = summary[:120] if summary else latest_article.title
        return (
            f"市場解讀：近期相關報導多聚焦在 {target_point.revenue_month:%Y-%m} 月營收變化"
            f"背後的出貨、訂單或需求能見度；最新可比新聞提到「{summary_segment}」。"
        )

    def _format_monthly_revenue_pct_sentence(self, label: str, value: float | None) -> str:
        if value is None:
            return ""
        return f"{label}約 {value:.2f}%。"

    def _calculate_percentile(self, values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        lower_value = ordered[lower_index]
        upper_value = ordered[upper_index]
        weight = position - lower_index
        return lower_value + (upper_value - lower_value) * weight

    def _calculate_percentile_rank(self, values: list[float], current_value: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        lower_or_equal = sum(1 for value in ordered if value <= current_value)
        return lower_or_equal / len(ordered)

    def _to_hundred_million_revenue(self, raw_value: float) -> float:
        if raw_value >= 1_000_000:
            return raw_value / 100_000
        if raw_value >= 10_000:
            return raw_value / 100
        return raw_value

    def _describe_pe_valuation_zone(self, percentile_rank: float) -> str:
        if percentile_rank >= 0.8:
            return "歷史偏高區"
        if percentile_rank <= 0.2:
            return "歷史偏低區"
        return "歷史中段區"

    def _describe_pe_entry_view(self, percentile_rank: float) -> str:
        if percentile_rank >= 0.85:
            return "現在進場的估值不算便宜，需接受評價偏高的風險"
        if percentile_rank >= 0.65:
            return "現在進場大致屬中性偏貴，較適合分批或等待更佳價格區間"
        if percentile_rank <= 0.2:
            return "現在進場的估值相對偏低，歷史比較上不算買貴"
        return "現在進場的估值大致落在合理區間，是否買貴仍要搭配獲利成長一起看"

    def _technical_overbought_status(self, rsi_value: float, k_value: float, d_value: float) -> str:
        if rsi_value >= 70 and max(k_value, d_value) >= 80:
            return "已進入超買區"
        if rsi_value >= 65 or max(k_value, d_value) >= 75:
            return "疑似進入超買區"
        return "尚未進入超買區"

    def _describe_macd_trend(self, macd_line: float, signal_line: float, histogram: float) -> str:
        if macd_line >= signal_line and histogram > 0:
            return "偏多"
        if macd_line >= signal_line:
            return "轉強"
        if histogram < 0 and macd_line < signal_line:
            return "偏空"
        return "轉弱"

    def _describe_bollinger_position(
        self,
        close_price: float,
        upper_band: float,
        middle_band: float,
        lower_band: float,
    ) -> str:
        band_range = upper_band - lower_band
        if band_range <= 0:
            return "位於中軌附近"

        normalized_position = (close_price - lower_band) / band_range
        if normalized_position >= 0.8:
            return "接近布林上軌"
        if normalized_position <= 0.2:
            return "接近布林下軌"
        if close_price >= middle_band:
            return "位於布林中軌偏上"
        return "位於布林中軌偏下"

    def _describe_bias_status(self, ma5_bias: float, ma20_bias: float) -> str:
        if ma5_bias >= 5 or ma20_bias >= 8:
            return "明顯偏熱"
        if ma5_bias >= 2 or ma20_bias >= 4:
            return "溫和偏熱"
        if ma5_bias <= -5 or ma20_bias <= -8:
            return "明顯偏弱"
        if ma5_bias <= -2 or ma20_bias <= -4:
            return "溫和偏弱"
        return "中性"

    def _describe_season_line_status(
        self,
        latest_close: float,
        season_line: float,
        previous_close: float,
        previous_season_line: float,
    ) -> str:
        if latest_close < season_line and previous_close >= previous_season_line:
            return "近期跌破季線"
        if latest_close < season_line:
            return "仍在季線下方"
        if latest_close >= season_line and previous_close < previous_season_line:
            return "重新站回季線"
        return "尚未跌破季線"

    def _describe_margin_balance_status(
        self,
        utilization_pct: float | None,
        delta_vs_average_pct: float | None,
    ) -> str:
        score = 0
        if utilization_pct is not None:
            if utilization_pct >= 35:
                score += 2
            elif utilization_pct >= 20:
                score += 1
        if delta_vs_average_pct is not None:
            if delta_vs_average_pct >= 20:
                score += 2
            elif delta_vs_average_pct >= 10:
                score += 1
            elif delta_vs_average_pct <= -15:
                score -= 1

        if score >= 3:
            return "偏高"
        if score >= 1:
            return "中性偏高"
        if score <= -1:
            return "中性偏低"
        return "中性"

    def _extract_annual_profitability_metrics(
        self,
        items: list[FinancialStatementItem],
    ) -> dict[int, dict[str, float | datetime]]:
        grouped: dict[int, list[FinancialStatementItem]] = defaultdict(list)
        for item in items:
            if item.statement_date.month == 12 and item.statement_date.day == 31:
                grouped[item.statement_date.year].append(item)

        annual_metrics: dict[int, dict[str, float | datetime]] = {}
        for year, year_items in grouped.items():
            revenue = self._find_statement_value_exact(
                year_items,
                ("revenue", "operatingrevenue", "salesrevenue", "營業收入", "營業收入合計"),
            )
            gross_profit = self._find_statement_value_exact(
                year_items,
                ("grossprofit", "grossprofitlossfromoperations", "營業毛利", "營業毛利（毛損）", "營業毛利(毛損)"),
            )
            operating_income = self._find_statement_value_exact(
                year_items,
                ("operatingincome", "operatingprofitloss", "營業利益（損失）", "營業利益", "營業損失"),
            )
            pretax_income = self._find_statement_value_exact(
                year_items,
                ("pretaxincome", "incomebeforetax", "稅前淨利（淨損）", "稅前淨利", "稅前淨損"),
            )
            income_after_taxes = self._find_statement_value_exact(
                year_items,
                ("incomeaftertaxes", "profitloss", "本期淨利（淨損）", "本期淨利", "本期淨損"),
            )
            parent_net_income = self._find_statement_value_exact(
                year_items,
                (
                    "equityattributabletoownersofparent",
                    "profitlossattributabletoownersofparent",
                    "淨利（淨損）歸屬於母公司業主",
                ),
            )
            non_operating_income = self._find_statement_value_exact(
                year_items,
                ("totalnonoperatingincomeandexpense", "營業外收入及支出"),
            )
            eps_item = next((item for item in year_items if self._is_eps_item(item)), None)
            if parent_net_income is None and income_after_taxes is not None:
                parent_net_income = income_after_taxes
            if eps_item is None or parent_net_income is None:
                continue
            annual_metrics[year] = {
                "statement_date": max(item.statement_date for item in year_items),
                "eps": eps_item.value,
                "revenue": revenue or 0.0,
                "revenue_100m": (revenue or 0.0) / 100_000_000,
                "gross_profit": gross_profit or 0.0,
                "gross_profit_100m": (gross_profit or 0.0) / 100_000_000,
                "operating_income": operating_income or 0.0,
                "operating_income_100m": (operating_income or 0.0) / 100_000_000,
                "pretax_income": pretax_income or 0.0,
                "pretax_income_100m": (pretax_income or 0.0) / 100_000_000,
                "income_after_taxes": income_after_taxes or 0.0,
                "income_after_taxes_100m": (income_after_taxes or 0.0) / 100_000_000,
                "parent_net_income": parent_net_income,
                "parent_net_income_100m": parent_net_income / 100_000_000,
                "non_operating_income": non_operating_income or 0.0,
                "non_operating_income_100m": (non_operating_income or 0.0) / 100_000_000,
            }
        return annual_metrics

    def _describe_profitability_stability(
        self,
        years: list[int],
        annual_metrics: dict[int, dict[str, float | datetime]],
    ) -> str:
        eps_values = [float(annual_metrics[year]["eps"]) for year in years if annual_metrics[year].get("eps") is not None]
        loss_years = [year for year in years if float(annual_metrics[year]["parent_net_income"]) <= 0]
        if loss_years:
            return "近五年並非每年都有穩定獲利"
        if not eps_values:
            return "近五年獲利資料仍不足以判斷是否穩定"
        min_eps = min(eps_values)
        max_eps = max(eps_values)
        if max_eps <= 0:
            return "近五年獲利資料偏弱，未能呈現穩定正向獲利"
        stability_ratio = min_eps / max_eps
        if stability_ratio >= 0.7:
            return "近五年都有獲利，且獲利波動相對可控"
        if stability_ratio >= 0.4:
            return "近五年雖維持獲利，但獲利波動不算小"
        return "近五年雖未必每年轉虧，但獲利波動明顯，未必能視為穩定獲利"

    def _build_loss_reason_excerpt(
        self,
        label: str,
        loss_year: int,
        loss_metrics: dict[str, float | datetime],
        previous_metrics: dict[str, float | datetime] | None,
    ) -> str:
        revenue = float(loss_metrics.get("revenue") or 0.0)
        operating_income = float(loss_metrics.get("operating_income") or 0.0)
        non_operating_income = float(loss_metrics.get("non_operating_income") or 0.0)
        net_income = float(loss_metrics.get("parent_net_income") or 0.0)
        eps_value = float(loss_metrics.get("eps") or 0.0)

        reason_parts: list[str] = [
            f"{label} 在 {loss_year} 年歸屬母公司淨損約 {abs(net_income) / 100_000_000:.2f} 億元，EPS 約 {eps_value:.2f} 元。"
        ]
        if previous_metrics is not None:
            previous_revenue = float(previous_metrics.get("revenue") or 0.0)
            if previous_revenue > 0 and revenue > 0:
                revenue_yoy = ((revenue - previous_revenue) / previous_revenue) * 100
                reason_parts.append(f"若與前一年相比，營收年增率約 {revenue_yoy:.2f}%。")

        if operating_income < 0 and non_operating_income <= 0:
            reason_parts.append(
                f"若只看財報結構推估，主因較像本業轉弱，營業利益已轉為損失約 {abs(operating_income) / 100_000_000:.2f} 億元，"
                f"且營業外也未提供明顯支撐。"
            )
        elif operating_income < 0 and non_operating_income > 0:
            reason_parts.append(
                f"若只看財報結構推估，主因仍偏向本業虧損，營業利益為損失約 {abs(operating_income) / 100_000_000:.2f} 億元；"
                f"雖有營業外收益約 {non_operating_income / 100_000_000:.2f} 億元，仍不足以把全年拉回獲利。"
            )
        elif operating_income >= 0 and non_operating_income < 0:
            reason_parts.append(
                f"若只看財報結構推估，本業仍有一定支撐，但營業外損失約 {abs(non_operating_income) / 100_000_000:.2f} 億元，"
                "是拖累全年獲利的重要因素。"
            )
        else:
            reason_parts.append("若只看財報結構推估，當年度獲利惡化應是多項因素同時作用，仍需搭配公司說明與法說內容解讀。")

        return "".join(reason_parts)

    def _extract_gross_margin_snapshots(
        self,
        items: list[FinancialStatementItem],
    ) -> list[dict[str, float | datetime]]:
        grouped: dict[datetime, list[FinancialStatementItem]] = defaultdict(list)
        for item in items:
            if item.item_type.endswith("_per"):
                continue
            grouped[item.statement_date].append(item)

        snapshots: list[dict[str, float | datetime]] = []
        for statement_date in sorted(grouped):
            row_items = grouped[statement_date]
            revenue = self._find_statement_value_exact(
                row_items,
                (
                    "revenue",
                    "operatingrevenue",
                    "salesrevenue",
                    "營業收入",
                    "營業收入合計",
                ),
            )
            gross_profit = self._find_statement_value_exact(
                row_items,
                (
                    "grossprofit",
                    "grossprofitlossfromoperations",
                    "營業毛利",
                    "營業毛利（毛損）",
                    "營業毛利(毛損)",
                ),
            )
            operating_income = self._find_statement_value_exact(
                row_items,
                (
                    "operatingincome",
                    "operatingprofitloss",
                    "營業利益（損失）",
                    "營業利益",
                    "營業損失",
                ),
            )
            if revenue is None or gross_profit is None or revenue == 0:
                continue
            snapshots.append(
                {
                    "statement_date": statement_date,
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "operating_income": operating_income or 0.0,
                    "revenue_100m": revenue / 100_000_000,
                    "gross_profit_100m": gross_profit / 100_000_000,
                    "operating_income_100m": (operating_income or 0.0) / 100_000_000,
                    "gross_margin_pct": (gross_profit / revenue) * 100,
                    "operating_margin_pct": ((operating_income or 0.0) / revenue) * 100,
                }
            )
        return snapshots

    def _describe_gross_margin_efficiency_view(self, margin_gap: float) -> str:
        if margin_gap >= 5:
            return "相對占優勢"
        if margin_gap >= 2:
            return "略占優勢"
        return "差距有限"

    def _describe_gross_margin_turnaround_status(self, previous_margin: float, latest_margin: float) -> str:
        if previous_margin < 0 <= latest_margin:
            return "毛利率已由負轉正。"
        if previous_margin >= 0 and latest_margin >= 0:
            return "毛利率未出現由負轉正，仍維持正值。"
        if previous_margin < 0 and latest_margin < 0:
            return "毛利率仍為負值，尚未轉正。"
        return "毛利率由正轉負。"

    def _describe_operating_turnaround_status(
        self,
        previous_operating_income: float,
        latest_operating_income: float,
    ) -> str:
        if previous_operating_income < 0 <= latest_operating_income:
            return "營業利益已同步轉正。"
        if previous_operating_income >= 0 and latest_operating_income >= 0:
            return "營業利益維持正值。"
        if latest_operating_income < 0 <= previous_operating_income:
            return "營業利益由正轉負。"
        return "營業利益尚未同步轉正。"

    def _describe_profitability_improvement_view(
        self,
        previous_margin: float,
        latest_margin: float,
        previous_operating_income: float,
        latest_operating_income: float,
    ) -> str:
        margin_turned_positive = previous_margin < 0 <= latest_margin
        operating_turned_positive = previous_operating_income < 0 <= latest_operating_income
        if margin_turned_positive and operating_turned_positive:
            return "最新季毛利率已由負轉正，且營業利益也同步轉正，目前可視為本業層面的實質獲利改善。"
        if margin_turned_positive and latest_operating_income < 0:
            return "最新季毛利率雖已由負轉正，但營業利益仍未同步轉正，較像本業修復初期，暫時還不能直接視為實質獲利改善。"
        if latest_margin >= 0 and latest_operating_income < 0:
            return "最新季毛利率仍維持正值，但營業利益依舊為負，本業獲利修復仍不完整，目前尚難視為實質獲利改善。"
        if latest_margin < 0 and latest_operating_income >= 0:
            return "雖然營業利益已轉正，但毛利率仍未回到正值，需留意是否有一次性因素扭曲本業判讀。"
        return "毛利率與營業利益都尚未同時站穩正值，目前仍難視為實質獲利改善。"

    def _extract_balance_sheet_snapshots(
        self,
        items: list[FinancialStatementItem],
    ) -> list[dict[str, float | datetime | None]]:
        grouped: dict[datetime, list[FinancialStatementItem]] = defaultdict(list)
        for item in items:
            if item.item_type.endswith("_per"):
                continue
            grouped[item.statement_date].append(item)

        snapshots: list[dict[str, float | datetime | None]] = []
        for statement_date in sorted(grouped):
            row_items = grouped[statement_date]
            total_assets = self._find_statement_value_exact(
                row_items,
                ("totalassets", "assetstotal", "資產總額", "資產總計"),
            )
            total_liabilities = self._find_statement_value_exact(
                row_items,
                ("liabilities", "totalliabilities", "負債總額", "負債總計"),
            )
            cash_and_equivalents = self._find_statement_value_exact(
                row_items,
                ("cashandcashequivalents", "現金及約當現金"),
            )
            if total_assets is None or total_liabilities is None:
                continue
            snapshots.append(
                {
                    "statement_date": statement_date,
                    "assets": total_assets,
                    "liabilities": total_liabilities,
                    "cash": cash_and_equivalents,
                    "assets_100m": total_assets / 100_000_000,
                    "liabilities_100m": total_liabilities / 100_000_000,
                    "cash_100m": (cash_and_equivalents or 0.0) / 100_000_000,
                    "debt_ratio": (total_liabilities / total_assets) * 100 if total_assets else None,
                }
            )
        return snapshots

    def _find_year_ago_snapshot(
        self,
        snapshots: list[dict[str, float | datetime | None]],
        latest_statement_date: datetime,
    ) -> dict[str, float | datetime | None] | None:
        for snapshot in reversed(snapshots[:-1]):
            statement_date = snapshot["statement_date"]
            if not isinstance(statement_date, datetime):
                continue
            if statement_date.month == latest_statement_date.month and statement_date.year == latest_statement_date.year - 1:
                return snapshot
        return None

    def _describe_debt_ratio_status(
        self,
        latest_ratio: float | None,
        previous_ratio: float | None,
        year_ago_ratio: float | None,
    ) -> str:
        if latest_ratio is None:
            return "目前資料不足"

        quarter_delta = (latest_ratio - previous_ratio) if previous_ratio is not None else 0.0
        year_delta = (latest_ratio - year_ago_ratio) if year_ago_ratio is not None else 0.0

        if quarter_delta >= 4 or year_delta >= 5:
            return "有明顯升高"
        if quarter_delta >= 2 or year_delta >= 3:
            return "有溫和升高"
        if quarter_delta <= -2 and year_delta <= 1:
            return "未見突然升高，反而較前幾期回落"
        if abs(quarter_delta) <= 1.5 and abs(year_delta) <= 2:
            return "未見突然升高，大致持平"
        return "近期沒有明顯異常升高"

    def _describe_cash_dividend_coverage(self, coverage_ratio: float) -> str:
        if coverage_ratio >= 3:
            return "若只看帳上現金，現金部位看起來足以支應現金股利"
        if coverage_ratio >= 1.5:
            return "若只看帳上現金，現金部位大致可支應現金股利，但仍要留意後續營運與資本支出"
        if coverage_ratio >= 1.0:
            return "若只看帳上現金，現金部位勉強可支應現金股利，但緩衝不算特別厚"
        return "若只看帳上現金，支應現金股利的緩衝偏薄"

    def _extract_annual_fcf_metrics(
        self,
        items: list[FinancialStatementItem],
    ) -> dict[int, dict[str, float | datetime]]:
        grouped: dict[int, list[FinancialStatementItem]] = defaultdict(list)
        for item in items:
            if item.statement_date.month == 12 and item.statement_date.day == 31:
                grouped[item.statement_date.year].append(item)

        metrics: dict[int, dict[str, float | datetime]] = {}
        for year, year_items in grouped.items():
            operating_cash_flow = self._find_statement_value(
                year_items,
                (
                    "netcashinflowfromoperatingactivities",
                    "cashflowsfromoperatingactivities",
                    "營業活動之淨現金流入",
                    "營業活動之淨現金流入（流出）",
                ),
            )
            capex_raw = self._find_statement_value(
                year_items,
                (
                    "propertyandplantandequipment",
                    "取得不動產、廠房及設備",
                ),
            )
            if operating_cash_flow is None or capex_raw is None:
                continue
            capex = abs(capex_raw)
            fcf_value = operating_cash_flow - capex
            published_at = max(item.statement_date for item in year_items)
            metrics[year] = {
                "operating_cash_flow": operating_cash_flow,
                "capex": capex,
                "fcf": fcf_value,
                "operating_cash_flow_100m": operating_cash_flow / 100_000_000,
                "capex_100m": capex / 100_000_000,
                "fcf_100m": fcf_value / 100_000_000,
                "published_at": published_at,
            }
        return metrics

    def _extract_annual_dividend_totals(
        self,
        dividends: list[DividendPolicy],
    ) -> dict[int, dict[str, float | datetime]]:
        annual: dict[int, dict[str, float | datetime]] = {}
        for policy in sorted(dividends, key=self._dividend_published_at, reverse=True):
            year = self._normalize_policy_year(policy)
            if year is None or year in annual:
                continue
            cash_dividend_per_share = self._cash_dividend_amount(policy)
            shares = policy.participate_distribution_of_total_shares
            total_payout = cash_dividend_per_share * shares if shares is not None else None
            annual[year] = {
                "cash_dividend_per_share": cash_dividend_per_share,
                "shares": shares,
                "total_payout": total_payout,
                "total_payout_100m": (total_payout / 100_000_000) if total_payout is not None else None,
                "published_at": self._dividend_published_at(policy),
            }
        return annual

    def _normalize_policy_year(self, policy: DividendPolicy) -> int | None:
        digits = "".join(char for char in policy.year_label if char.isdigit())
        if digits:
            year = int(digits)
            return year + 1911 if year < 1911 else year
        return policy.date.year if policy.date.year >= 2000 else None

    def _describe_dividend_sustainability(self, coverage_ratios: list[float]) -> str:
        if not coverage_ratios:
            return "目前資料不足，無法確認股利政策的永續性"
        minimum_ratio = min(coverage_ratios)
        average_ratio = sum(coverage_ratios) / len(coverage_ratios)
        if minimum_ratio >= 1.2:
            return "近三年自由現金流均高於現金股利支出，顯示目前股利政策具一定永續性"
        if average_ratio >= 1.0 and sum(ratio >= 1.0 for ratio in coverage_ratios) >= max(len(coverage_ratios) - 1, 1):
            return "近三年大致能以自由現金流支應現金股利，整體永續性偏穩健"
        if average_ratio >= 0.8:
            return "自由現金流對股利支應能力接近打平，後續仍需留意資本支出與獲利變化"
        return "自由現金流對現金股利支應能力偏弱，股利政策永續性需保守看待"

    def _find_statement_value(
        self,
        items: list[FinancialStatementItem],
        candidates: tuple[str, ...],
    ) -> float | None:
        for item in items:
            if self._statement_item_matches(item, candidates):
                return item.value
        return None

    def _find_statement_value_exact(
        self,
        items: list[FinancialStatementItem],
        candidates: tuple[str, ...],
    ) -> float | None:
        normalized_candidates = {token.replace(" ", "").lower() for token in candidates}
        for item in items:
            item_type = item.item_type.replace(" ", "").lower()
            origin_name = item.origin_name.replace(" ", "").lower()
            if item_type in normalized_candidates or origin_name in normalized_candidates:
                return item.value
        return self._find_statement_value(items, candidates)

    def _statement_item_matches(
        self,
        item: FinancialStatementItem,
        candidates: tuple[str, ...],
    ) -> bool:
        item_type = item.item_type.replace(" ", "").lower()
        origin_name = item.origin_name.replace(" ", "").lower()
        return any(token.replace(" ", "").lower() in item_type or token.replace(" ", "").lower() in origin_name for token in candidates)

    def _cash_dividend_amount(self, policy: DividendPolicy) -> float:
        return (policy.cash_earnings_distribution or 0.0) + (policy.cash_statutory_surplus or 0.0)

    def _stock_dividend_amount(self, policy: DividendPolicy) -> float:
        return (policy.stock_earnings_distribution or 0.0) + (policy.stock_statutory_surplus or 0.0)

    def _is_eps_item(self, item: FinancialStatementItem) -> bool:
        return self._statement_item_matches(
            item,
            (
                "eps",
                "基本每股盈餘",
                "每股盈餘",
                "基本每股盈餘（元）",
                "基本每股盈餘(元)",
                "稀釋每股盈餘",
            ),
        )

    def _dividend_published_at(self, policy: DividendPolicy) -> datetime:
        return policy.announcement_date or policy.date

    def _build_finmind_url(self, dataset: str, ticker: str, start_date: date, end_date: date) -> str:
        return (
            "https://api.finmindtrade.com/api/v4/data"
            f"?dataset={dataset}&data_id={ticker}&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
        )

    def _infer_news_tier(self, source_name: str) -> SourceTier:
        lowered = source_name.lower()
        if any(
            token in lowered
            for token in (
                "cmoney",
                "同學會",
                "ptt",
                "dcard",
                "mobile01",
                "facebook",
                "x.com",
                "forum",
                "爆料",
            )
        ):
            return SourceTier.LOW
        if any(token in lowered for token in ("mops", "twse", "tpex", "company", "corp", "official")):
            return SourceTier.HIGH
        if any(token in source_name for token in ("公開資訊觀測站", "證交所", "公司投資人關係", "鉅亨", "豐雲")):
            return SourceTier.HIGH
        return SourceTier.MEDIUM

    def _infer_news_topics(self, article: NewsArticle) -> list[Topic]:
        content = f"{article.title} {article.summary or ''}"
        topics = [Topic.NEWS]
        if any(token in content for token in ("股利", "配息", "除息", "公告", "董事會")):
            topics.append(Topic.ANNOUNCEMENT)
        if any(token in content for token in ("財報", "法說", "EPS", "營收", "獲利")):
            topics.append(Topic.EARNINGS)
        return topics

    def _describe_market_reaction(self, close_price: float, reference_price: float, close_fill_ratio: float) -> str:
        if close_price >= reference_price and close_fill_ratio >= 100:
            return "強勁"
        if close_price >= reference_price and close_fill_ratio >= 50:
            return "正向"
        if close_price >= reference_price:
            return "中性"
        return "偏弱"


class PostgresStockResolver(StockResolver):
    def __init__(self, gateway: FinMindPostgresGateway) -> None:
        self._gateway = gateway

    def resolve(self, query_text: str) -> tuple[str | None, str | None]:
        return self._gateway.resolve_company(query_text)


class PostgresMarketDocumentRepository(DocumentRepository):
    def __init__(self, gateway: FinMindPostgresGateway) -> None:
        self._gateway = gateway

    def upsert_documents(self, documents: list[Document]) -> int:
        _ = documents
        return 0

    def search_documents(self, query: StructuredQuery) -> list[Document]:
        return self._gateway.build_documents(query)
