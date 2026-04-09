from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ConfidenceLight,
    ConsistencyStatus,
    FreshnessStatus,
    SourceTier,
    StanceBias,
    SufficiencyStatus,
    Topic,
)


class QueryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str
    ticker: str | None = None
    topic: Topic | None = None
    time_range: str | None = Field(default=None, alias="timeRange")


class StructuredQuery(BaseModel):
    user_query: str
    ticker: str | None = None
    company_name: str | None = None
    comparison_ticker: str | None = None
    comparison_company_name: str | None = None
    topic: Topic = Topic.COMPOSITE
    time_range_label: str = "7d"
    time_range_days: int = 7
    question_type: str = "market_summary"
    stance_bias: StanceBias = StanceBias.NEUTRAL


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    ticker: str
    title: str
    content: str
    source_name: str
    source_type: str
    source_tier: SourceTier
    url: str
    published_at: datetime
    author: str | None = None
    topics: list[Topic] = Field(default_factory=list)
    is_valid: bool = True


class StockInfo(BaseModel):
    stock_id: str
    stock_name: str
    industry_category: str | None = None
    market_type: str | None = None
    reference_date: datetime | None = None


class PriceBar(BaseModel):
    ticker: str
    trading_date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    trading_volume: int | None = None
    trading_money: int | None = None
    spread: float | None = None
    turnover: int | None = None


class FinancialStatementItem(BaseModel):
    ticker: str
    statement_date: datetime
    item_type: str
    value: float
    origin_name: str


class MonthlyRevenuePoint(BaseModel):
    ticker: str
    revenue_month: datetime
    revenue: float
    prior_year_month_revenue: float | None = None
    month_over_month_pct: float | None = None
    year_over_year_pct: float | None = None
    cumulative_revenue: float | None = None
    prior_year_cumulative_revenue: float | None = None
    cumulative_yoy_pct: float | None = None
    report_date: datetime | None = None
    notes: str | None = None


class ValuationPoint(BaseModel):
    ticker: str
    valuation_month: datetime
    pe_ratio: float | None = None
    peer_pe_ratio: float | None = None
    pb_ratio: float | None = None
    peer_pb_ratio: float | None = None


class DividendPolicy(BaseModel):
    ticker: str
    date: datetime
    year_label: str
    cash_earnings_distribution: float | None = None
    cash_statutory_surplus: float | None = None
    stock_earnings_distribution: float | None = None
    stock_statutory_surplus: float | None = None
    participate_distribution_of_total_shares: float | None = None
    announcement_date: datetime | None = None
    announcement_time: str | None = None
    cash_ex_dividend_trading_date: datetime | None = None
    cash_dividend_payment_date: datetime | None = None


class NewsArticle(BaseModel):
    ticker: str
    published_at: datetime
    title: str
    summary: str | None = None
    source_name: str
    url: str
    source_tier: SourceTier = SourceTier.MEDIUM
    source_type: str = "news_article"
    provider_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class MarginPurchaseShortSale(BaseModel):
    ticker: str
    trading_date: datetime
    margin_purchase_buy: int | None = None
    margin_purchase_cash_repayment: int | None = None
    margin_purchase_limit: int | None = None
    margin_purchase_sell: int | None = None
    margin_purchase_today_balance: int | None = None
    margin_purchase_yesterday_balance: int | None = None
    offset_loan_and_short: int | None = None
    short_sale_buy: int | None = None
    short_sale_cash_repayment: int | None = None
    short_sale_limit: int | None = None
    short_sale_sell: int | None = None
    short_sale_today_balance: int | None = None
    short_sale_yesterday_balance: int | None = None
    note: str | None = None


class Evidence(BaseModel):
    document_id: str
    title: str
    excerpt: str
    source_name: str
    source_tier: SourceTier
    url: str
    published_at: datetime
    support_score: float
    corroboration_count: int = 1


class GovernanceReport(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    dropped_document_ids: list[str] = Field(default_factory=list)
    sufficiency: SufficiencyStatus = SufficiencyStatus.INSUFFICIENT
    consistency: ConsistencyStatus = ConsistencyStatus.MOSTLY_CONSISTENT
    freshness: FreshnessStatus = FreshnessStatus.OUTDATED
    high_trust_ratio: float = 0.0


class SourceCitation(BaseModel):
    title: str
    source_name: str
    source_tier: SourceTier
    url: str
    published_at: datetime
    excerpt: str
    support_score: float
    corroboration_count: int


class AnswerDraft(BaseModel):
    summary: str
    highlights: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    impacts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)


class ValidationResult(BaseModel):
    confidence_score: float
    confidence_light: ConfidenceLight
    validation_status: str
    warnings: list[str] = Field(default_factory=list)


class DataStatus(BaseModel):
    sufficiency: SufficiencyStatus
    consistency: ConsistencyStatus
    freshness: FreshnessStatus


class QueryResponse(BaseModel):
    query_id: str
    summary: str
    highlights: list[str]
    facts: list[str]
    impacts: list[str]
    risks: list[str]
    data_status: DataStatus = Field(alias="dataStatus")
    confidence_light: ConfidenceLight = Field(alias="confidenceLight")
    confidence_score: float = Field(alias="confidenceScore")
    sources: list[SourceCitation]
    disclaimer: str

    model_config = ConfigDict(populate_by_name=True)


class SourceResponse(BaseModel):
    query_id: str
    ticker: str | None
    topic: Topic
    source_count: int
    sources: list[SourceCitation]
