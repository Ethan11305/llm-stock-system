from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    ConfidenceLight,
    ConsistencyStatus,
    DataFacet,
    ForecastDirection,
    ForecastMode,
    FreshnessStatus,
    Intent,
    QueryProfile,
    SourceTier,
    StanceBias,
    SufficiencyStatus,
    Topic,
    TopicTag,
)


QUESTION_TYPE_TO_INTENT: dict[str, Intent] = {
    "market_summary": Intent.NEWS_DIGEST,
    "theme_impact_review": Intent.NEWS_DIGEST,
    "shipping_rate_impact_review": Intent.NEWS_DIGEST,
    "electricity_cost_impact_review": Intent.NEWS_DIGEST,
    "macro_yield_sentiment_review": Intent.NEWS_DIGEST,
    "guidance_reaction_review": Intent.NEWS_DIGEST,
    "listing_revenue_review": Intent.NEWS_DIGEST,
    "earnings_summary": Intent.EARNINGS_REVIEW,
    "eps_dividend_review": Intent.EARNINGS_REVIEW,
    "monthly_revenue_yoy_review": Intent.EARNINGS_REVIEW,
    "margin_turnaround_review": Intent.EARNINGS_REVIEW,
    "pe_valuation_review": Intent.VALUATION_CHECK,
    "fundamental_pe_review": Intent.VALUATION_CHECK,
    "price_range": Intent.VALUATION_CHECK,
    "price_outlook": Intent.VALUATION_CHECK,
    "dividend_yield_review": Intent.DIVIDEND_ANALYSIS,
    "ex_dividend_performance": Intent.DIVIDEND_ANALYSIS,
    "fcf_dividend_sustainability_review": Intent.DIVIDEND_ANALYSIS,
    "debt_dividend_safety_review": Intent.DIVIDEND_ANALYSIS,
    "profitability_stability_review": Intent.FINANCIAL_HEALTH,
    "gross_margin_comparison_review": Intent.FINANCIAL_HEALTH,
    "revenue_growth_review": Intent.FINANCIAL_HEALTH,
    "technical_indicator_review": Intent.TECHNICAL_VIEW,
    "season_line_margin_review": Intent.TECHNICAL_VIEW,
    "investment_support": Intent.INVESTMENT_ASSESSMENT,
    "risk_review": Intent.INVESTMENT_ASSESSMENT,
    "announcement_summary": Intent.INVESTMENT_ASSESSMENT,
}


# Backward compatibility: when callers construct StructuredQuery with only
# ``question_type`` (no controlled_tags / free_keywords), these fallback tags
# are injected so Generation adapters can still route on topic_tags.
QUESTION_TYPE_FALLBACK_TOPIC_TAGS: dict[str, tuple[str, ...]] = {
    "theme_impact_review":              ("題材", "產業"),
    "shipping_rate_impact_review":      ("航運", "SCFI"),
    "electricity_cost_impact_review":   ("電價", "成本"),
    "macro_yield_sentiment_review":     ("CPI", "殖利率"),
    "guidance_reaction_review":         ("法說", "指引"),
    "listing_revenue_review":           ("上市", "營收"),
    "monthly_revenue_yoy_review":       ("月營收",),
    "margin_turnaround_review":         ("毛利率", "轉正"),
    "gross_margin_comparison_review":   ("毛利率", "比較"),
    "pe_valuation_review":              ("本益比",),
    "fundamental_pe_review":            ("基本面", "本益比"),
    "price_range":                      ("股價區間",),
    "price_outlook":                    ("股價", "展望"),
    "dividend_yield_review":            ("股利", "殖利率"),
    "ex_dividend_performance":          ("除息", "填息"),
    "fcf_dividend_sustainability_review": ("股利", "現金流"),
    "debt_dividend_safety_review":      ("股利", "負債"),
    "profitability_stability_review":   ("獲利", "穩定性"),
    "revenue_growth_review":            ("營收", "成長"),
    "technical_indicator_review":       ("技術面",),
    "season_line_margin_review":        ("季線", "籌碼"),
    "earnings_summary":                 ("財報",),
    "eps_dividend_review":              ("EPS", "股利"),
    "investment_support":               ("投資評估", "基本面", "本益比"),
    "risk_review":                      ("風險",),
    "announcement_summary":             ("公告",),

    # ── 新增 question_type fallback（對應新分類）──
    "profitability_review":             ("獲利能力", "ROE"),
    "operating_efficiency_review":      ("營運效率", "存貨週轉"),
    "capex_rd_review":                  ("資本支出", "研發"),
    "institutional_flow_review":        ("外資", "投信"),
    "competitive_review":               ("競爭優勢", "市佔率"),
    "supply_chain_review":              ("供應鏈", "供應商"),
    "esg_review":                       ("ESG", "永續"),
    "regulatory_review":                ("政策", "監管"),
    "event_review":                     ("併購", "重大事件"),
    "index_rebal_review":               ("MSCI", "指數調整"),
    "fx_impact_review":                 ("匯率", "匯損"),
    "sentiment_review":                 ("市場情緒", "VIX"),
    "risk_mgmt_review":                 ("停損", "風險管理"),
}


@dataclass(frozen=True)
class FacetSpec:
    required: frozenset[DataFacet]
    preferred: frozenset[DataFacet]


INTENT_FACET_SPECS: dict[Intent, FacetSpec] = {
    Intent.NEWS_DIGEST: FacetSpec(
        required=frozenset({DataFacet.NEWS}),
        preferred=frozenset({DataFacet.PRICE_HISTORY}),
    ),
    Intent.EARNINGS_REVIEW: FacetSpec(
        required=frozenset({DataFacet.FINANCIAL_STATEMENTS}),
        preferred=frozenset({DataFacet.MONTHLY_REVENUE, DataFacet.NEWS}),
    ),
    Intent.VALUATION_CHECK: FacetSpec(
        required=frozenset({DataFacet.PE_VALUATION}),
        preferred=frozenset(
            {
                DataFacet.PRICE_HISTORY,
                DataFacet.FINANCIAL_STATEMENTS,
                DataFacet.NEWS,
            }
        ),
    ),
    Intent.DIVIDEND_ANALYSIS: FacetSpec(
        required=frozenset({DataFacet.DIVIDEND}),
        preferred=frozenset(
            {
                DataFacet.CASH_FLOW,
                DataFacet.BALANCE_SHEET,
                DataFacet.FINANCIAL_STATEMENTS,
            }
        ),
    ),
    Intent.FINANCIAL_HEALTH: FacetSpec(
        required=frozenset({DataFacet.FINANCIAL_STATEMENTS}),
        preferred=frozenset({DataFacet.MONTHLY_REVENUE, DataFacet.NEWS}),
    ),
    Intent.TECHNICAL_VIEW: FacetSpec(
        required=frozenset({DataFacet.PRICE_HISTORY}),
        preferred=frozenset({DataFacet.MARGIN_DATA}),
    ),
    Intent.INVESTMENT_ASSESSMENT: FacetSpec(
        required=frozenset({DataFacet.FINANCIAL_STATEMENTS, DataFacet.PE_VALUATION}),
        preferred=frozenset({DataFacet.DIVIDEND, DataFacet.NEWS, DataFacet.PRICE_HISTORY}),
    ),
}

INTENT_FACETS: dict[Intent, frozenset[DataFacet]] = {
    intent: spec.required | spec.preferred for intent, spec in INTENT_FACET_SPECS.items()
}


def infer_intent_from_question_type(question_type: str | None) -> Intent:
    if question_type is None:
        return Intent.NEWS_DIGEST
    return QUESTION_TYPE_TO_INTENT.get(question_type, Intent.NEWS_DIGEST)


def infer_data_facets(
    intent: Intent | str | None,
    extra_required: set[DataFacet | str] | list[DataFacet | str] | None = None,
    extra_preferred: set[DataFacet | str] | list[DataFacet | str] | None = None,
) -> tuple[set[DataFacet | str], set[DataFacet | str]]:
    required = set(extra_required or [])
    preferred = set(extra_preferred or [])
    try:
        resolved_intent = intent if isinstance(intent, Intent) else Intent(intent) if intent is not None else None
    except ValueError:
        resolved_intent = None

    if resolved_intent is None:
        return required, preferred

    spec = INTENT_FACET_SPECS.get(resolved_intent, FacetSpec(frozenset(), frozenset()))
    return set(spec.required) | required, set(spec.preferred) | preferred


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


class ForecastWindow(BaseModel):
    label: str
    start_date: str
    end_date: str


class ScenarioRange(BaseModel):
    low: float
    high: float
    basis_type: str  # e.g. "analyst_target", "historical_proxy", "support_resistance"


class ForecastBlock(BaseModel):
    """Structured forecast output attached to QueryResponse."""
    mode: ForecastMode
    forecast_window: ForecastWindow
    direction: ForecastDirection = ForecastDirection.UNDETERMINED
    scenario_range: ScenarioRange | None = None
    forecast_basis: list[str] = Field(default_factory=list)


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
    intent: Intent = Intent.NEWS_DIGEST
    required_facets: set[DataFacet] = Field(default_factory=set)
    preferred_facets: set[DataFacet] = Field(default_factory=set)
    data_facets: set[DataFacet] = Field(default_factory=set)
    controlled_tags: list[TopicTag] = Field(default_factory=list)
    free_keywords: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    tag_source: str = "empty"
    question_type: str = "market_summary"
    stance_bias: StanceBias = StanceBias.NEUTRAL
    classifier_source: str = "rule"  # "rule" | "llm" | "mixed"
    query_profile: QueryProfile = QueryProfile.LEGACY
    # --- Forecast semantic fields ---
    is_forecast_query: bool = False
    wants_direction: bool = False
    wants_scenario_range: bool = False
    forecast_horizon_label: str | None = None
    forecast_horizon_days: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _populate_intent_metadata(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data

        values = dict(data)
        question_type = values.get("question_type")
        if values.get("intent") in (None, ""):
            values["intent"] = infer_intent_from_question_type(question_type)

        required_facets, preferred_facets = infer_data_facets(
            values.get("intent"),
            values.get("required_facets"),
            values.get("preferred_facets"),
        )

        legacy_data_facets = set(values.get("data_facets") or [])
        preferred_facets |= legacy_data_facets - required_facets

        # --- Forecast facet override ---
        # price_outlook with is_forecast_query should require PRICE_HISTORY
        # (not PE_VALUATION) and demote PE_VALUATION/NEWS to preferred.
        if values.get("is_forecast_query") and question_type == "price_outlook":
            required_facets = {DataFacet.PRICE_HISTORY}
            preferred_facets = (preferred_facets | {DataFacet.PE_VALUATION, DataFacet.NEWS}) - required_facets

        values["required_facets"] = required_facets
        values["preferred_facets"] = preferred_facets
        values["data_facets"] = required_facets | preferred_facets

        controlled_tags = list(values.get("controlled_tags") or [])
        free_keywords = list(values.get("free_keywords") or [])
        fallback_tags = list(QUESTION_TYPE_FALLBACK_TOPIC_TAGS.get(question_type or "", ()))
        if controlled_tags or free_keywords:
            values["topic_tags"] = _dedupe_preserving_order(
                [tag.value if isinstance(tag, TopicTag) else str(tag) for tag in controlled_tags]
                + [str(keyword) for keyword in free_keywords]
                + fallback_tags
            )
        elif "topic_tags" not in values:
            values["topic_tags"] = fallback_tags

        if values.get("tag_source") in (None, ""):
            if controlled_tags:
                values["tag_source"] = "matched"
            elif free_keywords or fallback_tags:
                values["tag_source"] = "fallback"
            else:
                values["tag_source"] = "empty"

        return values


@dataclass
class HydrationResult:
    query_id: str = dataclass_field(default_factory=lambda: str(uuid4()))
    synced_facets: set[DataFacet] = dataclass_field(default_factory=set)
    failed_facets: dict[DataFacet, str] = dataclass_field(default_factory=dict)
    facet_miss_list: list[str] = dataclass_field(default_factory=list)
    preferred_miss_list: list[str] = dataclass_field(default_factory=list)
    total_duration_ms: float = 0.0


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
    forecast: ForecastBlock | None = None


class ValidationResult(BaseModel):
    confidence_score: float
    confidence_light: ConfidenceLight
    validation_status: str
    warnings: list[str] = Field(default_factory=list)
    facet_miss_list: list[str] = Field(default_factory=list)


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
    forecast: ForecastBlock | None = None
    # P3 UI parity：把回查 QueryLogDetail 才看得到的 meta 補進即時回應，
    # 讓前端不必二次查詢 /query-log/{id} 就能顯示 warnings / 分類來源 / digest 標籤。
    warnings: list[str] = Field(default_factory=list)
    classifier_source: str = Field(default="rule", alias="classifierSource")
    query_profile: QueryProfile = Field(
        default=QueryProfile.LEGACY,
        alias="queryProfile",
    )

    model_config = ConfigDict(populate_by_name=True)


class QueryLogDetail(BaseModel):
    """單一查詢的完整回查資料。

    採用組合式設計：直接嵌入 ``QueryResponse``，避免欄位漂移時兩套 schema 不同步。

    Digest 產品線與 legacy 路徑皆可共用這個模型。``query_profile`` 欄位標示
    該次查詢是否走 digest 路徑，便於前端或分析工具分流。
    """

    query_id: str = Field(alias="queryId")
    query_profile: QueryProfile = Field(alias="queryProfile")
    classifier_source: str = Field(alias="classifierSource")
    validation_status: str = Field(alias="validationStatus")
    warnings: list[str] = Field(default_factory=list)
    source_count: int = Field(default=0, alias="sourceCount")
    schema_version: int = Field(default=1, alias="schemaVersion")

    structured_query: dict = Field(default_factory=dict, alias="structuredQuery")
    response: QueryResponse

    model_config = ConfigDict(populate_by_name=True)


class SourceResponse(BaseModel):
    query_id: str
    ticker: str | None
    topic: Topic
    source_count: int
    sources: list[SourceCitation] = Field(default_factory=list)
