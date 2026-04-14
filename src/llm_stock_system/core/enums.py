from enum import Enum


class Topic(str, Enum):
    NEWS = "news"
    EARNINGS = "earnings"
    ANNOUNCEMENT = "announcement"
    COMPOSITE = "composite"


class SourceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StanceBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ConfidenceLight(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class ConsistencyStatus(str, Enum):
    CONSISTENT = "consistent"
    MOSTLY_CONSISTENT = "mostly_consistent"
    CONFLICTING = "conflicting"


class FreshnessStatus(str, Enum):
    RECENT = "recent"
    STALE = "stale"
    OUTDATED = "outdated"


# --- Phase 0: New enums for intent-based routing ---


class Intent(str, Enum):
    NEWS_DIGEST = "news_digest"
    EARNINGS_REVIEW = "earnings_review"
    VALUATION_CHECK = "valuation_check"
    DIVIDEND_ANALYSIS = "dividend_analysis"
    FINANCIAL_HEALTH = "financial_health"
    TECHNICAL_VIEW = "technical_view"
    INVESTMENT_ASSESSMENT = "investment_assessment"


class DataFacet(str, Enum):
    PRICE_HISTORY = "price_history"
    FINANCIAL_STATEMENTS = "financial_statements"
    MONTHLY_REVENUE = "monthly_revenue"
    DIVIDEND = "dividend"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    PE_VALUATION = "pe_valuation"
    MARGIN_DATA = "margin_data"
    NEWS = "news"
