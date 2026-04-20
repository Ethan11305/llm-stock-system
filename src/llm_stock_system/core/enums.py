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


class TopicTag(str, Enum):
    """Controlled vocabulary for analysis topics.

    Used as ``controlled_tags`` on ``StructuredQuery``.  Each value is a
    human-readable Chinese label that is also stable enough to be used as a
    routing signal in the Retrieval and Generation layers.
    """
    # ── 原有分類 ──
    SHIPPING       = "航運"
    ELECTRICITY    = "電價"
    MACRO          = "總經"
    GUIDANCE       = "法說"
    TECHNICAL      = "技術面"
    MARGIN_FLOW    = "籌碼"
    SEMICON_EQUIP  = "半導體設備"
    EV             = "電動車"
    AI             = "AI"
    DIVIDEND       = "股利"
    REVENUE        = "月營收"
    GROSS_MARGIN   = "毛利率"
    VALUATION      = "本益比"
    FUNDAMENTAL    = "基本面"
    CASH_FLOW      = "現金流"
    DEBT           = "負債"
    LISTING        = "上市"

    # ── 新增分類 ──
    PROFITABILITY        = "獲利能力"
    OPERATING_EFFICIENCY = "營運效率"
    CAPEX_RD             = "資本支出研發"
    INSTITUTIONAL        = "法人動態"
    COMPETITIVE          = "競爭優勢"
    SUPPLY_CHAIN         = "供應鏈"
    ESG                  = "ESG永續"
    REGULATORY           = "政策法規"
    EVENT                = "重大事件"
    INDEX_REBAL          = "指數調整"
    FX                   = "匯率"
    SENTIMENT            = "市場情緒"
    RISK_MGMT            = "風險管理"

    # ── Digest 產品線專用 controlled tags ──
    THEME                = "題材"
    INDUSTRY             = "產業"


class QueryProfile(str, Enum):
    """最高階的產品分流欄位。

    - ``LEGACY``：沿用 question_type 衍生 intent 的舊路徑。
    - ``SINGLE_STOCK_DIGEST``：單股近 7 天新聞/公告/綜合 digest 可信產品線。

    Digest 路徑裡主流程只看 ``query_profile + intent + topic + topic_tags +
    time_range``；``question_type`` 僅保留作為相容與分析標籤，不再參與控制流。
    """

    LEGACY = "legacy"
    SINGLE_STOCK_DIGEST = "single_stock_digest"


class ForecastMode(str, Enum):
    """How the forecast was derived."""
    SCENARIO_ESTIMATE = "scenario_estimate"
    HISTORICAL_PROXY = "historical_proxy"
    UNSUPPORTED = "unsupported"


class ForecastDirection(str, Enum):
    """Directional bias for a forecast."""
    BULLISH_BIAS = "bullish_bias"
    BEARISH_BIAS = "bearish_bias"
    RANGE_BOUND = "range_bound"
    UNDETERMINED = "undetermined"
