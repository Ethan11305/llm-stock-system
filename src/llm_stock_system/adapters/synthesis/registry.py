from llm_stock_system.core.enums import Intent

from .base import IntentStrategy
from .dividend_analysis import DividendAnalysisStrategy
from .earnings_review import EarningsReviewStrategy
from .fallback import FallbackStrategy
from .financial_health import FinancialHealthStrategy
from .news_digest import NewsDigestStrategy
from .technical_view import TechnicalViewStrategy

# Wave 2 sunset：移除 VALUATION_CHECK / INVESTMENT_ASSESSMENT 專屬 strategy
# 兩者交由 FallbackStrategy 處理（registry.get 找不到時 fall through）。
_REGISTRY: dict[Intent, IntentStrategy] = {
    Intent.NEWS_DIGEST:       NewsDigestStrategy(),
    Intent.EARNINGS_REVIEW:   EarningsReviewStrategy(),
    Intent.DIVIDEND_ANALYSIS: DividendAnalysisStrategy(),
    Intent.FINANCIAL_HEALTH:  FinancialHealthStrategy(),
    Intent.TECHNICAL_VIEW:    TechnicalViewStrategy(),
}

_FALLBACK: IntentStrategy = FallbackStrategy()


def get_strategy(intent: Intent) -> IntentStrategy:
    return _REGISTRY.get(intent, _FALLBACK)
