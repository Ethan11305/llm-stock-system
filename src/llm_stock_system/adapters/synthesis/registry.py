from llm_stock_system.core.enums import Intent

from .base import IntentStrategy
from .dividend_analysis import DividendAnalysisStrategy
from .earnings_review import EarningsReviewStrategy
from .fallback import FallbackStrategy
from .financial_health import FinancialHealthStrategy
from .investment_assessment import InvestmentAssessmentStrategy
from .news_digest import NewsDigestStrategy
from .technical_view import TechnicalViewStrategy
from .valuation_check import ValuationCheckStrategy

_REGISTRY: dict[Intent, IntentStrategy] = {
    Intent.NEWS_DIGEST:           NewsDigestStrategy(),
    Intent.EARNINGS_REVIEW:       EarningsReviewStrategy(),
    Intent.VALUATION_CHECK:       ValuationCheckStrategy(),
    Intent.DIVIDEND_ANALYSIS:     DividendAnalysisStrategy(),
    Intent.FINANCIAL_HEALTH:      FinancialHealthStrategy(),
    Intent.TECHNICAL_VIEW:        TechnicalViewStrategy(),
    Intent.INVESTMENT_ASSESSMENT: InvestmentAssessmentStrategy(),
}

_FALLBACK: IntentStrategy = FallbackStrategy()


def get_strategy(intent: Intent) -> IntentStrategy:
    return _REGISTRY.get(intent, _FALLBACK)
