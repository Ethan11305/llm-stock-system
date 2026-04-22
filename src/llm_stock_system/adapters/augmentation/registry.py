"""registry.py — intent → AugmentationStrategy 的查找表

使用延遲初始化確保 import 時不產生副作用。
"""
from __future__ import annotations

from llm_stock_system.core.enums import Intent
from llm_stock_system.adapters.augmentation.base import AugmentationStrategy

_REGISTRY: dict[Intent, AugmentationStrategy] | None = None


def _build_registry() -> dict[Intent, AugmentationStrategy]:
    from llm_stock_system.adapters.augmentation.news_digest import NewsDigestAugmentation
    from llm_stock_system.adapters.augmentation.earnings_review import EarningsReviewAugmentation
    from llm_stock_system.adapters.augmentation.valuation_check import ValuationCheckAugmentation
    from llm_stock_system.adapters.augmentation.investment_assessment import InvestmentAssessmentAugmentation
    from llm_stock_system.adapters.augmentation.financial_health import FinancialHealthAugmentation
    from llm_stock_system.adapters.augmentation.dividend_analysis import DividendAnalysisAugmentation
    from llm_stock_system.adapters.augmentation.technical_view import TechnicalViewAugmentation

    return {
        Intent.NEWS_DIGEST: NewsDigestAugmentation(),
        Intent.EARNINGS_REVIEW: EarningsReviewAugmentation(),
        Intent.VALUATION_CHECK: ValuationCheckAugmentation(),
        Intent.INVESTMENT_ASSESSMENT: InvestmentAssessmentAugmentation(),
        Intent.FINANCIAL_HEALTH: FinancialHealthAugmentation(),
        Intent.DIVIDEND_ANALYSIS: DividendAnalysisAugmentation(),
        Intent.TECHNICAL_VIEW: TechnicalViewAugmentation(),
    }


def get_augmentation_strategy(intent: Intent) -> AugmentationStrategy | None:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY.get(intent)
