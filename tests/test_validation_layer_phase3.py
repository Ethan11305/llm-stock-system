from datetime import datetime, timezone
import unittest

from llm_stock_system.core.enums import ConfidenceLight, ConsistencyStatus, FreshnessStatus, Intent, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, SourceCitation, StructuredQuery
from llm_stock_system.layers.validation_layer import ValidationLayer


UTC_NOW = datetime(2026, 4, 14, tzinfo=timezone.utc)


_SIGNATURE_TO_INTENT: dict[str, Intent] = {
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


def build_query(rule_signature: str, **overrides) -> StructuredQuery:
    defaults = {
        "user_query": f"validate {rule_signature}",
        "ticker": "2330",
        "company_name": "TSMC",
        "intent": _SIGNATURE_TO_INTENT[rule_signature],
        "time_range_label": "1y",
        "time_range_days": 365,
    }
    defaults.update(overrides)
    return StructuredQuery(**defaults)


def build_evidence(source_name: str, title: str, excerpt: str) -> Evidence:
    return Evidence(
        document_id=f"{source_name}-{title}",
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=SourceTier.HIGH,
        url="https://example.com/evidence",
        published_at=UTC_NOW,
        support_score=1.0,
        corroboration_count=1,
    )


def build_governance_report(evidence: list[Evidence]) -> GovernanceReport:
    return GovernanceReport(
        evidence=evidence,
        sufficiency=SufficiencyStatus.SUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT,
        freshness=FreshnessStatus.RECENT,
        high_trust_ratio=1.0,
    )


def build_answer_draft(summary: str, evidence: list[Evidence]) -> AnswerDraft:
    sources = [
        SourceCitation(
            title=item.title,
            source_name=item.source_name,
            source_tier=item.source_tier,
            url=item.url,
            published_at=item.published_at,
            excerpt=item.excerpt,
            support_score=item.support_score,
            corroboration_count=item.corroboration_count,
        )
        for item in evidence
    ]
    return AnswerDraft(
        summary=summary,
        highlights=[],
        facts=[],
        impacts=[],
        risks=["risk 1", "risk 2", "risk 3"],
        sources=sources,
    )


class ValidationLayerPhase3TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    def test_no_required_miss_keeps_full_confidence(self) -> None:
        query = build_query("investment_support")
        evidence = [
            build_evidence("FinMind TaiwanStockFinancialStatements", "EPS snapshot", "latest EPS 32.1 and gross margin stable"),
            build_evidence("TWSE valuation", "Valuation snapshot", "valuation percentile 62% and PE ratio 18.2"),
            build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot", "cash dividend 12.0"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price history remains stable"),
        ]

        result = self.validation.validate(
            query,
            build_governance_report(evidence),
            build_answer_draft("Grounded investment assessment.", evidence),
            facet_miss_list=[],
            preferred_miss_list=[],
        )

        self.assertEqual(result.confidence_score, 1.0)
        self.assertEqual(result.confidence_light, ConfidenceLight.GREEN)
        self.assertFalse(any("Required facet sync failed" in warning for warning in result.warnings))

    def test_partial_required_miss_caps_confidence(self) -> None:
        query = build_query("investment_support")
        evidence = [
            build_evidence("FinMind TaiwanStockFinancialStatements", "EPS snapshot", "latest EPS 32.1 and gross margin stable"),
            build_evidence("TWSE valuation", "Valuation snapshot", "valuation percentile 62% and PE ratio 18.2"),
            build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot", "cash dividend 12.0"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price history remains stable"),
        ]

        result = self.validation.validate(
            query,
            build_governance_report(evidence),
            build_answer_draft("Grounded investment assessment.", evidence),
            facet_miss_list=["pe_valuation"],
            preferred_miss_list=[],
        )

        self.assertEqual(result.confidence_score, 0.5)
        self.assertIn("Required facet sync failed (partial)", result.warnings[0])

    def test_all_required_miss_caps_confidence_to_red(self) -> None:
        query = build_query("earnings_summary")
        evidence = [
            build_evidence("FinMind TaiwanStockFinancialStatements", "Earnings snapshot 1", "EPS 10.0"),
            build_evidence("FinMind TaiwanStockFinancialStatements", "Earnings snapshot 2", "EPS 11.0"),
            build_evidence("FinMind TaiwanStockFinancialStatements", "Earnings snapshot 3", "EPS 12.0"),
            build_evidence("FinMind TaiwanStockFinancialStatements", "Earnings snapshot 4", "EPS 13.0"),
        ]

        result = self.validation.validate(
            query,
            build_governance_report(evidence),
            build_answer_draft("Grounded earnings review.", evidence),
            facet_miss_list=["financial_statements"],
            preferred_miss_list=[],
        )

        self.assertEqual(result.confidence_score, 0.25)
        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertIn("All required facets failed to sync", result.warnings[0])

    def test_preferred_miss_applies_penalty(self) -> None:
        query = build_query("technical_indicator_review")
        evidence = [
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 1", "price trend remains constructive"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 2", "price momentum remains constructive"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 3", "price range remains constructive"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 4", "price breakout remains constructive"),
        ]

        result = self.validation.validate(
            query,
            build_governance_report(evidence),
            build_answer_draft("Grounded technical review.", evidence),
            facet_miss_list=[],
            preferred_miss_list=["margin_data"],
        )

        self.assertEqual(result.confidence_score, 0.9)
        self.assertIn("Preferred facets not synced (1)", result.warnings[0])

    def test_preferred_penalty_is_capped_at_thirty_points(self) -> None:
        query = build_query("dividend_yield_review")
        evidence = [
            build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot 1", "cash dividend 10.0"),
            build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot 2", "cash dividend 11.0"),
            build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot 3", "cash dividend 12.0"),
            build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot 4", "cash dividend 13.0"),
        ]

        result = self.validation.validate(
            query,
            build_governance_report(evidence),
            build_answer_draft("Grounded dividend review.", evidence),
            facet_miss_list=[],
            preferred_miss_list=["cash_flow", "balance_sheet", "financial_statements"],
        )

        self.assertEqual(result.confidence_score, 0.7)
        self.assertIn("Preferred facets not synced (3)", result.warnings[0])

    def test_validate_accepts_missing_preferred_miss_list(self) -> None:
        query = build_query("technical_indicator_review")
        evidence = [
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 1", "price trend remains constructive"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 2", "price momentum remains constructive"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 3", "price range remains constructive"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 4", "price breakout remains constructive"),
        ]

        result = self.validation.validate(
            query,
            build_governance_report(evidence),
            build_answer_draft("Grounded technical review.", evidence),
            facet_miss_list=None,
        )

        self.assertEqual(result.confidence_score, 1.0)
        self.assertEqual(result.confidence_light, ConfidenceLight.GREEN)


if __name__ == "__main__":
    unittest.main()
