from datetime import datetime, timezone
import unittest

from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, SourceCitation, StructuredQuery
from llm_stock_system.core.validation_profiles import get_profile
from llm_stock_system.layers.validation_layer import ValidationLayer


UTC_NOW = datetime(2026, 4, 16, tzinfo=timezone.utc)


def build_query(question_type: str, **overrides) -> StructuredQuery:
    defaults = {
        "user_query": f"validate {question_type}",
        "ticker": "2330",
        "company_name": "TSMC",
        "question_type": question_type,
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


class ValidationRuleEngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    def test_warning_only_profile_keeps_score(self) -> None:
        profile = get_profile("technical_indicator_review")
        assert profile is not None
        query = build_query("technical_indicator_review")
        evidence = [
            build_evidence("Generic source", "Context 1", "market commentary"),
            build_evidence("Generic source", "Context 2", "market commentary"),
            build_evidence("Generic source", "Context 3", "market commentary"),
            build_evidence("Generic source", "Context 4", "market commentary"),
        ]
        warnings: list[str] = []

        score = self.validation._evaluate_profile(
            profile,
            query,
            build_governance_report(evidence),
            build_answer_draft("Grounded technical review.", evidence),
            1.0,
            warnings,
        )

        self.assertEqual(score, 1.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("no price evidence", warnings[0])

    def test_dual_signal_profile_caps_to_yellow(self) -> None:
        profile = get_profile("season_line_margin_review")
        assert profile is not None
        query = build_query("season_line_margin_review")
        evidence = [
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 1", "price trend remains weak"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 2", "price remains below season line"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 3", "price volume remains active"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot 4", "price volatility remains elevated"),
        ]
        warnings: list[str] = []

        score = self.validation._evaluate_profile(
            profile,
            query,
            build_governance_report(evidence),
            build_answer_draft("Partial season-line review.", evidence),
            1.0,
            warnings,
        )

        self.assertEqual(score, 0.5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("missing one of price or margin evidence", warnings[0])

    def test_custom_validator_profile_caps_target_price_context(self) -> None:
        profile = get_profile("price_outlook")
        assert profile is not None
        query = build_query(
            "price_outlook",
            user_query="TSMC 未來一年目標價是多少？",
        )
        evidence = [
            build_evidence("Broker research", "Analyst update", "法人目標價上看 1280 元"),
            build_evidence("Broker research", "Valuation note", "target price 1280 based on demand recovery"),
            build_evidence("TWSE valuation", "Valuation snapshot", "valuation percentile 62% and PE ratio 18.2"),
            build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price history remains stable"),
        ]
        warnings: list[str] = []

        score = self.validation._evaluate_profile(
            profile,
            query,
            build_governance_report(evidence),
            build_answer_draft("Forward-looking price view.", evidence),
            1.0,
            warnings,
        )

        self.assertEqual(score, 0.55)
        self.assertEqual(len(warnings), 1)
        self.assertIn("scenario-dependent", warnings[0])


if __name__ == "__main__":
    unittest.main()
