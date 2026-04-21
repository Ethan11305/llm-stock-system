"""Behaviour tests for ValidationProfile — surviving question_type scenarios only.

Wave 3 sunset：ex_dividend / profitability_stability / margin_turnaround /
season_line_margin / fcf_dividend_sustainability / debt_dividend_safety /
gross_margin_comparison 的 profile 全部下架，此檔案只保留仍活著的兩條：
  * technical_indicator_review（TAG_TECHNICAL，warning-only）
  * monthly_revenue_yoy_review（TAG_MONTHLY_REVENUE，warning-only）
"""

from datetime import datetime, timezone
import unittest

from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, Intent, SourceTier, SufficiencyStatus, TopicTag
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, SourceCitation, StructuredQuery
from llm_stock_system.core.validation_profiles import get_profile
from llm_stock_system.layers.validation_layer import ValidationLayer


UTC_NOW = datetime(2026, 4, 16, tzinfo=timezone.utc)

# Partial-revenue tokens (previously a ValidationLayer class constant, now inline)
_PARTIAL_REVENUE_TOKEN = "部分月份"


_SIGNATURE_TO_INTENT: dict[str, Intent] = {
    "technical_indicator_review": Intent.TECHNICAL_VIEW,
    "monthly_revenue_yoy_review": Intent.EARNINGS_REVIEW,
}

_SIGNATURE_TO_CONTROLLED_TAGS: dict[str, list[TopicTag]] = {
    "technical_indicator_review": [TopicTag.TECHNICAL],
    "monthly_revenue_yoy_review": [TopicTag.REVENUE],
}


def build_query(rule_signature: str, **overrides) -> StructuredQuery:
    defaults = {
        "user_query": f"validate {rule_signature}",
        "ticker": "2330",
        "company_name": "TSMC",
        "intent": _SIGNATURE_TO_INTENT[rule_signature],
        "controlled_tags": list(_SIGNATURE_TO_CONTROLLED_TAGS.get(rule_signature, [])),
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


class ValidationProfileParityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    def test_profile_score_parity(self) -> None:
        cases = [
            ("technical_indicator_review", "happy", 1.0),
            ("technical_indicator_review", "degraded", 1.0),
            ("monthly_revenue_yoy_review", "happy", 1.0),
            ("monthly_revenue_yoy_review", "degraded", 1.0),
        ]

        for question_type, scenario, expected_score in cases:
            with self.subTest(question_type=question_type, scenario=scenario):
                query, governance_report, answer_draft = self._build_case(question_type, scenario)
                profile = get_profile(query.intent)
                self.assertIsNotNone(profile, f"No profile registered for intent of {question_type!r}")

                warnings: list[str] = []
                profile_score = self.validation._evaluate_profile(
                    profile,
                    query,
                    governance_report,
                    answer_draft,
                    1.0,
                    warnings,
                )

                self.assertEqual(
                    profile_score,
                    expected_score,
                    f"Profile score mismatch for {question_type}/{scenario}: "
                    f"got {profile_score}, expected {expected_score}. "
                    f"Warnings: {warnings}",
                )

    def _build_case(
        self,
        question_type: str,
        scenario: str,
    ) -> tuple[StructuredQuery, GovernanceReport, AnswerDraft]:
        if question_type == "technical_indicator_review":
            evidence = [build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price trend remains constructive")]
            if scenario == "degraded":
                evidence = [build_evidence("Generic source", "Context", "market commentary")]
            return self._case_from_evidence(question_type, evidence, "Technical review.")

        if question_type == "monthly_revenue_yoy_review":
            evidence = [build_evidence("TWSE monthly revenue", "Revenue snapshot", "monthly revenue year over year improved")]
            summary = "Monthly revenue review."
            if scenario == "degraded":
                summary = f"Monthly revenue review. {_PARTIAL_REVENUE_TOKEN}"
            return self._case_from_evidence(question_type, evidence, summary)

        raise ValueError(f"Unsupported parity case: {question_type}/{scenario}")

    def _case_from_evidence(
        self,
        question_type: str,
        evidence: list[Evidence],
        summary: str,
    ) -> tuple[StructuredQuery, GovernanceReport, AnswerDraft]:
        query = build_query(question_type)
        return query, build_governance_report(evidence), build_answer_draft(summary, evidence)


if __name__ == "__main__":
    unittest.main()
