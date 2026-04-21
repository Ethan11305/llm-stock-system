from datetime import datetime, timezone
import unittest

from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, Intent, SourceTier, SufficiencyStatus, TopicTag
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, SourceCitation, StructuredQuery
from llm_stock_system.core.validation_profiles import get_profile
from llm_stock_system.layers.validation_layer import ValidationLayer


UTC_NOW = datetime(2026, 4, 16, tzinfo=timezone.utc)


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


class ValidationRuleEngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    def test_warning_only_profile_keeps_score(self) -> None:
        query = build_query("technical_indicator_review")
        profile = get_profile(query.intent)
        assert profile is not None
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

if __name__ == "__main__":
    unittest.main()
