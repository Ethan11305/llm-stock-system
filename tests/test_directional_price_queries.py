"""Tests for directional price outlook queries.

Covers three concerns:
1. Input layer classifies directional sentences as price_outlook.
2. Validation layer applies the directional cap (0.55 with context, 0.25 without).
3. Validation layer emits the correct warning messages.
"""
import unittest
from datetime import datetime

from llm_stock_system.core.enums import (
    ConfidenceLight,
    ConsistencyStatus,
    FreshnessStatus,
    SourceTier,
    SufficiencyStatus,
    Topic,
)
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest, StructuredQuery
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.validation_layer import ValidationLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evidence(title: str, excerpt: str, source_name: str = "moneydj") -> Evidence:
    return Evidence(
        document_id=f"doc-{abs(hash((title, excerpt))) % 100000}",
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=SourceTier.MEDIUM,
        url="https://example.com/article",
        published_at=datetime(2026, 4, 14, 10, 0, 0),
        support_score=0.8,
        corroboration_count=1,
    )


def make_report(*evidence: Evidence, freshness: FreshnessStatus = FreshnessStatus.RECENT) -> GovernanceReport:
    return GovernanceReport(
        evidence=list(evidence),
        sufficiency=SufficiencyStatus.SUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT,
        freshness=freshness,
        high_trust_ratio=0.9,
    )


def make_empty_report() -> GovernanceReport:
    return GovernanceReport(
        evidence=[],
        sufficiency=SufficiencyStatus.INSUFFICIENT,
        consistency=ConsistencyStatus.CONFLICTING,
        freshness=FreshnessStatus.OUTDATED,
        high_trust_ratio=0.0,
    )


def make_directional_query(user_query: str, ticker: str = "2330", company: str = "台積電") -> StructuredQuery:
    return StructuredQuery(
        user_query=user_query,
        ticker=ticker,
        company_name=company,
        topic=Topic.COMPOSITE,
        question_type="price_outlook",
        time_range_label="7d",
        time_range_days=7,
    )


# ---------------------------------------------------------------------------
# 1. Input layer classification
# ---------------------------------------------------------------------------

class DirectionalPriceClassificationTestCase(unittest.TestCase):
    """New sentence patterns that should be routed to price_outlook."""

    def setUp(self) -> None:
        self.layer = InputLayer()

    def _classify(self, query: str) -> str:
        return self.layer.parse(QueryRequest(query=query)).question_type

    def test_continue_rising_pattern(self) -> None:
        self.assertEqual(self._classify("台積電 2330 明天還會繼續漲嗎？"), "price_outlook")

    def test_continuation_abbreviated_rising(self) -> None:
        self.assertEqual(self._classify("2330 續漲機會大嗎？"), "price_outlook")

    def test_will_it_rise_or_not(self) -> None:
        self.assertEqual(self._classify("台積電短線還會不會漲？"), "price_outlook")

    def test_continue_falling_pattern(self) -> None:
        self.assertEqual(self._classify("2330 還會繼續跌嗎？"), "price_outlook")

    def test_upside_room_pattern(self) -> None:
        self.assertEqual(self._classify("台積電下週還有上漲空間嗎？"), "price_outlook")

    def test_how_much_more_can_it_rise(self) -> None:
        self.assertEqual(self._classify("這支股票還能漲多少？"), "price_outlook")

    def test_continuation_abbreviated_falling(self) -> None:
        self.assertEqual(self._classify("2330 短線續跌風險高嗎？"), "price_outlook")

    def test_downside_room_pattern(self) -> None:
        self.assertEqual(self._classify("台積電還有多少下跌空間？"), "price_outlook")

    # --- Boundaries: should NOT be price_outlook ---

    def test_investment_decision_is_not_price_outlook(self) -> None:
        # "還可以追嗎" is an investment decision, not a directional prediction
        result = self._classify("2330 還可以追嗎？")
        self.assertNotEqual(result, "price_outlook")

    def test_vague_trajectory_stays_market_summary(self) -> None:
        # "接下來會怎麼走" has no directional keyword → market_summary
        result = self._classify("台積電接下來會怎麼走？")
        self.assertNotEqual(result, "price_outlook")

    # --- Regression: existing patterns still work ---

    def test_existing_will_rise_pattern_unchanged(self) -> None:
        self.assertEqual(self._classify("台積電明天會漲嗎？"), "price_outlook")

    def test_existing_trend_pattern_unchanged(self) -> None:
        self.assertEqual(self._classify("2330 後續走勢如何？"), "price_outlook")

    def test_existing_target_price_unchanged(self) -> None:
        self.assertEqual(self._classify("華邦電 2344 未來半年目標價是多少？"), "price_outlook")


# ---------------------------------------------------------------------------
# 2 & 3. Validation layer cap and warning
# ---------------------------------------------------------------------------

class DirectionalPriceValidationTestCase(unittest.TestCase):
    """Directional queries should be capped conservatively by the validation layer."""

    def setUp(self) -> None:
        self.validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    # --- RED: no evidence ---

    def test_no_evidence_gives_red_confidence(self) -> None:
        query = make_directional_query("台積電明天還會繼續漲嗎？")
        result = self.validation.validate(query, make_empty_report(), _empty_draft())

        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertLessEqual(result.confidence_score, 0.25)

    def test_no_evidence_emits_directional_warning(self) -> None:
        query = make_directional_query("2330 短線還會不會漲？")
        result = self.validation.validate(query, make_empty_report(), _empty_draft())

        self.assertIn(
            "Directional price query lacks analyst, technical, or price-level evidence"
            " to support a directional view.",
            result.warnings,
        )

    # --- YELLOW: has forward price context ---

    def test_with_analyst_context_gives_yellow(self) -> None:
        query = make_directional_query("台積電後續還會繼續漲嗎？")
        evidence = make_evidence(
            "外資持續看多台積電",
            "外資法人調升台積電評等，認為技術面支撐仍在，壓力區約 1050 元附近。",
        )
        result = self.validation.validate(query, make_report(evidence), _draft_with_source(evidence))

        self.assertEqual(result.confidence_light, ConfidenceLight.YELLOW)
        self.assertLessEqual(result.confidence_score, 0.55)

    def test_with_forward_context_emits_context_warning(self) -> None:
        query = make_directional_query("台積電還有上漲空間嗎？")
        evidence = make_evidence(
            "分析師看法",
            "分析師指出目前本益比仍在歷史合理區間，若業績符合預期，上漲空間仍有限但存在。",
        )
        result = self.validation.validate(query, make_report(evidence), _draft_with_source(evidence))

        self.assertIn(
            "Directional price query has related market context but cannot confirm"
            " specific future price movement from public data.",
            result.warnings,
        )

    def test_with_context_score_is_capped_at_055(self) -> None:
        query = make_directional_query("2330 續漲機會有多少？")
        evidence = make_evidence(
            "外資買超台積電",
            "外資上週連續買超台積電，分析師認為均線支撐仍強，但短線受美股牽連。",
        )
        result = self.validation.validate(query, make_report(evidence), _draft_with_source(evidence))

        self.assertLessEqual(result.confidence_score, 0.55)
        self.assertGreater(result.confidence_score, 0.0)

    # --- Existing price_outlook subtypes unaffected ---

    def test_target_price_subtype_still_uses_original_rules(self) -> None:
        # A target-price question should still route to _apply_price_outlook_rules,
        # not the new directional path.
        query = StructuredQuery(
            user_query="華邦電 2344 未來半年目標價是多少？",
            ticker="2344",
            company_name="華邦電",
            topic=Topic.COMPOSITE,
            question_type="price_outlook",
            time_range_label="6m",
            time_range_days=180,
        )
        result = self.validation.validate(query, make_empty_report(), _empty_draft())

        # Should still cap at 0.25 (RED) but via _apply_price_outlook_rules
        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertIn("Forward price query lacks direct target-price", result.warnings[-1])


# ---------------------------------------------------------------------------
# Minimal AnswerDraft helpers
# ---------------------------------------------------------------------------

def _empty_draft():
    from llm_stock_system.core.models import AnswerDraft
    return AnswerDraft(summary="資料不足，無法確認。", highlights=[], facts=[], impacts=[], risks=[], sources=[])


def _draft_with_source(evidence: Evidence):
    from llm_stock_system.core.models import AnswerDraft, SourceCitation
    return AnswerDraft(
        summary="根據現有資料，方向性判斷需保守看待。",
        highlights=[],
        facts=[],
        impacts=[],
        risks=["short-term risk 1", "short-term risk 2", "short-term risk 3"],
        sources=[
            SourceCitation(
                title=evidence.title,
                source_name=evidence.source_name,
                source_tier=evidence.source_tier,
                url=evidence.url,
                published_at=evidence.published_at,
                excerpt=evidence.excerpt,
                support_score=evidence.support_score,
                corroboration_count=evidence.corroboration_count,
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
