import unittest
from datetime import datetime

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.openai_responses import OpenAIResponsesSynthesisClient
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


def make_evidence(
    document_id: str,
    title: str,
    excerpt: str,
    source_name: str = "TWSE IIH Company Financial",
    source_tier: SourceTier = SourceTier.HIGH,
) -> Evidence:
    return Evidence(
        document_id=document_id,
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=source_tier,
        url="https://example.com",
        published_at=datetime(2026, 4, 13, 10, 0, 0),
        support_score=1.0,
        corroboration_count=1,
    )


def make_report(*evidence: Evidence) -> GovernanceReport:
    return GovernanceReport(
        evidence=list(evidence),
        sufficiency=SufficiencyStatus.SUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT,
        freshness=FreshnessStatus.RECENT,
        high_trust_ratio=1.0,
    )


class StubOpenAIFundamentalPEClient(OpenAIResponsesSynthesisClient):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-key",
            model_name="gpt-4.1-mini",
            preliminary_answers_enabled=False,
        )

    def _request(self, payload: dict) -> dict:
        return {
            "text": (
                '{"summary":"台積電基本面很好，所以值得買。",'
                '"highlights":["營運動能強"],'
                '"facts":["模型偏多"],'
                '"impacts":["若AI需求續強有利股價"],'
                '"risks":["市場仍會波動"]}'
            )
        }


class FundamentalPEQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_fundamental_pe_review(self) -> None:
        query = InputLayer().parse(QueryRequest(query="台積電 2330 基本面跟本益比如何？"))

        self.assertEqual(query.question_type, "fundamental_pe_review")
        self.assertEqual(query.time_range_days, 365)
        self.assertEqual(query.company_name, "台積電")

    def test_rule_based_summary_combines_fundamental_and_pe(self) -> None:
        query = StructuredQuery(
            user_query="台積電 2330 基本面跟本益比如何？",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="fundamental_pe_review",
            time_range_label="1y",
            time_range_days=365,
        )
        governance_report = make_report(
            make_evidence("eps", "台積電年度財報", "全年 EPS 約 38.5 元，毛利率維持高檔。"),
            make_evidence("pe", "台積電估值觀察", "本益比約 24.3 倍，歷史分位 61%。"),
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("基本面來看", draft.summary)
        self.assertIn("估值面來看", draft.summary)
        self.assertIn("EPS 約 38.5 元", draft.summary)
        self.assertIn("本益比約 24.3 倍", draft.summary)
        self.assertIn("基本面：", draft.highlights[0])
        self.assertIn("本益比：", draft.highlights[1])

    def test_investment_support_uses_combined_summary(self) -> None:
        query = InputLayer().parse(QueryRequest(query="台積電 2330 現在可以買嗎？"))
        governance_report = make_report(
            make_evidence("eps", "台積電年度財報", "全年 EPS 約 38.5 元，毛利率維持高檔。"),
            make_evidence("pe", "台積電估值觀察", "本益比約 24.3 倍，歷史分位 61%。"),
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertEqual(query.question_type, "investment_support")
        self.assertIn("基本面來看", draft.summary)
        self.assertIn("估值面來看", draft.summary)

    def test_validation_requires_both_fundamental_and_valuation(self) -> None:
        query = StructuredQuery(
            user_query="台積電 2330 基本面跟本益比如何？",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="fundamental_pe_review",
            time_range_label="1y",
            time_range_days=365,
        )
        governance_report = make_report(
            make_evidence("eps", "台積電年度財報", "全年 EPS 約 38.5 元，毛利率維持高檔。"),
        )
        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            draft,
        )

        self.assertEqual(result.confidence_light, ConfidenceLight.YELLOW)
        self.assertGreater(result.confidence_score, 0.55)
        self.assertIn(
            "Combined fundamental and valuation review is missing direct valuation evidence",
            result.warnings[0],
        )

    def test_openai_guardrails_force_combined_summary(self) -> None:
        query = StructuredQuery(
            user_query="台積電 2330 基本面跟本益比如何？",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="fundamental_pe_review",
            time_range_label="1y",
            time_range_days=365,
        )
        governance_report = make_report(
            make_evidence("eps", "台積電年度財報", "全年 EPS 約 38.5 元，毛利率維持高檔。"),
            make_evidence("pe", "台積電估值觀察", "本益比約 24.3 倍，歷史分位 61%。"),
        )

        draft = StubOpenAIFundamentalPEClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("基本面來看", draft.summary)
        self.assertIn("估值面來看", draft.summary)
       