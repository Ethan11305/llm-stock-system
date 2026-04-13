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
    title: str,
    excerpt: str,
    source_name: str = "moneydj",
    source_tier: SourceTier = SourceTier.MEDIUM,
) -> Evidence:
    return Evidence(
        document_id=f"doc-{abs(hash((title, excerpt, source_name))) % 100000}",
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=source_tier,
        url="https://example.com/article",
        published_at=datetime(2026, 4, 13, 10, 0, 0),
        support_score=0.8,
        corroboration_count=1,
    )


def make_governance_report(*evidence: Evidence) -> GovernanceReport:
    return GovernanceReport(
        evidence=list(evidence),
        sufficiency=SufficiencyStatus.SUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT,
        freshness=FreshnessStatus.RECENT,
        high_trust_ratio=0.9,
    )


def make_target_price_query() -> StructuredQuery:
    return StructuredQuery(
        user_query="華邦電 2344 未來半年目標價是多少？",
        ticker="2344",
        company_name="華邦電",
        topic=Topic.COMPOSITE,
        question_type="price_outlook",
        time_range_label="30d",
        time_range_days=30,
    )


class StubOpenAITargetPriceClient(OpenAIResponsesSynthesisClient):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-key",
            model_name="gpt-4.1-mini",
            preliminary_answers_enabled=False,
        )

    def _request(self, payload: dict) -> dict:
        return {
            "text": (
                '{"summary":"華邦電未來半年目標價上看90元，法人看法相當樂觀。",'
                '"highlights":["法人持續偏多"],'
                '"facts":["模型自行推估半年目標價為90元"],'
                '"impacts":["若記憶體價格續漲有利評價提升"],'
                '"risks":["市場仍可能波動"]}'
            )
        }


class StubOpenAIPriceLevelClient(OpenAIResponsesSynthesisClient):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-key",
            model_name="gpt-4.1-mini",
            preliminary_answers_enabled=False,
        )

    def _request(self, payload: dict) -> dict:
        return {
            "text": (
                '{"summary":"台積電未來半年很有機會突破2100元。",'
                '"highlights":["營運動能強勁"],'
                '"facts":["模型推估半年內有機會站上2100元"],'
                '"impacts":["若AI需求強勁有利評價續升"],'
                '"risks":["市場仍可能波動"]}'
            )
        }


class TargetPriceQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_target_price_as_price_outlook(self) -> None:
        parsed = InputLayer().parse(QueryRequest(query="華邦電 2344 未來半年目標價是多少？"))

        self.assertEqual(parsed.question_type, "price_outlook")
        self.assertEqual(parsed.ticker, "2344")
        self.assertEqual(parsed.company_name, "華邦電")

    def test_rule_based_summary_stays_directional_without_numeric_target_price(self) -> None:
        query = make_target_price_query()
        governance_report = make_governance_report(
            make_evidence(
                "華邦電 外資維持買進評等",
                "外資看好記憶體景氣循環回升，維持買進評等，但報導未揭露具體目標價。",
                source_name="cnyes",
            ),
            make_evidence(
                "法人看好華邦電下半年營運回溫",
                "分析師指出需求改善，有助華邦電營運回升，但僅屬方向性觀點。",
                source_name="moneydj",
            ),
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("沒有直接揭露", draft.summary)
        self.assertIn("方向性訊號", draft.summary)
        self.assertIn("沒有直接給出明確目標價", draft.highlights[0])
        self.assertIn("未提供直接目標價數字", draft.facts[0])

    def test_validation_caps_context_only_target_price_answers_at_yellow(self) -> None:
        query = make_target_price_query()
        governance_report = make_governance_report(
            make_evidence(
                "華邦電 外資維持買進評等",
                "外資看好記憶體景氣循環回升，維持買進評等，但報導未揭露具體目標價。",
                source_name="cnyes",
            ),
            make_evidence(
                "法人看好華邦電下半年營運回溫",
                "分析師指出需求改善，有助華邦電營運回升，但僅屬方向性觀點。",
                source_name="moneydj",
            ),
        )
        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            draft,
        )

        self.assertEqual(result.confidence_light, ConfidenceLight.YELLOW)
        self.assertEqual(result.confidence_score, 0.55)

    def test_validation_marks_target_price_without_context_as_red(self) -> None:
        query = make_target_price_query()
        governance_report = make_governance_report(
            make_evidence(
                "華邦電第二季記憶體價格續揚",
                "報價回升帶動市場重新評估華邦電營運表現。",
                source_name="工商時報",
            ),
            make_evidence(
                "華邦電NOR Flash需求改善",
                "下游客戶拉貨回溫，市場關注毛利率修復空間。",
                source_name="經濟日報",
            ),
        )
        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            draft,
        )

        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertLessEqual(result.confidence_score, 0.25)
        self.assertIn("資料不足", draft.summary)

    def test_openai_guardrails_override_unsupported_target_price_number(self) -> None:
        query = make_target_price_query()
        governance_report = make_governance_report(
            make_evidence(
                "華邦電 外資維持買進評等",
                "外資看好記憶體景氣循環回升，維持買進評等，但報導未揭露具體目標價。",
                source_name="cnyes",
            ),
            make_evidence(
                "法人看好華邦電下半年營運回溫",
                "分析師指出需求改善，有助華邦電營運回升，但僅屬方向性觀點。",
                source_name="moneydj",
            ),
        )

        draft = StubOpenAITargetPriceClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("沒有直接揭露", draft.summary)
        self.assertNotIn("90元", draft.summary)
        self.assertIn("沒有直接給出明確目標價", draft.highlights[0])
        self.assertIn("未提供直接目標價數字", draft.facts[0])

    def test_input_layer_detects_future_price_level_question_as_price_outlook(self) -> None:
        parsed = InputLayer().parse(QueryRequest(query="台積電 2330 未來半年有機會突破2100嗎？"))

        self.assertEqual(parsed.question_type, "price_outlook")
        self.assertEqual(parsed.time_range_label, "30d")
        self.assertEqual(parsed.company_name, "台積電")

    def test_validation_marks_breakout_question_without_price_context_as_red(self) -> None:
        query = InputLayer().parse(QueryRequest(query="台積電 2330 未來半年有機會突破2100嗎？"))
        governance_report = make_governance_report(
            make_evidence(
                "台積電第一季營收創新高",
                "第一季營收表現強勁，市場持續關注AI需求帶動的先進製程動能。",
                source_name="經濟日報",
            ),
            make_evidence(
                "蘋果預約台積電SoIC產能",
                "供應鏈消息指出先進封裝需求續強，但未討論具體股價價位。",
                source_name="moneydj",
            ),
        )
        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            draft,
        )

        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertLessEqual(result.confidence_score, 0.25)
        self.assertIn("資料不足", draft.summary)

    def test_validation_caps_breakout_question_with_directional_context_at_yellow(self) -> None:
        query = InputLayer().parse(QueryRequest(query="台積電 2330 未來半年有機會突破2100嗎？"))
        governance_report = make_governance_report(
            make_evidence(
                "外資維持台積電買進評等",
                "外資維持買進評等，並提醒市場持續關注前高與整數關卡帶來的技術面壓力。",
                source_name="cnyes",
            ),
            make_evidence(
                "法人看台積電後市仍偏多",
                "分析師認為AI需求有利評價，但未直接討論特定整數關卡是否可突破。",
                source_name="moneydj",
            ),
        )
        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            draft,
        )

        self.assertEqual(result.confidence_light, ConfidenceLight.YELLOW)
        self.assertEqual(result.confidence_score, 0.55)
        self.assertIn("沒有直接討論2100元關卡", draft.summary)

    def test_openai_guardrails_override_unsupported_breakout_level(self) -> None:
        query = InputLayer().parse(QueryRequest(query="台積電 2330 未來半年有機會突破2100嗎？"))
        governance_report = make_governance_report(
            make_evidence(
                "台積電第一季營收創新高",
                "第一季營收表現強勁，市場持續關注AI需求帶動的先進製程動能。",
                source_name="經濟日報",
            ),
            make_evidence(
                "蘋果預約台積電SoIC產能",
                "供應鏈消息指出先進封裝需求續強，但未討論具體股價價位。",
                source_name="moneydj",
            ),
        )

        draft = StubOpenAIPriceLevelClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("資料不足", draft.summary)
        self.assertNotIn("很有機會突破2100元", draft.summary)
        self.assertIn("未提供可對應台積電2100元價位的前瞻證據", draft.highlights[0])


if __name__ == "__main__":
    unittest.main()
