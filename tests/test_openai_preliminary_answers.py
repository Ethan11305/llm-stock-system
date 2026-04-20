import unittest

import httpx

from llm_stock_system.adapters.openai_responses import OpenAIResponsesSynthesisClient
from llm_stock_system.core.enums import ConfidenceLight, ConsistencyStatus, FreshnessStatus, SufficiencyStatus, Topic
from llm_stock_system.core.models import DataStatus, GovernanceReport, QueryResponse, StructuredQuery
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.validation_layer import ValidationLayer


class StubOpenAIResponsesSynthesisClient(OpenAIResponsesSynthesisClient):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-key",
            model_name="gpt-4.1-mini",
            preliminary_answers_enabled=True,
        )

    def _request(self, payload: dict) -> dict:
        user_text = payload["input"][1]["content"][0]["text"]
        if "no local evidence was retrieved" in user_text:
            return {
                "text": (
                    '{"summary":"初步判讀：0050 屬於 ETF，不是單一公司，因此不適合用公司每年獲利是否穩定的方式直接判斷。",'
                    '"highlights":["0050 是 ETF，重點通常改看配息、費用率與追蹤指數表現。"],'
                    '"facts":["這段回覆不是根據本地已驗證資料生成。"],'
                    '"impacts":["若要作為退休部位，評估框架應改成基金型資產而不是企業財報。"],'
                    '"risks":["正式判斷仍需等待本地資料同步或改用 ETF 專屬指標。"]}'
                )
            }
        return {"text": '{"summary":"資料不足，無法確認。","highlights":[],"facts":[],"impacts":[],"risks":[]}'}


class FailingOpenAIResponsesSynthesisClient(OpenAIResponsesSynthesisClient):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-key",
            model_name="gpt-4.1-mini",
            preliminary_answers_enabled=True,
        )

    def _request(self, payload: dict) -> dict:
        raise httpx.ConnectError("network down")


class OpenAIPreliminaryAnswerTestCase(unittest.TestCase):
    def test_no_evidence_returns_preliminary_answer(self) -> None:
        client = StubOpenAIResponsesSynthesisClient()
        query = StructuredQuery(
            user_query="我想把0050 當成退休存股，請幫我摘要它過去五年是否每年都有穩定獲利？",
            ticker="0050",
            topic=Topic.EARNINGS,
            question_type="profitability_stability_review",
            time_range_label="5y",
            time_range_days=1825,
        )
        governance_report = GovernanceReport(
            evidence=[],
            sufficiency=SufficiencyStatus.INSUFFICIENT,
            consistency=ConsistencyStatus.CONFLICTING,
            freshness=FreshnessStatus.OUTDATED,
            high_trust_ratio=0.0,
        )

        draft = client.synthesize(query, governance_report, "system prompt")

        self.assertTrue(draft.summary.startswith("初步判讀："))
        self.assertIn("ETF", draft.summary)
        self.assertEqual(draft.sources, [])

    def test_validation_keeps_preliminary_answer_low_confidence(self) -> None:
        validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)
        query = StructuredQuery(
            user_query="我想把0050 當成退休存股，請幫我摘要它過去五年是否每年都有穩定獲利？",
            ticker="0050",
            topic=Topic.EARNINGS,
            question_type="profitability_stability_review",
            time_range_label="5y",
            time_range_days=1825,
        )
        governance_report = GovernanceReport(
            evidence=[],
            sufficiency=SufficiencyStatus.INSUFFICIENT,
            consistency=ConsistencyStatus.CONFLICTING,
            freshness=FreshnessStatus.OUTDATED,
            high_trust_ratio=0.0,
        )
        draft = StubOpenAIResponsesSynthesisClient().synthesize(query, governance_report, "system prompt")

        result = validation.validate(query, governance_report, draft)

        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertLessEqual(result.confidence_score, 0.35)
        self.assertIn(
            "Preliminary LLM answer returned without grounded local evidence.",
            result.warnings,
        )

    def test_is_preliminary_summary_recognizes_chinese_prefix(self) -> None:
        validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)
        self.assertTrue(validation._is_preliminary_summary("初步判讀：2330 近期資訊偏中性。"))
        self.assertTrue(validation._is_preliminary_summary("preliminary low-confidence answer"))
        self.assertFalse(validation._is_preliminary_summary("台積電近期表現穩定。"))

    def test_preliminary_falls_back_to_local_non_blank_answer_when_openai_fails(self) -> None:
        query = StructuredQuery(
            user_query="如果 ASML（艾司摩爾）最新展望不如預期，搜尋這對台灣半導體設備族群（如 3680 家登、6187 萬潤）的最新利空分析與情緒影響",
            ticker="3680",
            company_name="家登",
            comparison_ticker="6187",
            comparison_company_name="萬潤",
            topic=Topic.COMPOSITE,
            question_type="theme_impact_review",
            time_range_label="30d",
            time_range_days=30,
        )
        governance_report = GovernanceReport(
            evidence=[],
            sufficiency=SufficiencyStatus.INSUFFICIENT,
            consistency=ConsistencyStatus.CONFLICTING,
            freshness=FreshnessStatus.OUTDATED,
            high_trust_ratio=0.0,
        )

        draft = FailingOpenAIResponsesSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertTrue(draft.summary.startswith("初步判讀："))
        self.assertIn("家登、萬潤", draft.summary)
        self.assertEqual(draft.sources, [])

    def test_presentation_preserves_preliminary_summary_even_when_red(self) -> None:
        structured_query = StructuredQuery(
            user_query="9999 這家公司最近有什麼消息？",
            ticker="9999",
            topic=Topic.COMPOSITE,
            question_type="market_summary",
            time_range_label="7d",
            time_range_days=7,
        )
        draft = FailingOpenAIResponsesSynthesisClient()._build_local_preliminary_fallback(
            structured_query
        )
        response = PresentationLayer().present(
            structured_query,
            draft,
            GovernanceReport(
                evidence=[],
                sufficiency=SufficiencyStatus.INSUFFICIENT,
                consistency=ConsistencyStatus.CONFLICTING,
                freshness=FreshnessStatus.OUTDATED,
                high_trust_ratio=0.0,
            ),
            ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
                StructuredQuery(
                    user_query="9999 這家公司最近有什麼消息？",
                    ticker="9999",
                    topic=Topic.COMPOSITE,
                    question_type="market_summary",
                    time_range_label="7d",
                    time_range_days=7,
                ),
                GovernanceReport(
                    evidence=[],
                    sufficiency=SufficiencyStatus.INSUFFICIENT,
                    consistency=ConsistencyStatus.CONFLICTING,
                    freshness=FreshnessStatus.OUTDATED,
                    high_trust_ratio=0.0,
                ),
                draft,
            ),
        )

        self.assertTrue(response.summary.startswith("初步判讀："))


if __name__ == "__main__":
    unittest.main()
