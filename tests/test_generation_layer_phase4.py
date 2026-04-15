import unittest
from datetime import datetime

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.openai_responses import OpenAIResponsesSynthesisClient
from llm_stock_system.core.enums import (
    ConsistencyStatus,
    FreshnessStatus,
    Intent,
    SourceTier,
    SufficiencyStatus,
    TopicTag,
)
from llm_stock_system.core.models import Evidence, GovernanceReport, StructuredQuery


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


def make_report(*evidence: Evidence) -> GovernanceReport:
    return GovernanceReport(
        evidence=list(evidence),
        sufficiency=SufficiencyStatus.SUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT,
        freshness=FreshnessStatus.RECENT,
        high_trust_ratio=0.9,
    )


class CapturingOpenAIClient(OpenAIResponsesSynthesisClient):
    def __init__(self, response_text: str) -> None:
        super().__init__(
            api_key="test-key",
            model_name="gpt-4.1-mini",
            preliminary_answers_enabled=False,
        )
        self._response_text = response_text
        self.last_payload: dict | None = None

    def _request(self, payload: dict) -> dict:
        self.last_payload = payload
        return {"text": self._response_text}


class GenerationLayerPhase4TestCase(unittest.TestCase):
    def test_news_digest_routes_by_shipping_tag_without_question_type_dependency(self) -> None:
        query = StructuredQuery(
            user_query="長榮航運近期還會受惠 SCFI 與紅海事件嗎？",
            ticker="2603",
            company_name="長榮",
            intent=Intent.NEWS_DIGEST,
            controlled_tags=[TopicTag.SHIPPING],
            question_type="market_summary",
        )
        report = make_report(
            make_evidence("SCFI 反彈", "紅海航線受阻帶動 SCFI 反彈，市場關注現貨運價續航。"),
            make_evidence("法人調升評等", "分析師指出目標價上修，外資看好運價支撐延續。"),
        )

        draft = RuleBasedSynthesisClient().synthesize(query, report, "")

        self.assertIn("SCFI", draft.summary)
        self.assertIn("紅海", draft.summary)

    def test_earnings_review_routes_by_gross_margin_tag(self) -> None:
        query = StructuredQuery(
            user_query="台積電毛利率是否已轉正？",
            ticker="2330",
            company_name="台積電",
            intent=Intent.EARNINGS_REVIEW,
            controlled_tags=[TopicTag.GROSS_MARGIN],
            question_type="earnings_summary",
        )
        report = make_report(
            make_evidence(
                "毛利率回升",
                "毛利率約 12.3%。毛利率已由負轉正。最新季營業利益約 4.5 億元。營業利益已同步轉正。"
                "上一季（2025-12-31）毛利率約 -3.2%。上一季營業利益約 -1.1 億元。"
                "最新季毛利率已由負轉正，且營業利益也同步轉正，目前可視為本業層面的實質獲利改善。",
            )
        )

        draft = RuleBasedSynthesisClient().synthesize(query, report, "")

        self.assertIn("毛利率約 12.3%", draft.summary)
        self.assertIn("營業利益約 4.5 億元", draft.summary)

    def test_openai_target_price_guardrail_uses_intent_instead_of_question_type(self) -> None:
        query = StructuredQuery(
            user_query="華邦電 2344 未來半年目標價是多少？",
            ticker="2344",
            company_name="華邦電",
            intent=Intent.VALUATION_CHECK,
            controlled_tags=[TopicTag.VALUATION],
            question_type="pe_valuation_review",
        )
        report = make_report(
            make_evidence("法人目標價更新", "法人將華邦電目標價由 120 元上修至 138 元，維持買進評等。"),
        )
        client = CapturingOpenAIClient(
            '{"summary":"華邦電目標價上看999元。","highlights":["模型自估目標價999元"],'
            '"facts":["模型猜測999元"],"impacts":["正面"],"risks":["波動"]}'
        )

        draft = client.synthesize(query, report, "system prompt")

        self.assertNotIn("999", draft.summary)
        self.assertIn("120", draft.summary)
        self.assertIn("目標價", draft.highlights[0])

    def test_dividend_analysis_routes_by_cash_flow_tag(self) -> None:
        query = StructuredQuery(
            user_query="中華電自由現金流能支撐股利嗎？",
            ticker="2412",
            company_name="中華電",
            intent=Intent.DIVIDEND_ANALYSIS,
            controlled_tags=[TopicTag.CASH_FLOW],
            question_type="dividend_yield_review",
        )
        report = make_report(
            make_evidence(
                "自由現金流與股利",
                "2023 年營業活動淨現金流入約 520 億元，資本支出約 180 億元，推估自由現金流約 340 億元。"
                "2023 年現金股利每股約 4.8 元。依參與分派總股數約 77.0 股估算，現金股利發放總額約 295 億元。"
                "近三年自由現金流均高於現金股利支出，顯示目前股利政策具一定永續性。",
            )
        )

        draft = RuleBasedSynthesisClient().synthesize(query, report, "")

        self.assertIn("自由現金流", draft.summary)
        self.assertIn("股利政策具一定永續性", draft.summary)

    def test_technical_view_routes_by_margin_flow_tag(self) -> None:
        query = StructuredQuery(
            user_query="聯發科跌破季線後融資籌碼怎麼看？",
            ticker="2454",
            company_name="聯發科",
            intent=Intent.TECHNICAL_VIEW,
            controlled_tags=[TopicTag.MARGIN_FLOW],
            question_type="technical_indicator_review",
        )
        report = make_report(
            make_evidence(
                "季線與融資",
                "最新收盤價約 1220 元。季線(MA60)約 1250 元。近期跌破季線。"
                "最新融資餘額約 15300 張。融資使用率約 21.5%。相較近 20 日平均變動約 8.2%。籌碼面屬偏高。",
            )
        )

        draft = RuleBasedSynthesisClient().synthesize(query, report, "")

        self.assertIn("季線", draft.summary)
        self.assertIn("融資餘額約 15300 張", draft.summary)

    def test_openai_user_prompt_uses_intent_and_topic_tags_without_question_type(self) -> None:
        query = StructuredQuery(
            user_query="長榮近期還會受惠 SCFI 嗎？",
            ticker="2603",
            company_name="長榮",
            intent=Intent.NEWS_DIGEST,
            controlled_tags=[TopicTag.SHIPPING],
            question_type="shipping_rate_impact_review",
        )
        report = make_report(make_evidence("SCFI 反彈", "紅海航線受阻帶動 SCFI 反彈。"))
        client = CapturingOpenAIClient(
            '{"summary":"資料不足，無法確認。","highlights":[],"facts":[],"impacts":[],"risks":[]}'
        )

        client.synthesize(query, report, "system prompt")
        user_text = client.last_payload["input"][1]["content"][0]["text"]  # type: ignore[index]

        self.assertIn("Intent: news_digest", user_text)
        self.assertIn("Topic tags: 航運", user_text)
        self.assertNotIn("Question type:", user_text)

    def test_legacy_question_type_backfills_topic_tags_for_phase4_routing(self) -> None:
        query = StructuredQuery(
            user_query="台積電毛利率是否轉正？",
            ticker="2330",
            company_name="台積電",
            question_type="margin_turnaround_review",
        )
        report = make_report(
            make_evidence(
                "毛利率回升",
                "毛利率約 12.3%。毛利率已由負轉正。最新季營業利益約 4.5 億元。營業利益已同步轉正。"
                "上一季（2025-12-31）毛利率約 -3.2%。上一季營業利益約 -1.1 億元。",
            )
        )

        draft = RuleBasedSynthesisClient().synthesize(query, report, "")

        self.assertEqual(query.intent, Intent.EARNINGS_REVIEW)
        self.assertIn("毛利率", query.topic_tags)
        self.assertIn("毛利率約 12.3%", draft.summary)


if __name__ == "__main__":
    unittest.main()
