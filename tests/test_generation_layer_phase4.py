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
        )
        report = make_report(
            make_evidence("SCFI 反彈", "紅海航線受阻帶動 SCFI 反彈，市場關注現貨運價續航。"),
            make_evidence("法人調升評等", "分析師指出目標價上修，外資看好運價支撐延續。"),
        )

        draft = RuleBasedSynthesisClient().synthesize(query, report, "")

        self.assertIn("SCFI", draft.summary)
        self.assertIn("紅海", draft.summary)

    def test_openai_user_prompt_uses_intent_and_topic_tags_without_question_type(self) -> None:
        query = StructuredQuery(
            user_query="長榮近期還會受惠 SCFI 嗎？",
            ticker="2603",
            company_name="長榮",
            intent=Intent.NEWS_DIGEST,
            controlled_tags=[TopicTag.SHIPPING],
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


if __name__ == "__main__":
    unittest.main()
