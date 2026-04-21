from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import Intent, SourceTier
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


class GuidanceReactionQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_guidance_reaction_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="聯 發科 (2454）法說會剛結束，請摘要媒體與法人對於該公司下半年營運指引的正面與負面反應。"
            )
        )

        self.assertEqual(query.ticker, "2454")
        self.assertEqual(query.company_name, "聯發科")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(query.time_range_days, 30)

    def test_rule_based_summary_reports_mixed_guidance_reaction(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="聯 發科 (2454）法說會剛結束，請摘要媒體與法人對於該公司下半年營運指引的正面與負面反應。"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="positive-digest",
                    title="聯發科 法說後正面反應整理",
                    excerpt="聯發科 法說後與下半年營運指引相關的正面解讀共有 2 則中高可信來源。主要觀點包括：聯發科下半年 AI 與旗艦晶片動能獲法人看好。",
                    source_name="Multi-source guidance digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/positive",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=2,
                ),
                Evidence(
                    document_id="negative-digest",
                    title="聯發科 法說後負面反應整理",
                    excerpt="聯發科 法說後與下半年營運指引相關的負面解讀共有 1 則中高可信來源。主要觀點包括：部分法人對需求能見度與毛利率壓力抱持保守態度。",
                    source_name="Multi-source guidance digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/negative",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=1,
                ),
            ]
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("正負並存", draft.summary)
        self.assertEqual(draft.risks[0], "法說後的媒體與法人解讀常帶有主觀判斷，未必等同公司最終實際營運結果。")
        self.assertEqual(draft.impacts[0], "法說後的媒體與法人反應，可協助觀察市場如何解讀公司對下半年需求、毛利與出貨節奏的說法。")


if __name__ == "__main__":
    unittest.main()
