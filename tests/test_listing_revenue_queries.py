from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import Intent, SourceTier
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


class ListingRevenueQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_listing_revenue_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="星 宇航空 (2646) 轉上市後的股價波動原因，是否有重大營收增長消息？"
            )
        )

        self.assertEqual(query.ticker, "2646")
        self.assertEqual(query.company_name, "星宇航空")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(query.time_range_days, 30)

    def test_input_layer_detects_ipo_variant_of_listing_template(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="想看 2646 星宇航空 IPO 後股價波動原因，近期有沒有營收利多消息？"
            )
        )

        self.assertEqual(query.ticker, "2646")
        self.assertEqual(query.company_name, "星宇航空")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)

    def test_rule_based_summary_mentions_revenue_signal_and_news_focus(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="星 宇航空 (2646) 轉上市後的股價波動原因，是否有重大營收增長消息？"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="revenue-doc",
                    title="星宇航空 最新月營收增長訊號",
                    excerpt="星宇航空 目前最新已公布月營收為 2026-02，單月營收約 48.18 億元。月增率約 14.31%。年增率約 41.50%。若以年增率觀察，這屬偏強的營收成長訊號。",
                    source_name="TWSE OpenAPI Monthly Revenue",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/revenue",
                    published_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="news-doc",
                    title="星宇航空收燃油附加費 短程45美元、長程117美元",
                    excerpt="經濟日報指出，星宇航空近期焦點之一為燃油附加費調整。",
                    source_name="經濟日報",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/news",
                    published_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=1,
                ),
            ]
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("星宇航空", draft.summary)
        self.assertIn("2026-02", draft.summary)
        self.assertIn("41.50%", draft.summary)
        self.assertIn("偏強的營收成長訊號", draft.summary)
        self.assertIn("燃油附加費", draft.summary)
        self.assertEqual(draft.risks[0], "轉上市初期的價格波動可能夾雜籌碼與情緒因素，未必完全反映中長期基本面。")

    def test_rule_based_summary_uses_ipo_label_when_query_mentions_ipo(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="想看 2646 星宇航空 IPO 後股價波動原因，近期有沒有營收利多消息？"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="revenue-doc",
                    title="星宇航空 最新月營收增長訊號",
                    excerpt="星宇航空 目前最新已公布月營收為 2026-02，單月營收約 48.18 億元。年增率約 41.50%。",
                    source_name="TWSE OpenAPI Monthly Revenue",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/revenue",
                    published_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
                    support_score=1.0,
                    corroboration_count=1,
                )
            ]
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("IPO 後", draft.summary)


if __name__ == "__main__":
    unittest.main()
