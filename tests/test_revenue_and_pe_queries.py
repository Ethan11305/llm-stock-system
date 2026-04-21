"""Revenue YoY 路徑的回歸測試。

Wave 2 sunset：PE / 估值相關路徑已移除（ValuationCheckStrategy 下架），
僅保留 monthly_revenue_yoy_review 這條仍活著的路徑。
"""

from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, Intent, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


UTC_NOW = datetime(2026, 4, 7, tzinfo=timezone.utc)


class RevenueYoYQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_monthly_revenue_yoy_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="最近新聞說台積電(2330)營收亮眼，請幫我對比它今年前三個月的累計營收，跟去年同期相比成長了百分之幾？"
            )
        )

        self.assertEqual(query.ticker, "2330")
        self.assertEqual(query.intent, Intent.EARNINGS_REVIEW)
        self.assertEqual(query.time_range_days, 365)

    def test_rule_based_summary_mentions_revenue_yoy(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="最近新聞說台積電(2330)營收亮眼，請幫我對比它今年前三個月的累計營收，跟去年同期相比成長了百分之幾？"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="revenue-yoy",
                    title="台積電 2026 年前 3 個月累計營收",
                    excerpt="台積電 2026 年前 3 個月累計營收約 8959.65 億元；2025 年同期約 5928.89 億元；年增率約 51.12%。",
                    source_name="TWSE IIH Company Financial",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/revenue-yoy",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                )
            ],
            sufficiency=SufficiencyStatus.INSUFFICIENT,
            consistency=ConsistencyStatus.CONFLICTING,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=1.0,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("今年前 3 個月累計營收", draft.summary)
        self.assertIn("年增率約 51.12%", draft.summary)


if __name__ == "__main__":
    unittest.main()
