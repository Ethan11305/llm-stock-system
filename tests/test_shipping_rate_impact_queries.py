from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import Intent, SourceTier
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


class ShippingRateImpactQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_shipping_rate_impact_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query=(
                    "新聞指出紅海航線受阻加劇，摘要這對「2603 長榮」或「2609 陽明」"
                    "運價指數 (SCFI) 的支撐力道與分析師的目標價調整。"
                )
            )
        )

        self.assertEqual(query.ticker, "2603")
        self.assertEqual(query.company_name, "長榮")
        self.assertEqual(query.comparison_ticker, "2609")
        self.assertEqual(query.comparison_company_name, "陽明")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(query.time_range_days, 30)

    def test_rule_based_summary_reports_shipping_rate_and_target_price_view(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query=(
                    "新聞指出紅海航線受阻加劇，摘要這對「2603 長榮」或「2609 陽明」"
                    "運價指數 (SCFI) 的支撐力道與分析師的目標價調整。"
                )
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="shipping-support",
                    title="長榮、陽明 紅海與 SCFI 支撐摘要",
                    excerpt=(
                        "長榮、陽明 近期與紅海航線及 SCFI 相關的新聞重點顯示，"
                        "市場多把紅海繞道、航線受阻與運力吃緊，視為短線支撐 SCFI 與現貨運價的主要因素。"
                        "不過報導也提醒，支撐力道能否延續，仍要看 SCFI 是否續彈與塞港因素是否持續。"
                    ),
                    source_name="Multi-source shipping digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/shipping/support",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=2,
                ),
                Evidence(
                    document_id="shipping-target",
                    title="長榮、陽明 法人目標價調整摘要",
                    excerpt=(
                        "長榮、陽明 近期分析師與法人報導多圍繞目標價、評等與運價續航解讀；"
                        "整體反應正負解讀並存。"
                    ),
                    source_name="Multi-source shipping digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/shipping/target",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=2,
                ),
            ]
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("長榮與陽明", draft.summary)
        self.assertIn("SCFI", draft.summary)
        self.assertIn("目標價調整仍偏分歧", draft.summary)
        self.assertEqual(
            draft.impacts[0],
            "紅海航線受阻與 SCFI 變化，常是貨櫃航運股短線評價最直接的事件指標。",
        )
        self.assertEqual(
            draft.risks[0],
            "SCFI 與紅海事件多屬短線催化，未必能直接等同全年獲利或長期運價中樞上移。",
        )


if __name__ == "__main__":
    unittest.main()
