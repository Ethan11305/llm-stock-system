from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import Intent, SourceTier
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


class ElectricityCostImpactQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_electricity_cost_impact_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query=(
                    "政府宣佈工業電價將調漲 10% 以上，請過濾出「用電大戶」個股"
                    "（如 2002 中鋼、1101 台泥）可能面臨的成本增加額度與因應對策。"
                )
            )
        )

        self.assertEqual(query.ticker, "2002")
        self.assertEqual(query.company_name, "中鋼")
        self.assertEqual(query.comparison_ticker, "1101")
        self.assertEqual(query.comparison_company_name, "台泥")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(query.time_range_days, 30)

    def test_rule_based_summary_reports_cost_pressure_and_responses(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query=(
                    "政府宣佈工業電價將調漲 10% 以上，請過濾出「用電大戶」個股"
                    "（如 2002 中鋼、1101 台泥）可能面臨的成本增加額度與因應對策。"
                )
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="cost-digest",
                    title="中鋼、台泥 電價成本壓力摘要",
                    excerpt=(
                        "中鋼、台泥 近期與工業電價調整相關的新聞重點顯示，"
                        "市場普遍把電價調漲視為高耗電產業的成本壓力來源，可能影響毛利與獲利彈性。"
                        "目前多數報導仍停留在方向性壓力描述，未充分揭露單一公司可精算的成本增加額度。"
                    ),
                    source_name="Multi-source electricity digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/electricity/cost",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=2,
                ),
                Evidence(
                    document_id="response-digest",
                    title="中鋼、台泥 電價調漲因應對策摘要",
                    excerpt=(
                        "中鋼、台泥 目前較常見的因應方向包括節能與降耗、售價或報價轉嫁，"
                        "但實際落地幅度仍要看公司正式說明。"
                    ),
                    source_name="Multi-source electricity digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/electricity/response",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=2,
                ),
            ]
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("中鋼與台泥", draft.summary)
        self.assertIn("成本壓力來源", draft.summary)
        self.assertIn("因應方向", draft.summary)
        self.assertEqual(
            draft.impacts[0],
            "工業電價調漲通常會先影響高耗電產業的成本結構，再慢慢反映到毛利率與報價策略。",
        )
        self.assertEqual(
            draft.risks[0],
            "電價調漲帶來的成本壓力不一定能完整轉嫁，對毛利率的影響仍要看產品報價與景氣狀況。",
        )


if __name__ == "__main__":
    unittest.main()
