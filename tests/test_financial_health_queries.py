from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


UTC_NOW = datetime(2026, 4, 7, tzinfo=timezone.utc)


class FinancialHealthQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_debt_dividend_safety_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="這支股票（如 2353 宏碁）最近股價跌很多，請幫我檢查它的負債比率是否有突然升高？公司手上的現金還夠不夠發股利？"
            )
        )

        self.assertEqual(query.ticker, "2353")
        self.assertEqual(query.company_name, "宏碁")
        self.assertEqual(query.question_type, "debt_dividend_safety_review")
        self.assertEqual(query.time_range_days, 1095)

    def test_rule_based_summary_mentions_debt_ratio_and_cash_coverage(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="這支股票（如 2353 宏碁）最近股價跌很多，請幫我檢查它的負債比率是否有突然升高？公司手上的現金還夠不夠發股利？"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="宏碁 最新資產負債表重點",
                    excerpt=(
                        "截至 2025-12-31，宏碁 資產總額約 2546.14 億元，負債總額約 1563.76 億元，"
                        "負債比率約 61.42%。現金及約當現金約 318.70 億元。"
                    ),
                    source_name="FinMind TaiwanStockBalanceSheet",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/balance-latest",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="宏碁 負債比率變化",
                    excerpt=(
                        "宏碁 最新負債比率約 61.42%。前一季約 62.35%。去年同期約 61.92%。"
                        "近 8 季區間約 61.42% 至 66.38%。若與最近幾期相比，負債比率未見突然升高，大致持平。"
                    ),
                    source_name="FinMind TaiwanStockBalanceSheet",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/balance-trend",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="3",
                    title="宏碁 現金股利支應能力",
                    excerpt=(
                        "宏碁 最新可取得的現金股利年度為 2025 年，現金股利每股約 1.700 元，"
                        "現金股利發放總額約 51.81 億元。若以最新資產負債表的現金及約當現金約 318.70 億元估算，"
                        "約可覆蓋 6.15 倍。若只看帳上現金，現金部位看起來足以支應現金股利。"
                    ),
                    source_name="FinMind TaiwanStockBalanceSheet x TaiwanStockDividend",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/cash-coverage",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=1.0,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("負債比率約 61.42%", draft.summary)
        self.assertIn("未見突然升高", draft.summary)
        self.assertIn("約可覆蓋 6.15 倍", draft.summary)


if __name__ == "__main__":
    unittest.main()
