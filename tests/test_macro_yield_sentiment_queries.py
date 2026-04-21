from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import Intent, SourceTier
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


class MacroYieldSentimentQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_macro_yield_sentiment_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query=(
                    "對於美國公布的 CPI，摘要這對高殖利率個股（如 2412 中華電、金控股）"
                    "的負面情緒影響與法人的最新觀點。"
                )
            )
        )

        self.assertEqual(query.ticker, "2412")
        self.assertEqual(query.company_name, "中華電信")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(query.time_range_days, 30)

    def test_rule_based_summary_reports_macro_yield_sentiment(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query=(
                    "對於美國公布的 CPI，摘要這對高殖利率個股（如 2412 中華電、金控股）"
                    "的負面情緒影響與法人的最新觀點。"
                )
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="macro-digest",
                    title="中華電信 高殖利率情緒摘要",
                    excerpt=(
                        "中華電信 相關的高殖利率與總經新聞重點顯示，美國 CPI 若偏熱，"
                        "市場通常會把它解讀為利率下修延後或債息上行壓力，進而壓抑高殖利率股情緒。"
                        "這種情緒通常也會延伸到金控股與其他防禦型高殖利率標的。整體解讀偏保守或觀望。"
                    ),
                    source_name="Multi-source macro digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/macro/sentiment",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=2,
                ),
                Evidence(
                    document_id="view-digest",
                    title="中華電信 法人最新觀點摘要",
                    excerpt=(
                        "中華電信 相關法人與外資最新觀點多圍繞利率走勢、債息變化與防禦型現金流評價；"
                        "目前整體觀點偏保守。"
                    ),
                    source_name="Multi-source macro digest",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/macro/view",
                    published_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
                    support_score=0.75,
                    corroboration_count=2,
                ),
            ]
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "system prompt")

        self.assertIn("中華電信", draft.summary)
        self.assertIn("高殖利率股情緒", draft.summary)
        self.assertIn("法人最新觀點偏向保守", draft.summary)
        self.assertEqual(
            draft.impacts[0],
            "美國 CPI 與通膨預期，常會先透過利率與債息路徑影響高殖利率股的評價。",
        )
        self.assertEqual(
            draft.risks[0],
            "CPI 對高殖利率股的影響常先反映在評價與情緒面，不一定立刻反映到基本面。",
        )


if __name__ == "__main__":
    unittest.main()
