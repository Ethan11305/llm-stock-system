from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer


UTC_NOW = datetime(2026, 4, 7, tzinfo=timezone.utc)


class RevenueAndPEQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_monthly_revenue_yoy_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="最近新聞說台積電(2330)營收亮眼，請幫我對比它今年前三個月的累計營收，跟去年同期相比成長了百分之幾？"
            )
        )

        self.assertEqual(query.ticker, "2330")
        self.assertEqual(query.question_type, "monthly_revenue_yoy_review")
        self.assertEqual(query.time_range_days, 365)

    def test_input_layer_detects_pe_valuation_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="我想長期投資中華電(2412)，請問它目前的本益比(P/E Ratio)處於歷史的高位還是低位？現在進場算不算買貴了？"
            )
        )

        self.assertEqual(query.ticker, "2412")
        self.assertEqual(query.question_type, "pe_valuation_review")
        # 查詢含「歷史」關鍵字，語意為「本益比的歷史分位比較」，
        # 需要足夠長的歷史區間才能判斷高低位，5y（1825d）比 policy 預設 1y 更符合語意。
        # 若使用者未提及任何時間關鍵字，才由 policy 預設（1y）填入。
        self.assertEqual(query.time_range_days, 1825)

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

    def test_rule_based_summary_mentions_pe_zone(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="我想長期投資中華電(2412)，請問它目前的本益比(P/E Ratio)處於歷史的高位還是低位？現在進場算不算買貴了？"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="pe-current",
                    title="中華電信 目前本益比",
                    excerpt="截至 2026-03，中華電信 本益比約 26.57 倍。同業本益比約 74.40 倍。",
                    source_name="TWSE IIH Company Financial",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/pe-current",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="pe-history",
                    title="中華電信 近 13 個月本益比區間",
                    excerpt="若以近 13 個月月資料觀察，本益比區間約 17.20 至 30.80 倍，中位數約 22.10 倍，25 分位約 19.80 倍，75 分位約 25.40 倍。目前約落在近 13 個月歷史分位 82.0% 左右，屬歷史偏高區。",
                    source_name="TWSE IIH Company Financial",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/pe-history",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="pe-assessment",
                    title="中華電信 估值進場評估",
                    excerpt="若以近 13 個月歷史本益比區間衡量，中華電信 目前估值屬歷史偏高區，對長期投資來說，現在進場的估值不算便宜，需接受評價偏高的風險。",
                    source_name="TWSE IIH Company Financial",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/pe-assessment",
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

        self.assertIn("本益比約 26.57 倍", draft.summary)
        self.assertIn("歷史偏高區", draft.summary)


if __name__ == "__main__":
    unittest.main()
