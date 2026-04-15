from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import (
    ConfidenceLight,
    ConsistencyStatus,
    FreshnessStatus,
    SourceTier,
    SufficiencyStatus,
)
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.validation_layer import ValidationLayer


UTC_NOW = datetime(2026, 4, 7, tzinfo=timezone.utc)


class FCFDividendSustainabilityTestCase(unittest.TestCase):
    def test_input_layer_detects_fcf_dividend_sustainability_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="中華電(2412)過去三年的自由現金流(FCF)與現金股利發放總額是多少，摘要其目前的股利政策在未來是否具有永續性？"
            )
        )

        self.assertEqual(query.ticker, "2412")
        self.assertEqual(query.question_type, "fcf_dividend_sustainability_review")
        self.assertEqual(query.time_range_days, 1095)

    def test_rule_based_summary_mentions_fcf_and_dividend_totals(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="中華電(2412)過去三年的自由現金流(FCF)與現金股利發放總額是多少，摘要其目前的股利政策在未來是否具有永續性？"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="fcf-2022",
                    title="中華電信 2022 自由現金流",
                    excerpt="根據 FinMind TaiwanStockCashFlowsStatement，中華電信 2022 年營業活動淨現金流入約 759.51 億元，資本支出約 315.35 億元，推估自由現金流約 444.16 億元。",
                    source_name="FinMind TaiwanStockCashFlowsStatement",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/fcf-2022",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="fcf-2023",
                    title="中華電信 2023 自由現金流",
                    excerpt="根據 FinMind TaiwanStockCashFlowsStatement，中華電信 2023 年營業活動淨現金流入約 745.60 億元，資本支出約 307.41 億元，推估自由現金流約 438.18 億元。",
                    source_name="FinMind TaiwanStockCashFlowsStatement",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/fcf-2023",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="dividend-2022",
                    title="中華電信 2022 現金股利發放總額",
                    excerpt="中華電信 2022 年現金股利每股約 4.702 元。依參與分派總股數約 7757446545 股估算，現金股利發放總額約 364.75 億元。",
                    source_name="FinMind TaiwanStockDividend",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/dividend-2022",
                    published_at=UTC_NOW,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="sustainability",
                    title="中華電信 股利政策永續性評估",
                    excerpt="以目前已揭露且可對齊的 2022 至 2024 年資料觀察，自由現金流對現金股利發放總額的覆蓋倍數分別為：2022 年約 1.22 倍、2023 年約 1.19 倍、2024 年約 1.37 倍。近三年大致能以自由現金流支應現金股利，整體永續性偏穩健。",
                    source_name="FinMind TaiwanStockCashFlowsStatement x TaiwanStockDividend",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/sustainability",
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

        self.assertIn("自由現金流", draft.summary)
        self.assertIn("現金股利發放總額", draft.summary)
        self.assertIn("永續性偏穩健", draft.summary)

    def test_validation_requires_cash_flow_and_dividend_evidence(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="中華電(2412)過去三年的自由現金流(FCF)與現金股利發放總額是多少，摘要其目前的股利政策在未來是否具有永續性？"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="fcf-only",
                    title="中華電信 2024 自由現金流",
                    excerpt="根據 FinMind TaiwanStockCashFlowsStatement，中華電信 2024 年營業活動淨現金流入約 792.44 億元，資本支出約 287.56 億元，推估自由現金流約 504.89 億元。",
                    source_name="FinMind TaiwanStockCashFlowsStatement",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/fcf-only",
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
        answer_draft = AnswerDraft(
            summary="中華電信目前只取得部分自由現金流資料，仍不足以完整評估股利政策永續性。",
            highlights=[],
            facts=[],
            impacts=[],
            risks=[],
            sources=[],
        )

        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            answer_draft,
        )

        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertIn(
            "FCF dividend sustainability review: missing one of cash flow or dividend evidence.",
            result.warnings,
        )


if __name__ == "__main__":
    unittest.main()