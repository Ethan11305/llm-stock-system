import unittest
from datetime import datetime, timezone

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import ConfidenceLight, ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.validation_layer import ValidationLayer


class ComparisonQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_gross_margin_comparison_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="我想比較長榮 (2603) 跟陽明 (2609)，這兩家公司誰的毛利率比較高？這代表哪一家的經營效率比較好？"
            )
        )

        self.assertEqual(query.ticker, "2603")
        self.assertEqual(query.company_name, "長榮")
        self.assertEqual(query.comparison_ticker, "2609")
        self.assertEqual(query.comparison_company_name, "陽明")
        self.assertEqual(query.question_type, "gross_margin_comparison_review")
        self.assertEqual(query.time_range_label, "latest_quarter")

    def test_rule_based_summary_compares_two_companies(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="我想比較長榮 (2603) 跟陽明 (2609)，這兩家公司誰的毛利率比較高？這代表哪一家的經營效率比較好？"
            )
        )
        published_at = datetime(2025, 12, 31, tzinfo=timezone.utc)
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="長榮 最新毛利率",
                    excerpt="截至 2025-12-31，長榮 營業收入約 4635.20 億元，營業毛利約 1183.04 億元，毛利率約 25.52%。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/2603/latest",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="陽明 最新毛利率",
                    excerpt="截至 2025-12-31，陽明 營業收入約 2760.15 億元，營業毛利約 552.03 億元，毛利率約 20.00%。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/2609/latest",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="3",
                    title="長榮 vs 陽明 毛利率比較",
                    excerpt=(
                        "若以 2025-12-31 最新可比口徑比較，長榮 毛利率約 25.52%，陽明 毛利率約 20.00%，"
                        "由 長榮 較高，高出 5.52 個百分點。單看毛利率，長榮 在定價能力或成本結構上相對占優勢；"
                        "但經營效率仍需搭配營益率、費用率與資產周轉一起看。"
                    ),
                    source_name="FinMind TaiwanStockFinancialStatements Comparison",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/comparison",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="4",
                    title="長榮 vs 陽明 近 4 季毛利率",
                    excerpt="若看最近 4 個可比較季度，長榮 平均毛利率約 24.80%，陽明 平均毛利率約 19.70%。",
                    source_name="FinMind TaiwanStockFinancialStatements Comparison",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/trend",
                    published_at=published_at,
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

        self.assertIn("長榮 毛利率約 25.52%", draft.summary)
        self.assertIn("陽明 毛利率約 20.00%", draft.summary)
        self.assertIn("由 長榮 較高", draft.summary)
        self.assertIn("經營效率更好", draft.summary)

    def test_validation_accepts_comparison_with_financial_evidence(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="我想比較長榮 (2603) 跟陽明 (2609)，這兩家公司誰的毛利率比較高？這代表哪一家的經營效率比較好？"
            )
        )
        published_at = datetime(2025, 12, 31, tzinfo=timezone.utc)
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="長榮 最新毛利率",
                    excerpt="截至 2025-12-31，長榮 營業收入約 4635.20 億元，營業毛利約 1183.04 億元，毛利率約 25.52%。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/2603/latest",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="陽明 最新毛利率",
                    excerpt="截至 2025-12-31，陽明 營業收入約 2760.15 億元，營業毛利約 552.03 億元，毛利率約 20.00%。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/2609/latest",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="3",
                    title="長榮 vs 陽明 毛利率比較",
                    excerpt=(
                        "若以 2025-12-31 最新可比口徑比較，長榮 毛利率約 25.52%，陽明 毛利率約 20.00%，"
                        "由 長榮 較高，高出 5.52 個百分點。"
                    ),
                    source_name="FinMind TaiwanStockFinancialStatements Comparison",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/comparison",
                    published_at=published_at,
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
        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            draft,
        )

        self.assertIn(result.confidence_light, {ConfidenceLight.GREEN, ConfidenceLight.YELLOW})
        self.assertGreaterEqual(result.confidence_score, 0.55)


if __name__ == "__main__":
    unittest.main()
