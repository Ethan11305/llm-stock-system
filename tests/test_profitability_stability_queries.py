import unittest
from datetime import datetime, timezone

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import ConfidenceLight, ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import Evidence, GovernanceReport, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.validation_layer import ValidationLayer


class ProfitabilityStabilityQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_profitability_stability_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="台化 (1326)當成退休存股，請幫我摘要它過去五年是否每年都有穩定獲利？有沒有哪一年是突然大虧損的？原因是什麼？"
            )
        )

        self.assertEqual(query.ticker, "1326")
        self.assertEqual(query.question_type, "profitability_stability_review")
        self.assertEqual(query.time_range_label, "5y")

    def test_rule_based_summary_mentions_loss_year_and_reason(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="台化 (1326)當成退休存股，請幫我摘要它過去五年是否每年都有穩定獲利？有沒有哪一年是突然大虧損的？原因是什麼？"
            )
        )
        published_at = datetime(2025, 12, 31, tzinfo=timezone.utc)
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="台化 近五年年度獲利",
                    excerpt="台化 近五個完整年度的獲利表現為：2021 年 獲利，歸屬母公司淨利約 250.00 億元，EPS 約 4.10 元；2022 年 獲利，歸屬母公司淨利約 180.00 億元，EPS 約 2.95 元；2023 年 獲利，歸屬母公司淨利約 65.00 億元，EPS 約 1.05 元；2024 年 虧損，歸屬母公司淨利約 -10.67 億元，EPS 約 -0.19 元；2025 年 虧損，歸屬母公司淨利約 -2.95 億元，EPS 約 -0.05 元。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/profitability-history",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="台化 獲利穩定性評估",
                    excerpt="若以 2021 至 2025 年的年度財報觀察，台化近五年並非每年都有穩定獲利。近五年中有 3 年為正獲利，2024 年轉虧；2025 年轉虧。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/stability",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="3",
                    title="台化 2024 年虧損原因",
                    excerpt="台化 在 2024 年歸屬母公司淨損約 10.67 億元，EPS 約 -0.19 元。若與前一年相比，營收年增率約 -18.40%。若只看財報結構推估，主因較像本業轉弱，營業利益已轉為損失約 -8.67 億元，且營業外也未提供明顯支撐。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/loss-reason",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.OUTDATED,
            high_trust_ratio=1.0,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("近五年並非每年都有穩定獲利", draft.summary)
        self.assertIn("2024 年是較明顯的虧損年度", draft.summary)
        self.assertIn("財報結構推估", draft.summary)

    def test_validation_caps_inferred_reason_without_news(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="台化 (1326)當成退休存股，請幫我摘要它過去五年是否每年都有穩定獲利？有沒有哪一年是突然大虧損的？原因是什麼？"
            )
        )
        published_at = datetime(2025, 12, 31, tzinfo=timezone.utc)
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="台化 近五年年度獲利",
                    excerpt="台化 近五個完整年度的獲利表現為：2021 年 獲利，歸屬母公司淨利約 250.00 億元，EPS 約 4.10 元；2022 年 獲利，歸屬母公司淨利約 180.00 億元，EPS 約 2.95 元；2023 年 獲利，歸屬母公司淨利約 65.00 億元，EPS 約 1.05 元；2024 年 虧損，歸屬母公司淨利約 -10.67 億元，EPS 約 -0.19 元；2025 年 虧損，歸屬母公司淨利約 -2.95 億元，EPS 約 -0.05 元。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/profitability-history",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="台化 獲利穩定性評估",
                    excerpt="若以 2021 至 2025 年的年度財報觀察，台化近五年並非每年都有穩定獲利。近五年中有 3 年為正獲利，2024 年轉虧；2025 年轉虧。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/stability",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="3",
                    title="台化 2024 年虧損原因",
                    excerpt="台化 在 2024 年歸屬母公司淨損約 10.67 億元，EPS 約 -0.19 元。若只看財報結構推估，主因較像本業轉弱，營業利益已轉為損失約 -8.67 億元，且營業外也未提供明顯支撐。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/loss-reason",
                    published_at=published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.OUTDATED,
            high_trust_ratio=1.0,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")
        result = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55).validate(
            query,
            governance_report,
            draft,
        )

        self.assertEqual(result.confidence_light, ConfidenceLight.YELLOW)
        self.assertLessEqual(result.confidence_score, 0.75)


if __name__ == "__main__":
    unittest.main()
