from datetime import datetime, timezone
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import GovernanceReport, Evidence, QueryRequest
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def sync_stock_info(self, force: bool = False) -> int:
        self.calls.append(("sync_stock_info", (force,)))
        return 0

    def sync_financial_statements(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_financial_statements", (ticker, start_date, end_date)))
        return 0


class MarginTurnaroundQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_margin_turnaround_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="友達 (2409）最新季報的毛利率是否由負轉正？請快速摘要其營業利益是否同步轉正，判斷是否為實質獲利改善。"
            )
        )

        self.assertEqual(query.ticker, "2409")
        self.assertEqual(query.company_name, "友達")
        self.assertEqual(query.question_type, "margin_turnaround_review")

    def test_rule_based_summary_reports_not_a_real_turnaround(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="友達 (2409）最新季報的毛利率是否由負轉正？請快速摘要其營業利益是否同步轉正，判斷是否為實質獲利改善。"
            )
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="友達 最新季毛利率觀察",
                    excerpt="截至 2025-12-31，友達 最新季營收約 701.42 億元，營業毛利約 75.25 億元，毛利率約 10.73%。上一季（2025-09-30）毛利率約 9.57%。毛利率未出現由負轉正，仍維持正值。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/gross-margin",
                    published_at=datetime(2025, 12, 31, tzinfo=timezone.utc),
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="友達 最新季營業利益觀察",
                    excerpt="截至 2025-12-31，友達 最新季營業利益約 -18.93 億元，營業利益率約 -2.70%。上一季營業利益約 -18.06 億元，營業利益率約 -2.58%。營業利益尚未同步轉正。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/operating-income",
                    published_at=datetime(2025, 12, 31, tzinfo=timezone.utc),
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="3",
                    title="友達 實質獲利改善判讀",
                    excerpt="若同時觀察毛利率與營業利益，最新季毛利率仍維持正值，但營業利益依舊為負，本業獲利修復仍不完整，目前尚難視為實質獲利改善。",
                    source_name="FinMind TaiwanStockFinancialStatements",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/profitability-view",
                    published_at=datetime(2025, 12, 31, tzinfo=timezone.utc),
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

        self.assertIn("毛利率約 10.73%", draft.summary)
        self.assertIn("營業利益約 -18.93 億元", draft.summary)
        self.assertIn("尚難視為實質獲利改善", draft.summary)

    def test_hydrator_fetches_financials_for_margin_turnaround_review(self) -> None:
        gateway = FakeGateway()
        hydrator = QueryDataHydrator(gateway)
        query = InputLayer().parse(
            QueryRequest(
                query="友達 (2409）最新季報的毛利率是否由負轉正？請快速摘要其營業利益是否同步轉正，判斷是否為實質獲利改善。"
            )
        )

        hydrator.hydrate(query)

        methods = [method_name for method_name, _ in gateway.calls]
        self.assertIn("sync_financial_statements", methods)


if __name__ == "__main__":
    unittest.main()
