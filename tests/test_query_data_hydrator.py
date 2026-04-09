from datetime import datetime, timezone
from pathlib import Path
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.repositories import InMemoryQueryLogStore
from llm_stock_system.core.enums import SourceTier, Topic
from llm_stock_system.core.models import Document, QueryRequest
from llm_stock_system.layers.data_governance_layer import DataGovernanceLayer
from llm_stock_system.layers.generation_layer import GenerationLayer
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.retrieval_layer import RetrievalLayer
from llm_stock_system.layers.validation_layer import ValidationLayer
from llm_stock_system.orchestrator.pipeline import QueryPipeline
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def sync_stock_info(self, force: bool = False) -> int:
        self.calls.append(("sync_stock_info", (force,)))
        return 0

    def sync_price_history(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_price_history", (ticker, start_date, end_date)))
        return 0

    def sync_financial_statements(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_financial_statements", (ticker, start_date, end_date)))
        return 0

    def sync_balance_sheet_items(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_balance_sheet_items", (ticker, start_date, end_date)))
        return 0

    def sync_cash_flow_statements(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_cash_flow_statements", (ticker, start_date, end_date)))
        return 0

    def sync_dividend_policies(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_dividend_policies", (ticker, start_date, end_date)))
        return 0

    def sync_stock_news(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_stock_news", (ticker, start_date, end_date)))
        return 0

    def sync_margin_purchase_short_sale(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_margin_purchase_short_sale", (ticker, start_date, end_date)))
        return 0

    def sync_monthly_revenue_points(self, ticker) -> int:
        self.calls.append(("sync_monthly_revenue_points", (ticker,)))
        return 0

    def sync_pe_valuation_points(self, ticker) -> int:
        self.calls.append(("sync_pe_valuation_points", (ticker,)))
        return 0


class HydrationAwareRepository:
    def __init__(self, hydrator) -> None:
        self._hydrator = hydrator

    def upsert_documents(self, documents) -> int:
        _ = documents
        return 0

    def search_documents(self, query) -> list[Document]:
        if not self._hydrator.called:
            return []
        return [
            Document(
                ticker=query.ticker or "2603",
                title="長榮 最新毛利率",
                content="截至 2025-12-31，長榮 營業收入約 856.94 億元，營業毛利約 153.53 億元，毛利率約 17.92%。",
                source_name="FinMind TaiwanStockFinancialStatements",
                source_type="gross_margin_snapshot",
                source_tier=SourceTier.HIGH,
                url="https://example.com/2603/latest",
                published_at=datetime(2025, 12, 31, tzinfo=timezone.utc),
                topics=[Topic.EARNINGS],
            )
        ]


class TrackingHydrator:
    def __init__(self) -> None:
        self.called = False
        self.queries = []

    def hydrate(self, query) -> None:
        self.called = True
        self.queries.append(query)


class QueryDataHydratorTestCase(unittest.TestCase):
    def test_hydrator_syncs_financials_for_comparison_query(self) -> None:
        gateway = FakeGateway()
        hydrator = QueryDataHydrator(gateway)
        query = InputLayer().parse(
            QueryRequest(
                query="我想比較長榮 (2603) 跟陽明 (2609)，這兩家公司誰的毛利率比較高？這代表哪一家的經營效率比較好？"
            )
        )

        hydrator.hydrate(query)

        financial_sync_tickers = [
            args[0]
            for method_name, args in gateway.calls
            if method_name == "sync_financial_statements"
        ]
        self.assertIn("2603", financial_sync_tickers)
        self.assertIn("2609", financial_sync_tickers)

    def test_pipeline_calls_query_hydrator_before_retrieval(self) -> None:
        tracking_hydrator = TrackingHydrator()
        pipeline = QueryPipeline(
            input_layer=InputLayer(),
            retrieval_layer=RetrievalLayer(HydrationAwareRepository(tracking_hydrator), max_documents=4),
            data_governance_layer=DataGovernanceLayer(),
            generation_layer=GenerationLayer(
                llm_client=RuleBasedSynthesisClient(),
                prompt_path=Path(__file__).resolve().parents[1]
                / "src"
                / "llm_stock_system"
                / "prompts"
                / "system_prompt.md",
            ),
            validation_layer=ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55),
            presentation_layer=PresentationLayer(),
            query_log_store=InMemoryQueryLogStore(),
            query_hydrator=tracking_hydrator,
        )

        response = pipeline.handle_query(
            QueryRequest(query="長榮 (2603) 最新毛利率是多少？")
        )

        self.assertTrue(tracking_hydrator.called)
        self.assertEqual(tracking_hydrator.queries[0].ticker, "2603")
        self.assertGreaterEqual(len(response.sources), 1)


if __name__ == "__main__":
    unittest.main()
