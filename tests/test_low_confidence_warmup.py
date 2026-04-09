from pathlib import Path
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.repositories import InMemoryQueryLogStore
from llm_stock_system.core.enums import ConfidenceLight
from llm_stock_system.core.models import QueryRequest, StructuredQuery, ValidationResult
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
        return 1

    def sync_price_history(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_price_history", (ticker, start_date, end_date)))
        return 1

    def sync_financial_statements(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_financial_statements", (ticker, start_date, end_date)))
        return 1

    def sync_balance_sheet_items(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_balance_sheet_items", (ticker, start_date, end_date)))
        return 1

    def sync_cash_flow_statements(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_cash_flow_statements", (ticker, start_date, end_date)))
        return 1

    def sync_dividend_policies(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_dividend_policies", (ticker, start_date, end_date)))
        return 1

    def sync_stock_news(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_stock_news", (ticker, start_date, end_date)))
        return 1

    def sync_margin_purchase_short_sale(self, ticker, start_date, end_date) -> int:
        self.calls.append(("sync_margin_purchase_short_sale", (ticker, start_date, end_date)))
        return 1

    def sync_monthly_revenue_points(self, ticker) -> int:
        self.calls.append(("sync_monthly_revenue_points", (ticker,)))
        return 1

    def sync_pe_valuation_points(self, ticker) -> int:
        self.calls.append(("sync_pe_valuation_points", (ticker,)))
        return 1


class EmptyRepository:
    def upsert_documents(self, documents) -> int:
        _ = documents
        return 0

    def search_documents(self, query):
        _ = query
        return []


class TrackingHydrator:
    def __init__(self) -> None:
        self.hydrate_calls: list[StructuredQuery] = []
        self.follow_up_calls: list[tuple[StructuredQuery, ValidationResult]] = []

    def hydrate(self, query: StructuredQuery) -> None:
        self.hydrate_calls.append(query)

    def schedule_follow_up(self, query: StructuredQuery, validation_result: ValidationResult) -> bool:
        self.follow_up_calls.append((query, validation_result))
        return True


class LowConfidenceWarmupTestCase(unittest.TestCase):
    def test_low_confidence_follow_up_syncs_broad_dataset_bundle(self) -> None:
        gateway = FakeGateway()
        hydrator = QueryDataHydrator(
            gateway,
            low_confidence_warmup_enabled=True,
            low_confidence_warmup_threshold=0.80,
            follow_up_cooldown_hours=12,
            run_follow_up_async=False,
        )
        query = StructuredQuery(
            user_query="宏碁(2353) 負債比率是否突然升高？現金還夠不夠發股利？",
            ticker="2353",
            company_name="宏碁",
            question_type="debt_dividend_safety_review",
            time_range_label="3y",
            time_range_days=1095,
        )
        validation_result = ValidationResult(
            confidence_score=0.25,
            confidence_light=ConfidenceLight.RED,
            validation_status="blocked",
            warnings=["No supporting evidence retrieved."],
        )

        scheduled = hydrator.schedule_follow_up(query, validation_result)

        self.assertTrue(scheduled)
        methods = [method_name for method_name, _ in gateway.calls]
        self.assertIn("sync_stock_info", methods)
        self.assertIn("sync_price_history", methods)
        self.assertIn("sync_financial_statements", methods)
        self.assertIn("sync_balance_sheet_items", methods)
        self.assertIn("sync_cash_flow_statements", methods)
        self.assertIn("sync_dividend_policies", methods)
        self.assertIn("sync_monthly_revenue_points", methods)
        self.assertIn("sync_pe_valuation_points", methods)
        self.assertIn("sync_margin_purchase_short_sale", methods)
        self.assertIn("sync_stock_news", methods)

    def test_low_confidence_follow_up_respects_cooldown(self) -> None:
        gateway = FakeGateway()
        hydrator = QueryDataHydrator(
            gateway,
            low_confidence_warmup_enabled=True,
            low_confidence_warmup_threshold=0.80,
            follow_up_cooldown_hours=12,
            run_follow_up_async=False,
        )
        query = StructuredQuery(
            user_query="中華電(2412) 本益比是否在高位？",
            ticker="2412",
            company_name="中華電信",
            question_type="pe_valuation_review",
            time_range_label="1y",
            time_range_days=365,
        )
        validation_result = ValidationResult(
            confidence_score=0.25,
            confidence_light=ConfidenceLight.RED,
            validation_status="blocked",
            warnings=["Answer indicates insufficient data."],
        )

        first_schedule = hydrator.schedule_follow_up(query, validation_result)
        second_schedule = hydrator.schedule_follow_up(query, validation_result)

        self.assertTrue(first_schedule)
        self.assertFalse(second_schedule)

    def test_pipeline_triggers_follow_up_when_response_is_low_confidence(self) -> None:
        tracking_hydrator = TrackingHydrator()
        pipeline = QueryPipeline(
            input_layer=InputLayer(),
            retrieval_layer=RetrievalLayer(EmptyRepository(), max_documents=4),
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
            QueryRequest(query="宏碁(2353) 負債比率是否突然升高？公司手上的現金還夠不夠發股利？")
        )

        self.assertEqual(response.confidence_light.value, "red")
        self.assertEqual(len(tracking_hydrator.hydrate_calls), 1)
        self.assertEqual(len(tracking_hydrator.follow_up_calls), 1)
        scheduled_query, validation_result = tracking_hydrator.follow_up_calls[0]
        self.assertEqual(scheduled_query.ticker, "2353")
        self.assertLess(validation_result.confidence_score, 0.80)


if __name__ == "__main__":
    unittest.main()
