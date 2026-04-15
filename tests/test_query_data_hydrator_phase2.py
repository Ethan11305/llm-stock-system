from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock
import unittest

from llm_stock_system.core.enums import (
    ConfidenceLight,
    ConsistencyStatus,
    DataFacet,
    FreshnessStatus,
    Intent,
    SufficiencyStatus,
)
from llm_stock_system.core.models import (
    AnswerDraft,
    GovernanceReport,
    HydrationResult,
    QueryRequest,
    StructuredQuery,
    ValidationResult,
)
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.validation_layer import ValidationLayer
from llm_stock_system.orchestrator.pipeline import QueryPipeline
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator


def build_gateway(include_query_news: bool = True):
    gateway = type("Gateway", (), {})()
    gateway.sync_stock_info = Mock(return_value=1)
    gateway.sync_price_history = Mock(return_value=1)
    gateway.sync_financial_statements = Mock(return_value=1)
    gateway.sync_monthly_revenue_points = Mock(return_value=1)
    gateway.sync_dividend_policies = Mock(return_value=1)
    gateway.sync_balance_sheet_items = Mock(return_value=1)
    gateway.sync_cash_flow_statements = Mock(return_value=1)
    gateway.sync_pe_valuation_points = Mock(return_value=1)
    gateway.sync_margin_purchase_short_sale = Mock(return_value=1)
    gateway.sync_stock_news = Mock(return_value=1)
    if include_query_news:
        gateway.sync_query_news = Mock(return_value=1)
    return gateway


class StubInputLayer:
    def parse(self, request: QueryRequest) -> StructuredQuery:
        return StructuredQuery(
            user_query=request.query,
            ticker="2330",
            question_type="market_summary",
        )


class StubRetrievalLayer:
    def retrieve(self, query: StructuredQuery) -> list:
        _ = query
        return []


class StubGovernanceLayer:
    def curate(self, query: StructuredQuery, documents: list) -> GovernanceReport:
        _ = query
        _ = documents
        return GovernanceReport(
            evidence=[],
            sufficiency=SufficiencyStatus.INSUFFICIENT,
            consistency=ConsistencyStatus.MOSTLY_CONSISTENT,
            freshness=FreshnessStatus.OUTDATED,
            high_trust_ratio=0.0,
        )


class StubGenerationLayer:
    def generate(self, query: StructuredQuery, governance_report: GovernanceReport) -> AnswerDraft:
        _ = query
        _ = governance_report
        return AnswerDraft(
            summary="資料不足，無法確認。",
            highlights=[],
            facts=[],
            impacts=[],
            risks=[],
            sources=[],
        )


class CapturingQueryLogStore:
    def __init__(self) -> None:
        self.validation_result = None

    def save(
        self,
        query: StructuredQuery,
        response,
        governance_report: GovernanceReport,
        validation_result,
    ) -> str:
        _ = query
        _ = governance_report
        self.validation_result = validation_result
        return response.query_id

    def get_sources(self, query_id: str):
        _ = query_id
        return None


class ReturningHydrator:
    def hydrate(self, query: StructuredQuery) -> HydrationResult:
        _ = query
        return HydrationResult(facet_miss_list=["news"], preferred_miss_list=["price_history"])


class CapturingValidationLayer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str] | None, list[str] | None]] = []

    def validate(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        facet_miss_list: list[str] | None = None,
        preferred_miss_list: list[str] | None = None,
    ) -> ValidationResult:
        _ = query
        _ = governance_report
        _ = answer_draft
        self.calls.append((facet_miss_list, preferred_miss_list))
        return ValidationResult(
            confidence_score=0.5,
            confidence_light=ConfidenceLight.YELLOW,
            validation_status="review",
            warnings=[],
            facet_miss_list=list(facet_miss_list or []),
        )


class QueryDataHydratorPhase2TestCase(unittest.TestCase):
    def test_hydrate_without_ticker_returns_empty_result(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        result = hydrator.hydrate(StructuredQuery(user_query="market summary"))

        self.assertEqual(result.synced_facets, set())
        self.assertEqual(result.failed_facets, {})
        self.assertEqual(result.facet_miss_list, [])
        self.assertEqual(result.preferred_miss_list, [])
        self.assertEqual(result.total_duration_ms, 0.0)

    def test_news_digest_prefers_query_aware_news_sync(self) -> None:
        gateway = build_gateway(include_query_news=True)
        hydrator = QueryDataHydrator(gateway)
        query = StructuredQuery(
            user_query="theme review",
            ticker="2330",
            question_type="theme_impact_review",
            time_range_label="14d",
            time_range_days=14,
        )

        result = hydrator.hydrate(query)

        gateway.sync_query_news.assert_called_once_with(query)
        gateway.sync_stock_news.assert_not_called()
        gateway.sync_price_history.assert_called_once()
        start_date, end_date = gateway.sync_price_history.call_args.args[1:3]
        self.assertEqual((end_date - start_date).days, 120)
        self.assertEqual(result.synced_facets, {DataFacet.NEWS, DataFacet.PRICE_HISTORY})

    def test_price_history_window_has_120_day_floor(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="valuation",
            ticker="2330",
            question_type="price_range",
            time_range_label="30d",
            time_range_days=30,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.PRICE_HISTORY, query, today)

        self.assertEqual(start_date, today - timedelta(days=120))
        self.assertEqual(end_date, today)

    def test_technical_view_with_technical_tag_uses_180_day_price_window(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="technical view",
            ticker="2330",
            intent=Intent.TECHNICAL_VIEW,
            topic_tags=["技術面"],
            time_range_label="30d",
            time_range_days=30,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.PRICE_HISTORY, query, today)

        self.assertEqual(start_date, today - timedelta(days=180))
        self.assertEqual(end_date, today)

    def test_technical_view_with_season_line_and_chip_tags_uses_120_day_price_window(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="season line margin flow",
            ticker="6669",
            intent=Intent.TECHNICAL_VIEW,
            topic_tags=["季線", "籌碼"],
            time_range_label="30d",
            time_range_days=30,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.PRICE_HISTORY, query, today)

        self.assertEqual(start_date, today - timedelta(days=120))
        self.assertEqual(end_date, today)

    def test_legacy_technical_indicator_question_type_backfills_180_day_price_window(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="technical view",
            ticker="2330",
            question_type="technical_indicator_review",
            time_range_label="30d",
            time_range_days=30,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.PRICE_HISTORY, query, today)

        self.assertEqual(start_date, today - timedelta(days=180))
        self.assertEqual(end_date, today)

    def test_financial_statement_window_uses_multi_year_history(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="earnings",
            ticker="2330",
            question_type="earnings_summary",
            time_range_label="1y",
            time_range_days=365,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.FINANCIAL_STATEMENTS, query, today)

        self.assertEqual(start_date, date(today.year - 2, 1, 1))
        self.assertEqual(end_date, today)

    def test_dividend_window_is_capped_to_recent_two_years(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="dividend",
            ticker="2412",
            question_type="dividend_yield_review",
            time_range_label="10y",
            time_range_days=3650,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.DIVIDEND, query, today)

        self.assertEqual(start_date, today - timedelta(days=730))
        self.assertEqual(end_date, today)

    def test_balance_sheet_and_cash_flow_use_fixed_three_year_window(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="debt safety",
            ticker="2412",
            question_type="debt_dividend_safety_review",
            time_range_label="3y",
            time_range_days=1095,
        )

        balance_start, _ = hydrator._compute_facet_window(DataFacet.BALANCE_SHEET, query, today)
        cash_flow_start, _ = hydrator._compute_facet_window(DataFacet.CASH_FLOW, query, today)

        self.assertEqual(balance_start, date(today.year - 3, 1, 1))
        self.assertEqual(cash_flow_start, date(today.year - 3, 1, 1))

    def test_financial_health_profitability_tags_use_three_year_news_window(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="profitability stability",
            ticker="2454",
            intent=Intent.FINANCIAL_HEALTH,
            topic_tags=["獲利", "穩定性"],
            time_range_label="30d",
            time_range_days=30,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.NEWS, query, today)

        self.assertEqual(start_date, date(today.year - 3, 1, 1))
        self.assertEqual(end_date, today)

    def test_financial_health_non_profitability_tags_keep_default_news_floor(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        cases = [
            ["毛利率"],
            ["營收", "成長"],
        ]

        for topic_tags in cases:
            with self.subTest(topic_tags=topic_tags):
                query = StructuredQuery(
                    user_query="financial health",
                    ticker="2454",
                    intent=Intent.FINANCIAL_HEALTH,
                    topic_tags=topic_tags,
                    time_range_label="14d",
                    time_range_days=14,
                )

                start_date, end_date = hydrator._compute_facet_window(DataFacet.NEWS, query, today)

                self.assertEqual(start_date, today - timedelta(days=30))
                self.assertEqual(end_date, today)

    def test_legacy_profitability_stability_question_type_backfills_three_year_news_window(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="profitability stability",
            ticker="2454",
            question_type="profitability_stability_review",
            time_range_label="30d",
            time_range_days=30,
        )

        start_date, end_date = hydrator._compute_facet_window(DataFacet.NEWS, query, today)

        self.assertEqual(start_date, date(today.year - 3, 1, 1))
        self.assertEqual(end_date, today)

    def test_auto_fetch_facets_do_not_need_date_windows(self) -> None:
        hydrator = QueryDataHydrator(build_gateway())
        today = datetime.now(timezone.utc).date()
        query = StructuredQuery(
            user_query="valuation",
            ticker="2330",
            question_type="fundamental_pe_review",
            time_range_label="1y",
            time_range_days=365,
        )

        self.assertIsNone(hydrator._compute_facet_window(DataFacet.MONTHLY_REVENUE, query, today))
        self.assertIsNone(hydrator._compute_facet_window(DataFacet.PE_VALUATION, query, today))

    def test_required_facet_failure_populates_facet_miss_list(self) -> None:
        gateway = build_gateway(include_query_news=True)
        gateway.sync_query_news.side_effect = ConnectionError("news api down")
        hydrator = QueryDataHydrator(gateway)
        query = StructuredQuery(
            user_query="news digest",
            ticker="2603",
            question_type="theme_impact_review",
        )

        result = hydrator.hydrate(query)

        self.assertEqual(result.facet_miss_list, ["news"])
        self.assertIn(DataFacet.NEWS, result.failed_facets)
        self.assertEqual(result.failed_facets[DataFacet.NEWS], "news api down")

    def test_preferred_facet_failure_does_not_become_required_miss(self) -> None:
        gateway = build_gateway(include_query_news=False)
        gateway.sync_margin_purchase_short_sale.side_effect = RuntimeError("margin api down")
        hydrator = QueryDataHydrator(gateway)
        query = StructuredQuery(
            user_query="season line",
            ticker="6669",
            question_type="season_line_margin_review",
            time_range_label="90d",
            time_range_days=90,
        )

        result = hydrator.hydrate(query)

        self.assertEqual(result.facet_miss_list, [])
        self.assertEqual(result.preferred_miss_list, ["margin_data"])
        self.assertIn(DataFacet.MARGIN_DATA, result.failed_facets)
        self.assertIn(DataFacet.PRICE_HISTORY, result.synced_facets)

    def test_comparison_query_syncs_each_requested_facet_for_both_tickers(self) -> None:
        gateway = build_gateway(include_query_news=True)
        hydrator = QueryDataHydrator(gateway)
        query = StructuredQuery(
            user_query="compare valuation",
            ticker="2330",
            comparison_ticker="2454",
            question_type="fundamental_pe_review",
            time_range_label="1y",
            time_range_days=365,
        )

        hydrator.hydrate(query)

        self.assertEqual(gateway.sync_pe_valuation_points.call_count, 2)
        self.assertEqual(gateway.sync_price_history.call_count, 2)
        self.assertEqual(gateway.sync_financial_statements.call_count, 2)
        self.assertEqual(gateway.sync_stock_news.call_count, 2)
        self.assertEqual(gateway.sync_query_news.call_count, 0)

    def test_dividend_analysis_dispatches_required_and_preferred_facets(self) -> None:
        gateway = build_gateway()
        hydrator = QueryDataHydrator(gateway)
        query = StructuredQuery(
            user_query="fcf dividend",
            ticker="2412",
            question_type="fcf_dividend_sustainability_review",
            time_range_label="3y",
            time_range_days=1095,
        )

        result = hydrator.hydrate(query)

        self.assertEqual(
            result.synced_facets,
            {
                DataFacet.DIVIDEND,
                DataFacet.CASH_FLOW,
                DataFacet.BALANCE_SHEET,
                DataFacet.FINANCIAL_STATEMENTS,
            },
        )
        gateway.sync_dividend_policies.assert_called_once()
        gateway.sync_cash_flow_statements.assert_called_once()
        gateway.sync_balance_sheet_items.assert_called_once()
        gateway.sync_financial_statements.assert_called_once()

    def test_investment_assessment_routes_required_and_preferred_facets(self) -> None:
        gateway = build_gateway(include_query_news=True)
        hydrator = QueryDataHydrator(gateway)
        query = StructuredQuery(
            user_query="investment support",
            ticker="2330",
            question_type="investment_support",
            time_range_label="1y",
            time_range_days=365,
        )

        result = hydrator.hydrate(query)

        self.assertEqual(
            result.synced_facets,
            {
                DataFacet.FINANCIAL_STATEMENTS,
                DataFacet.PE_VALUATION,
                DataFacet.DIVIDEND,
                DataFacet.NEWS,
                DataFacet.PRICE_HISTORY,
            },
        )
        gateway.sync_financial_statements.assert_called_once()
        gateway.sync_pe_valuation_points.assert_called_once()
        gateway.sync_dividend_policies.assert_called_once()
        gateway.sync_stock_news.assert_called_once()
        gateway.sync_price_history.assert_called_once()
        gateway.sync_monthly_revenue_points.assert_not_called()
        gateway.sync_query_news.assert_not_called()

    def test_pipeline_propagates_facet_miss_list_to_validation_result(self) -> None:
        query_log_store = CapturingQueryLogStore()
        pipeline = QueryPipeline(
            input_layer=StubInputLayer(),
            retrieval_layer=StubRetrievalLayer(),
            data_governance_layer=StubGovernanceLayer(),
            generation_layer=StubGenerationLayer(),
            validation_layer=ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55),
            presentation_layer=PresentationLayer(),
            query_log_store=query_log_store,
            query_hydrator=ReturningHydrator(),
        )

        pipeline.handle_query(QueryRequest(query="show me market summary"))

        self.assertIsNotNone(query_log_store.validation_result)
        self.assertEqual(query_log_store.validation_result.facet_miss_list, ["news"])

    def test_pipeline_passes_preferred_miss_list_to_validation_layer(self) -> None:
        validation_layer = CapturingValidationLayer()
        pipeline = QueryPipeline(
            input_layer=StubInputLayer(),
            retrieval_layer=StubRetrievalLayer(),
            data_governance_layer=StubGovernanceLayer(),
            generation_layer=StubGenerationLayer(),
            validation_layer=validation_layer,
            presentation_layer=PresentationLayer(),
            query_log_store=CapturingQueryLogStore(),
            query_hydrator=ReturningHydrator(),
        )

        pipeline.handle_query(QueryRequest(query="show me market summary"))

        self.assertEqual(validation_layer.calls, [(["news"], ["price_history"])])


if __name__ == "__main__":
    unittest.main()
