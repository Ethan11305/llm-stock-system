import unittest

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import StructuredQuery
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.core.models import QueryRequest


class IntentMetadataTestCase(unittest.TestCase):
    def test_structured_query_infers_intent_and_facets_from_question_type(self) -> None:
        query = StructuredQuery(
            user_query="台積電 2330 基本面跟本益比如何？",
            ticker="2330",
            company_name="台積電",
            question_type="fundamental_pe_review",
            time_range_label="1y",
            time_range_days=365,
        )

        self.assertEqual(query.intent, Intent.VALUATION_CHECK)
        self.assertEqual(
            query.data_facets,
            {
                DataFacet.PE_VALUATION,
                DataFacet.PRICE_HISTORY,
                DataFacet.FINANCIAL_STATEMENTS,
                DataFacet.NEWS,
            },
        )

    def test_structured_query_merges_explicit_facets_with_intent_defaults(self) -> None:
        query = StructuredQuery(
            user_query="中華電 2412 股利安不安全？",
            ticker="2412",
            company_name="中華電信",
            question_type="debt_dividend_safety_review",
            data_facets={DataFacet.NEWS},
        )

        self.assertEqual(query.intent, Intent.DIVIDEND_ANALYSIS)
        self.assertEqual(
            query.data_facets,
            {
                DataFacet.DIVIDEND,
                DataFacet.CASH_FLOW,
                DataFacet.BALANCE_SHEET,
                DataFacet.FINANCIAL_STATEMENTS,
                DataFacet.NEWS,
            },
        )

    def test_input_layer_outputs_intent_topic_tags_and_facets(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="長榮 2603 和陽明 2609 因為紅海航線受阻，SCFI 運價上漲，法人有上修目標價嗎？"
            )
        )

        self.assertEqual(query.question_type, "shipping_rate_impact_review")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertIn("航運", query.topic_tags)
        self.assertIn("SCFI", query.topic_tags)
        self.assertEqual(
            query.data_facets,
            {DataFacet.NEWS, DataFacet.PRICE_HISTORY},
        )

    def test_input_layer_maps_fundamental_pe_to_valuation_intent(self) -> None:
        query = InputLayer().parse(QueryRequest(query="台積電 2330 基本面跟本益比如何？"))

        self.assertEqual(query.question_type, "fundamental_pe_review")
        self.assertEqual(query.intent, Intent.VALUATION_CHECK)
        self.assertIn("基本面", query.topic_tags)
        self.assertIn("本益比", query.topic_tags)


if __name__ == "__main__":
    unittest.main()
