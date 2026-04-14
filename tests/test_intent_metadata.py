import unittest

from llm_stock_system.core.enums import DataFacet, Intent, TopicTag
from llm_stock_system.core.models import QueryRequest, StructuredQuery
from llm_stock_system.layers.input_layer import InputLayer


class IntentMetadataTestCase(unittest.TestCase):
    def test_structured_query_infers_required_and_preferred_facets(self) -> None:
        query = StructuredQuery(
            user_query="台積電 2330 基本面跟本益比如何？",
            ticker="2330",
            company_name="台積電",
            question_type="fundamental_pe_review",
            time_range_label="1y",
            time_range_days=365,
        )

        self.assertEqual(query.intent, Intent.VALUATION_CHECK)
        self.assertEqual(query.required_facets, {DataFacet.PE_VALUATION})
        self.assertEqual(
            query.preferred_facets,
            {
                DataFacet.PRICE_HISTORY,
                DataFacet.FINANCIAL_STATEMENTS,
                DataFacet.NEWS,
            },
        )
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
            preferred_facets={DataFacet.NEWS},
        )

        self.assertEqual(query.intent, Intent.DIVIDEND_ANALYSIS)
        self.assertEqual(query.required_facets, {DataFacet.DIVIDEND})
        self.assertEqual(
            query.preferred_facets,
            {
                DataFacet.CASH_FLOW,
                DataFacet.BALANCE_SHEET,
                DataFacet.FINANCIAL_STATEMENTS,
                DataFacet.NEWS,
            },
        )

    def test_input_layer_outputs_controlled_tags_and_free_keywords(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="長榮 2603 和陽明 2609 因為紅海航線受阻，SCFI 運價上漲，法人有上修目標價嗎？"
            )
        )

        self.assertEqual(query.question_type, "shipping_rate_impact_review")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(query.required_facets, {DataFacet.NEWS})
        self.assertEqual(query.preferred_facets, {DataFacet.PRICE_HISTORY})
        self.assertIn(TopicTag.SHIPPING, query.controlled_tags)
        self.assertIn("SCFI", query.free_keywords)
        self.assertIn("航運", query.topic_tags)
        self.assertIn("SCFI", query.topic_tags)
        self.assertEqual(query.tag_source, "matched")

    def test_input_layer_uses_fallback_keywords_when_no_controlled_tag_matches(self) -> None:
        query = StructuredQuery(
            user_query="市場摘要",
            question_type="market_summary",
            free_keywords=["市場"],
            tag_source="fallback",
        )

        self.assertEqual(query.controlled_tags, [])
        self.assertEqual(query.topic_tags, ["市場"])
        self.assertEqual(query.tag_source, "fallback")

    def test_input_layer_maps_fundamental_pe_to_valuation_intent(self) -> None:
        query = InputLayer().parse(QueryRequest(query="台積電 2330 基本面跟本益比如何？"))

        self.assertEqual(query.question_type, "fundamental_pe_review")
        self.assertEqual(query.intent, Intent.VALUATION_CHECK)
        self.assertIn(TopicTag.VALUATION, query.controlled_tags)
        self.assertIn(TopicTag.FUNDAMENTAL, query.controlled_tags)
        self.assertIn("基本面", query.topic_tags)
        self.assertIn("本益比", query.topic_tags)


if __name__ == "__main__":
    unittest.main()
