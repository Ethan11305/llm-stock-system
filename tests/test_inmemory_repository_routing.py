import unittest
from datetime import datetime, timedelta, timezone

from llm_stock_system.adapters.repositories import InMemoryDocumentRepository
from llm_stock_system.core.enums import Intent, SourceTier, Topic
from llm_stock_system.core.models import Document, StructuredQuery


def make_document(
    *,
    ticker: str,
    title: str,
    topics: list[Topic] | None = None,
    source_tier: SourceTier = SourceTier.MEDIUM,
    published_at: datetime | None = None,
) -> Document:
    timestamp = published_at or datetime.now(timezone.utc)
    return Document(
        ticker=ticker,
        title=title,
        content=title,
        source_name="test",
        source_type="news_article",
        source_tier=source_tier,
        url=f"https://example.com/{ticker}/{title}",
        published_at=timestamp,
        topics=topics or [Topic.COMPOSITE],
    )


class InMemoryRepositoryRoutingTestCase(unittest.TestCase):
    def test_theme_query_includes_comparison_ticker_via_intent_and_topic_tags(self) -> None:
        repository = InMemoryDocumentRepository(
            [
                make_document(ticker="3680", title="家登主文"),
                make_document(ticker="6187", title="萬潤比較文"),
                make_document(ticker="2330", title="無關文件"),
            ]
        )
        query = StructuredQuery(
            user_query="ASML 對家登與萬潤的半導體設備影響",
            ticker="3680",
            company_name="家登",
            comparison_ticker="6187",
            comparison_company_name="萬潤",
            intent=Intent.NEWS_DIGEST,
            topic_tags=["半導體設備"],
        )

        documents = repository.search_documents(query)

        self.assertEqual([item.ticker for item in documents[:2]], ["3680", "6187"])

    def test_gross_margin_query_includes_comparison_ticker_without_question_type(self) -> None:
        repository = InMemoryDocumentRepository(
            [
                make_document(ticker="2603", title="長榮毛利率"),
                make_document(ticker="2609", title="陽明毛利率"),
            ]
        )
        query = StructuredQuery(
            user_query="比較長榮與陽明毛利率",
            ticker="2603",
            company_name="長榮",
            comparison_ticker="2609",
            comparison_company_name="陽明",
            intent=Intent.FINANCIAL_HEALTH,
            topic_tags=["毛利率"],
        )

        documents = repository.search_documents(query)

        self.assertEqual({item.ticker for item in documents}, {"2603", "2609"})

    def test_announcement_queries_boost_announcement_topic_documents(self) -> None:
        repository = InMemoryDocumentRepository()
        announcement_doc = make_document(
            ticker="2371",
            title="公告文件",
            topics=[Topic.ANNOUNCEMENT],
            source_tier=SourceTier.LOW,
        )
        generic_doc = make_document(
            ticker="2371",
            title="一般文件",
            topics=[Topic.COMPOSITE],
            source_tier=SourceTier.HIGH,
        )
        query = StructuredQuery(
            user_query="最新公告",
            ticker="2371",
            company_name="大同",
            topic=Topic.ANNOUNCEMENT,
            intent=Intent.INVESTMENT_ASSESSMENT,
            topic_tags=["公告"],
        )

        self.assertGreater(
            repository._score_document(announcement_doc, query),
            repository._score_document(generic_doc, query),
        )

    def test_valuation_queries_boost_price_outlook_and_fundamental_routes(self) -> None:
        repository = InMemoryDocumentRepository()
        doc = make_document(ticker="2330", title="估值文件")

        outlook_query = StructuredQuery(
            user_query="未來半年股價展望",
            ticker="2330",
            company_name="台積電",
            intent=Intent.VALUATION_CHECK,
            topic_tags=["股價", "展望"],
        )
        fundamental_query = StructuredQuery(
            user_query="基本面與本益比如何",
            ticker="2330",
            company_name="台積電",
            intent=Intent.INVESTMENT_ASSESSMENT,
            topic_tags=["基本面", "本益比"],
        )
        neutral_query = StructuredQuery(
            user_query="市場摘要",
            ticker="2330",
            company_name="台積電",
            intent=Intent.NEWS_DIGEST,
            topic_tags=[],
        )

        neutral_score = repository._score_document(doc, neutral_query)
        outlook_score = repository._score_document(doc, outlook_query)
        fundamental_score = repository._score_document(doc, fundamental_query)

        self.assertEqual(outlook_score, neutral_score + 1)
        self.assertEqual(fundamental_score, neutral_score + 1)

    def test_legacy_question_type_backfills_theme_tags_for_comparison_logic(self) -> None:
        repository = InMemoryDocumentRepository(
            [
                make_document(ticker="3680", title="家登"),
                make_document(ticker="6187", title="萬潤"),
            ]
        )
        query = StructuredQuery(
            user_query="題材股比較",
            ticker="3680",
            company_name="家登",
            comparison_ticker="6187",
            comparison_company_name="萬潤",
            question_type="theme_impact_review",
        )

        documents = repository.search_documents(query)

        self.assertEqual({item.ticker for item in documents}, {"3680", "6187"})
        self.assertIn("題材", query.topic_tags)


if __name__ == "__main__":
    unittest.main()
