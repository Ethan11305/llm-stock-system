import unittest
from datetime import datetime, timezone

from llm_stock_system.adapters.postgres_market_data import FinMindPostgresGateway
from llm_stock_system.core.enums import SourceTier, Topic
from llm_stock_system.core.models import Document


def make_document(
    *,
    title: str,
    source_type: str,
    source_tier: SourceTier,
    url: str,
    published_at: datetime,
) -> Document:
    return Document(
        ticker="2408",
        title=title,
        content=title,
        source_name="test",
        source_type=source_type,
        source_tier=source_tier,
        url=url,
        published_at=published_at,
        topics=[Topic.COMPOSITE],
    )


class PostgresDocumentSortingTestCase(unittest.TestCase):
    def test_investment_support_prefers_pe_and_financial_documents(self) -> None:
        gateway = object.__new__(FinMindPostgresGateway)
        docs = [
            make_document(
                title="latest news",
                source_type="news_article",
                source_tier=SourceTier.MEDIUM,
                url="https://example.com/news",
                published_at=datetime(2026, 4, 13, 11, 22, tzinfo=timezone.utc),
            ),
            make_document(
                title="duplicate latest news",
                source_type="news_article",
                source_tier=SourceTier.MEDIUM,
                url="https://example.com/news",
                published_at=datetime(2026, 4, 13, 11, 22, tzinfo=timezone.utc),
            ),
            make_document(
                title="current pe",
                source_type="pe_current",
                source_tier=SourceTier.HIGH,
                url="https://example.com/pe-current",
                published_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
            ),
            make_document(
                title="latest eps",
                source_type="financial_statement_latest",
                source_tier=SourceTier.HIGH,
                url="https://example.com/eps-latest",
                published_at=datetime(2025, 12, 31, tzinfo=timezone.utc),
            ),
        ]

        ordered = gateway._sorted(docs, question_type="investment_support")

        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0].source_type, "pe_current")
        self.assertEqual(ordered[1].source_type, "financial_statement_latest")
        self.assertEqual(ordered[2].source_type, "news_article")


if __name__ == "__main__":
    unittest.main()
