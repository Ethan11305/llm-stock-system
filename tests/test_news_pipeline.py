from datetime import date, datetime, timezone
import unittest

from llm_stock_system.adapters.news_pipeline import (
    BaseNewsProvider,
    GoogleNewsRssProvider,
    MultiSourceNewsPipeline,
)
from llm_stock_system.core.enums import SourceTier
from llm_stock_system.core.models import NewsArticle


RSS_PAYLOAD = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>News</title>
    <item>
      <title>ASML 展望轉弱，家登觀察 EUV 載具需求</title>
      <link>https://example.com/relevant</link>
      <description><![CDATA[<p>ASML 展望保守，家登持續關注半導體設備需求。</p>]]></description>
      <pubDate>Wed, 08 Apr 2026 02:00:00 GMT</pubDate>
      <source url="https://example.com">經濟日報</source>
    </item>
    <item>
      <title>國際美食展本週登場</title>
      <link>https://example.com/irrelevant</link>
      <description><![CDATA[<p>與半導體無關。</p>]]></description>
      <pubDate>Wed, 08 Apr 2026 01:00:00 GMT</pubDate>
      <source url="https://example.com">生活新聞</source>
    </item>
  </channel>
</rss>
"""


class StubResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class StubClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.request_urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, headers: dict | None = None) -> StubResponse:
        _ = headers
        self.request_urls.append(url)
        return StubResponse(self._text)


class StaticProvider(BaseNewsProvider):
    def __init__(self, provider_name: str, articles: list[NewsArticle]) -> None:
        self.provider_name = provider_name
        self._articles = articles

    def fetch_articles(
        self,
        ticker: str,
        company_name: str | None,
        start_date: date,
        end_date: date,
        search_terms: tuple[str, ...] = (),
    ) -> list[NewsArticle]:
        _ = ticker, company_name, start_date, end_date, search_terms
        return self._articles


class NewsPipelineTestCase(unittest.TestCase):
    def test_google_news_provider_filters_and_normalizes_feed(self) -> None:
        client = StubClient(RSS_PAYLOAD)
        provider = GoogleNewsRssProvider(client_factory=lambda: client)

        articles = provider.fetch_articles(
            ticker="3680",
            company_name="家登",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 8),
            search_terms=("ASML", "設備"),
        )

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "ASML 展望轉弱，家登觀察 EUV 載具需求")
        self.assertEqual(articles[0].source_name, "經濟日報")
        self.assertIn("ASML", articles[0].summary or "")
        self.assertEqual(articles[0].provider_name, "google_news_rss")
        self.assertEqual(articles[0].source_tier, SourceTier.MEDIUM)
        self.assertIn("q=%E5%AE%B6%E7%99%BB", client.request_urls[0])

    def test_multi_source_pipeline_dedupes_and_keeps_higher_tier_article(self) -> None:
        shared_url = "https://example.com/shared"
        medium_article = NewsArticle(
            ticker="3680",
            published_at=datetime(2026, 4, 8, 8, 0, tzinfo=timezone.utc),
            title="設備鏈觀察",
            summary="中可信版本",
            source_name="媒體A",
            url=shared_url,
            source_tier=SourceTier.MEDIUM,
            provider_name="provider-a",
        )
        high_article = NewsArticle(
            ticker="3680",
            published_at=datetime(2026, 4, 8, 7, 0, tzinfo=timezone.utc),
            title="設備鏈觀察",
            summary="高可信版本",
            source_name="官方公告",
            url=shared_url,
            source_tier=SourceTier.HIGH,
            provider_name="provider-b",
        )

        pipeline = MultiSourceNewsPipeline(
            [
                StaticProvider("provider-a", [medium_article]),
                StaticProvider("provider-b", [high_article]),
            ]
        )

        articles = pipeline.fetch_stock_news(
            ticker="3680",
            company_name="家登",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 8),
        )

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].source_tier, SourceTier.HIGH)
        self.assertEqual(articles[0].provider_name, "provider-b")


if __name__ == "__main__":
    unittest.main()
