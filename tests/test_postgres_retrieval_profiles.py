import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from llm_stock_system.adapters.postgres_market_data import FinMindPostgresGateway
from llm_stock_system.core.enums import Intent, SourceTier, Topic
from llm_stock_system.core.models import Document, StructuredQuery


def make_document(
    *,
    ticker: str = "2330",
    title: str,
    source_type: str,
    source_tier: SourceTier,
    url: str,
    published_at: datetime,
) -> Document:
    return Document(
        ticker=ticker,
        title=title,
        content=title,
        source_name="test",
        source_type=source_type,
        source_tier=source_tier,
        url=url,
        published_at=published_at,
        topics=[Topic.COMPOSITE],
    )


def make_query(
    *,
    user_query: str,
    intent: Intent,
    topic_tags: list[str] | None = None,
    ticker: str = "2330",
    company_name: str = "台積電",
    comparison_ticker: str | None = None,
    comparison_company_name: str | None = None,
    topic: Topic = Topic.COMPOSITE,
) -> StructuredQuery:
    return StructuredQuery(
        user_query=user_query,
        ticker=ticker,
        company_name=company_name,
        comparison_ticker=comparison_ticker,
        comparison_company_name=comparison_company_name,
        topic=topic,
        intent=intent,
        topic_tags=topic_tags or [],
    )


class PostgresRetrievalProfileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = object.__new__(FinMindPostgresGateway)

    def test_resolve_news_digest_profiles(self) -> None:
        cases = [
            (["航運"], "news_shipping"),
            (["電價"], "news_electricity"),
            (["CPI"], "news_macro"),
            (["半導體設備", "法說"], "news_theme"),
            (["法說"], "news_guidance"),
            (["上市"], "news_listing"),
            (["半導體設備"], "news_theme"),
            ([], "news_generic"),
        ]

        for topic_tags, expected in cases:
            with self.subTest(topic_tags=topic_tags, expected=expected):
                query = make_query(
                    user_query="測試查詢",
                    intent=Intent.NEWS_DIGEST,
                    topic_tags=topic_tags,
                )
                profile = self.gateway._resolve_retrieval_profile(query)
                self.assertEqual(profile.key, expected)

    def test_resolve_profiles_for_other_intents(self) -> None:
        cases = [
            (
                make_query(
                    user_query="月營收如何？",
                    intent=Intent.EARNINGS_REVIEW,
                    topic_tags=["月營收"],
                ),
                "earnings_monthly_revenue",
            ),
            (
                make_query(
                    user_query="毛利率有沒有轉正？",
                    intent=Intent.EARNINGS_REVIEW,
                    topic_tags=["毛利率", "轉正"],
                ),
                "earnings_margin_turnaround",
            ),
            (
                make_query(
                    user_query="EPS 與股利如何？",
                    intent=Intent.EARNINGS_REVIEW,
                    topic_tags=["EPS", "股利"],
                ),
                "earnings_eps_dividend",
            ),
            (
                make_query(
                    user_query="最近財報摘要",
                    intent=Intent.EARNINGS_REVIEW,
                    topic_tags=["財報"],
                ),
                "earnings_fundamental",
            ),
            (
                make_query(
                    user_query="目前股價區間",
                    intent=Intent.VALUATION_CHECK,
                    topic_tags=["股價區間"],
                ),
                "valuation_price_range",
            ),
            (
                make_query(
                    user_query="未來半年股價展望",
                    intent=Intent.VALUATION_CHECK,
                    topic_tags=["股價", "展望"],
                ),
                "valuation_price_outlook",
            ),
            (
                make_query(
                    user_query="基本面搭配本益比如何",
                    intent=Intent.VALUATION_CHECK,
                    topic_tags=["基本面", "本益比"],
                ),
                "valuation_fundamental",
            ),
            (
                make_query(
                    user_query="目前本益比高嗎",
                    intent=Intent.VALUATION_CHECK,
                    topic_tags=["本益比"],
                ),
                "valuation_pe_only",
            ),
            (
                make_query(
                    user_query="除息填息表現",
                    intent=Intent.DIVIDEND_ANALYSIS,
                    topic_tags=["除息", "填息"],
                ),
                "dividend_ex",
            ),
            (
                make_query(
                    user_query="現金流能不能支撐股利",
                    intent=Intent.DIVIDEND_ANALYSIS,
                    topic_tags=["股利", "現金流"],
                ),
                "dividend_fcf",
            ),
            (
                make_query(
                    user_query="負債會不會影響股利",
                    intent=Intent.DIVIDEND_ANALYSIS,
                    topic_tags=["股利", "負債"],
                ),
                "dividend_debt",
            ),
            (
                make_query(
                    user_query="殖利率多少",
                    intent=Intent.DIVIDEND_ANALYSIS,
                    topic_tags=["股利"],
                ),
                "dividend_yield",
            ),
            (
                make_query(
                    user_query="比較兩家公司毛利率",
                    intent=Intent.FINANCIAL_HEALTH,
                    topic_tags=["毛利率"],
                    comparison_ticker="2317",
                    comparison_company_name="鴻海",
                ),
                "health_gross_margin_cmp",
            ),
            (
                make_query(
                    user_query="獲利穩定嗎",
                    intent=Intent.FINANCIAL_HEALTH,
                    topic_tags=["獲利", "穩定性"],
                ),
                "health_profitability",
            ),
            (
                make_query(
                    user_query="AI 營收成長如何",
                    intent=Intent.FINANCIAL_HEALTH,
                    topic_tags=["營收", "成長", "AI"],
                ),
                "health_revenue_growth",
            ),
            (
                make_query(
                    user_query="是否跌破季線且籌碼偏熱",
                    intent=Intent.TECHNICAL_VIEW,
                    topic_tags=["季線", "籌碼"],
                ),
                "technical_margin_flow",
            ),
            (
                make_query(
                    user_query="MACD 與 RSI 如何",
                    intent=Intent.TECHNICAL_VIEW,
                    topic_tags=["技術面"],
                ),
                "technical_indicators",
            ),
            (
                make_query(
                    user_query="有沒有最新公告",
                    intent=Intent.INVESTMENT_ASSESSMENT,
                    topic_tags=["公告"],
                    topic=Topic.ANNOUNCEMENT,
                ),
                "investment_announcement",
            ),
            (
                make_query(
                    user_query="值不值得投資",
                    intent=Intent.INVESTMENT_ASSESSMENT,
                    topic_tags=["基本面", "本益比"],
                ),
                "investment_support",
            ),
            (
                make_query(
                    user_query="主要風險有哪些",
                    intent=Intent.INVESTMENT_ASSESSMENT,
                    topic_tags=["風險"],
                ),
                "investment_risk",
            ),
        ]

        for query, expected in cases:
            with self.subTest(expected=expected, user_query=query.user_query):
                profile = self.gateway._resolve_retrieval_profile(query)
                self.assertEqual(profile.key, expected)

    def test_news_digest_topic_fallbacks_preserve_legacy_behavior(self) -> None:
        earnings_query = make_query(
            user_query="最新財報摘要",
            intent=Intent.NEWS_DIGEST,
            topic=Topic.EARNINGS,
            topic_tags=[],
        )
        announcement_query = make_query(
            user_query="最新公告摘要",
            intent=Intent.NEWS_DIGEST,
            topic=Topic.ANNOUNCEMENT,
            topic_tags=[],
        )

        self.assertEqual(
            self.gateway._resolve_retrieval_profile(earnings_query).key,
            "earnings_fundamental",
        )
        self.assertEqual(
            self.gateway._resolve_retrieval_profile(announcement_query).key,
            "investment_announcement",
        )

    def test_build_news_search_terms_static_profile_includes_expected_labels(self) -> None:
        query = make_query(
            user_query="長榮和陽明還會受惠 SCFI 嗎？",
            intent=Intent.NEWS_DIGEST,
            topic_tags=["航運", "SCFI"],
            ticker="2603",
            company_name="長榮",
            comparison_ticker="2609",
            comparison_company_name="陽明",
        )

        terms = self.gateway._build_news_search_terms(query, "長榮")

        self.assertIn("紅海", terms)
        self.assertIn("SCFI", terms)
        self.assertIn("長榮", terms)
        self.assertIn("陽明", terms)

    def test_build_news_search_terms_theme_profile_uses_dynamic_keywords(self) -> None:
        query = make_query(
            user_query="如果 ASML 展望不如預期，對家登與萬潤的半導體設備族群有何影響？",
            intent=Intent.NEWS_DIGEST,
            topic_tags=["半導體設備"],
            ticker="3680",
            company_name="家登",
            comparison_ticker="6187",
            comparison_company_name="萬潤",
        )

        terms = self.gateway._build_news_search_terms(query, "家登")

        self.assertIn("ASML", terms)
        self.assertIn("半導體設備", terms)

    def test_build_news_search_terms_generic_news_uses_query_buckets(self) -> None:
        query = make_query(
            user_query="請整理不如預期與下修的市場反應",
            intent=Intent.NEWS_DIGEST,
            topic_tags=[],
        )

        terms = self.gateway._build_news_search_terms(query, "台積電")

        self.assertIn("不如預期", terms)
        self.assertIn("下修", terms)

    def test_sync_query_news_respects_include_comparison(self) -> None:
        shipping_query = make_query(
            user_query="長榮和陽明還會受惠 SCFI 嗎？",
            intent=Intent.NEWS_DIGEST,
            topic_tags=["航運", "SCFI"],
            ticker="2603",
            company_name="長榮",
            comparison_ticker="2609",
            comparison_company_name="陽明",
        )
        macro_query = make_query(
            user_query="CPI 對金融股影響如何？",
            intent=Intent.NEWS_DIGEST,
            topic_tags=["CPI"],
            ticker="2882",
            company_name="國泰金",
            comparison_ticker="2881",
            comparison_company_name="富邦金",
        )

        self.gateway.sync_stock_news = Mock(return_value=1)

        self.gateway.sync_query_news(shipping_query)
        self.assertEqual(self.gateway.sync_stock_news.call_count, 2)

        self.gateway.sync_stock_news.reset_mock()

        self.gateway.sync_query_news(macro_query)
        self.assertEqual(self.gateway.sync_stock_news.call_count, 1)

    def test_build_documents_uses_composite_builder_plan(self) -> None:
        query = make_query(
            user_query="基本面搭配本益比如何",
            intent=Intent.VALUATION_CHECK,
            topic_tags=["基本面", "本益比"],
        )
        self.gateway._build_fundamental_documents = Mock(
            return_value=[
                make_document(
                    title="financial",
                    source_type="financial_statement_latest",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/financial",
                    published_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
                )
            ]
        )
        self.gateway._build_pe_valuation_documents = Mock(
            return_value=[
                make_document(
                    title="pe",
                    source_type="pe_current",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/pe",
                    published_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
                )
            ]
        )
        self.gateway._build_market_documents = Mock(
            return_value=[
                make_document(
                    title="news",
                    source_type="news_article",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/news",
                    published_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
                )
            ]
        )

        ordered = self.gateway.build_documents(query)

        self.gateway._build_fundamental_documents.assert_called_once_with(query)
        self.gateway._build_pe_valuation_documents.assert_called_once_with(query)
        self.gateway._build_market_documents.assert_called_once_with(query)
        self.assertEqual([item.source_type for item in ordered], ["pe_current", "financial_statement_latest", "news_article"])


if __name__ == "__main__":
    unittest.main()
