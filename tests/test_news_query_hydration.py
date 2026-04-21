import unittest

from llm_stock_system.core.enums import Intent
from llm_stock_system.core.models import QueryRequest
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.services.query_data_hydrator import QueryDataHydrator


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def sync_stock_info(self, force: bool = False) -> int:
        self.calls.append(("sync_stock_info", force))
        return 1

    def sync_query_news(self, query) -> int:
        self.calls.append(("sync_query_news", query))
        return 2


class NewsQueryHydrationTestCase(unittest.TestCase):
    def test_theme_query_uses_query_aware_news_sync(self) -> None:
        hydrator = QueryDataHydrator(FakeGateway())
        query = InputLayer().parse(
            QueryRequest(
                query="如果 ASML（艾司摩爾）最新展望不如預期，搜尋這對台灣半導體設備族群（如 3680 家登、6187 萬潤）的最新利空分析與情緒影響"
            )
        )

        hydrator.hydrate(query)

        methods = [method for method, _ in hydrator._gateway.calls]
        self.assertIn("sync_stock_info", methods)
        self.assertIn("sync_query_news", methods)

        synced_query = next(payload for method, payload in hydrator._gateway.calls if method == "sync_query_news")
        self.assertEqual(synced_query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(synced_query.ticker, "3680")
        self.assertEqual(synced_query.comparison_ticker, "6187")


if __name__ == "__main__":
    unittest.main()
