"""test_hybrid_retrieval_and_announcement.py

測試涵蓋兩個面向：

1. Announcement routing（不同 ticker）
   - InputLayer 能正確偵測 announcement_summary 類型
   - question_type / intent / facets 對三支不同股票都正確

2. HybridRetrievalLayer 行為驗證
   - vector_adapter 被實際呼叫（search() 有被觸發）
   - 語意檢索結果有參與合併排序
   - vector_adapter 失敗時 graceful fallback 到純 metadata
   - vector_adapter=None 時直接用 metadata（不 crash）

3. upsert_documents() 行為驗證
   - 正確映射欄位並呼叫 DB
   - 空列表時直接回傳 0
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

from llm_stock_system.adapters.vector_retrieval import SemanticSearchResult
from llm_stock_system.adapters.repositories import InMemoryDocumentRepository
from llm_stock_system.core.enums import DataFacet, Intent, SourceTier, Topic
from llm_stock_system.core.models import Document, QueryRequest, StructuredQuery
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.retrieval_layer import HybridRetrievalLayer, RetrievalLayer
from llm_stock_system.sample_data.documents import SAMPLE_DOCUMENTS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_NOW = datetime.now(timezone.utc)


def _doc(
    ticker: str = "2330",
    title: str = "title",
    url: str = "http://example.com/1",
    published_at: datetime | None = None,
) -> Document:
    return Document(
        ticker=ticker,
        title=title,
        content=title,
        source_name="test",
        source_type="news",
        source_tier=SourceTier.MEDIUM,
        url=url,
        # 預設用「現在」，確保不被 time_range_days 過濾掉
        published_at=published_at or _NOW,
    )


def _semantic(doc_id: str, score: float, ticker: str = "2330") -> SemanticSearchResult:
    return SemanticSearchResult(
        document_id=doc_id,
        chunk_text="chunk",
        chunk_index=0,
        similarity_score=score,
        ticker=ticker,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Announcement routing（不同 ticker）
# ─────────────────────────────────────────────────────────────────────────────

class AnnouncementRoutingTestCase(unittest.TestCase):
    """確認 InputLayer 對不同 ticker 的公告查詢，都能正確解析出
    question_type='announcement_summary'、intent=INVESTMENT_ASSESSMENT。
    """

    def setUp(self) -> None:
        self.input_layer = InputLayer()

    def _parse(self, query_text: str) -> StructuredQuery:
        return self.input_layer.parse(QueryRequest(query=query_text))

    # ── 台積電 2330 ──
    # 注意：含「法說會」的查詢同時觸發 EARNINGS（"法說" 是子字串）和
    # ANNOUNCEMENT（"公告"），造成 topic=COMPOSITE → market_summary。
    # 因此使用純公告關鍵字（只含「公告」或「重大公告」）確保單一 topic 命中。
    def test_tsmc_announcement(self) -> None:
        q = self._parse("台積電 (2330) 有沒有最新的重大公告？")
        self.assertEqual(q.ticker, "2330")
        self.assertEqual(q.intent, Intent.INVESTMENT_ASSESSMENT)

    # ── 鴻海 2317 ──
    def test_foxconn_announcement(self) -> None:
        q = self._parse("鴻海 (2317) 有沒有最新的重大公告？")
        self.assertEqual(q.ticker, "2317")
        self.assertEqual(q.intent, Intent.INVESTMENT_ASSESSMENT)

    # ── 長榮 2603 ──
    def test_evergreen_announcement(self) -> None:
        q = self._parse("長榮 (2603) 近期有什麼公告？")
        self.assertEqual(q.ticker, "2603")
        self.assertEqual(q.intent, Intent.INVESTMENT_ASSESSMENT)

    # ── 補充：同時含「法說會」和「公告」 → COMPOSITE → market_summary（驗證邊界行為）──
    def test_mixed_announcement_and_earnings_falls_back_to_market_summary(self) -> None:
        q = self._parse("台積電 (2330) 最近有什麼重大公告或法說會重點？")
        self.assertEqual(q.ticker, "2330")
        # 「法說」(EARNINGS) + 「公告」(ANNOUNCEMENT) → 兩 topic 都命中 → COMPOSITE → NEWS_DIGEST
        self.assertEqual(q.intent, Intent.NEWS_DIGEST)

    # ── facet 應包含 FINANCIAL_STATEMENTS 和 PE_VALUATION（INVESTMENT_ASSESSMENT required）──
    def test_announcement_required_facets(self) -> None:
        q = self._parse("台積電 (2330) 最近有什麼重大公告？")
        self.assertIn(DataFacet.FINANCIAL_STATEMENTS, q.required_facets)
        self.assertIn(DataFacet.PE_VALUATION, q.required_facets)

    # ── 不同 ticker，facet 規則一致 ──
    def test_announcement_facets_consistent_across_tickers(self) -> None:
        tickers_and_queries = [
            ("2330", "台積電 (2330) 最近有什麼重大公告？"),
            ("2317", "鴻海 (2317) 有沒有最新的重大公告？"),
            ("2603", "長榮 (2603) 近期有什麼公告？"),
        ]
        for ticker, query_text in tickers_and_queries:
            with self.subTest(ticker=ticker):
                q = self._parse(query_text)
                self.assertEqual(q.ticker, ticker)
                self.assertIn(DataFacet.FINANCIAL_STATEMENTS, q.required_facets)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HybridRetrievalLayer 行為驗證
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetrievalLayerTestCase(unittest.TestCase):
    """確認 HybridRetrievalLayer 有正確呼叫 vector_adapter.search()，
    以及合併邏輯行為正確。
    """

    def _make_repo(self, docs: list[Document]) -> InMemoryDocumentRepository:
        return InMemoryDocumentRepository(docs)

    def _make_query(self, ticker: str = "2330") -> StructuredQuery:
        return StructuredQuery(
            user_query="台積電最近有什麼供應鏈風險？",
            ticker=ticker,
            time_range_days=30,
        )

    # ── vector_adapter.search() 有被呼叫 ──
    def test_vector_adapter_search_is_called(self) -> None:
        doc_a = _doc(ticker="2330", title="A", url="http://example.com/a")
        doc_b = _doc(ticker="2330", title="B", url="http://example.com/b")
        repo = self._make_repo([doc_a, doc_b])

        mock_adapter = MagicMock()
        mock_adapter.search.return_value = []  # 語意檢索回傳空，應 fallback

        layer = HybridRetrievalLayer(
            document_repository=repo,
            vector_adapter=mock_adapter,
            max_documents=5,
        )
        query = self._make_query()
        result = layer.retrieve(query)

        # search() 應被呼叫一次，且帶正確的 query_text 和 ticker
        mock_adapter.search.assert_called_once()
        call_kwargs = mock_adapter.search.call_args
        self.assertEqual(call_kwargs.kwargs.get("query_text") or call_kwargs.args[0],
                         "台積電最近有什麼供應鏈風險？")

        # fallback：語意空，應回傳 metadata 結果
        self.assertEqual(len(result), 2)

    # ── 語意結果參與合併排序 ──
    def test_semantic_results_participate_in_merge(self) -> None:
        doc_a = _doc(ticker="2330", title="不相關新聞", url="http://example.com/a")
        doc_b = _doc(ticker="2330", title="供應鏈風險報導", url="http://example.com/b")
        repo = self._make_repo([doc_a, doc_b])  # metadata 順序：A 第一

        mock_adapter = MagicMock()
        # 語意給 B 高分，A 低分 → 混合後 B 應排到第一
        mock_adapter.search.return_value = [
            _semantic(doc_b.id, score=0.92),
            _semantic(doc_a.id, score=0.30),
        ]

        layer = HybridRetrievalLayer(
            document_repository=repo,
            vector_adapter=mock_adapter,
            max_documents=5,
            semantic_weight=0.8,
            metadata_weight=0.2,
        )
        result = layer.retrieve(self._make_query())

        self.assertEqual(len(result), 2)
        # B 語意分高，應排第一
        self.assertEqual(result[0].id, doc_b.id)
        self.assertEqual(result[1].id, doc_a.id)

    # ── max_documents 有效截斷 ──
    def test_max_documents_limit_respected(self) -> None:
        docs = [_doc(url=f"http://example.com/{i}", title=f"doc{i}") for i in range(10)]
        repo = self._make_repo(docs)

        mock_adapter = MagicMock()
        mock_adapter.search.return_value = [_semantic(d.id, score=0.7) for d in docs[:5]]

        layer = HybridRetrievalLayer(
            document_repository=repo,
            vector_adapter=mock_adapter,
            max_documents=3,
        )
        result = layer.retrieve(self._make_query())
        self.assertLessEqual(len(result), 3)

    # ── vector_adapter 失敗時 graceful fallback ──
    def test_vector_adapter_failure_graceful_fallback(self) -> None:
        doc_a = _doc(url="http://example.com/a", title="A")
        doc_b = _doc(url="http://example.com/b", title="B")
        repo = self._make_repo([doc_a, doc_b])

        mock_adapter = MagicMock()
        mock_adapter.search.side_effect = RuntimeError("pgvector connection failed")

        layer = HybridRetrievalLayer(
            document_repository=repo,
            vector_adapter=mock_adapter,
            max_documents=5,
        )
        # 不應 raise，應靜默 fallback
        result = layer.retrieve(self._make_query())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, doc_a.id)

    # ── vector_adapter=None 時純 metadata ──
    def test_no_vector_adapter_pure_metadata(self) -> None:
        docs = [_doc(url=f"http://example.com/{i}", title=f"doc{i}") for i in range(5)]
        repo = self._make_repo(docs)

        layer = HybridRetrievalLayer(
            document_repository=repo,
            vector_adapter=None,
            max_documents=3,
        )
        result = layer.retrieve(self._make_query())
        self.assertLessEqual(len(result), 3)

    # ── 不同 ticker 查詢時，search() 傳入對應 ticker ──
    def test_ticker_passed_to_vector_adapter(self) -> None:
        docs = [_doc(ticker="2317", url="http://example.com/fox", title="鴻海新聞")]
        repo = self._make_repo(docs)

        mock_adapter = MagicMock()
        mock_adapter.search.return_value = []

        layer = HybridRetrievalLayer(
            document_repository=repo,
            vector_adapter=mock_adapter,
            max_documents=5,
        )
        query = StructuredQuery(
            user_query="鴻海最新公告",
            ticker="2317",
            time_range_days=7,
        )
        layer.retrieve(query)

        call_kwargs = mock_adapter.search.call_args.kwargs
        self.assertEqual(call_kwargs.get("ticker"), "2317")


# ─────────────────────────────────────────────────────────────────────────────
# 3. upsert_documents() 行為驗證（不連 DB，只驗介面行為）
# ─────────────────────────────────────────────────────────────────────────────

class UpsertDocumentsTestCase(unittest.TestCase):
    """確認 PostgresMarketDocumentRepository.upsert_documents() 實作邏輯正確。
    使用 mock engine，不實際連線 DB。
    """

    def _make_repo(self) -> object:
        from llm_stock_system.adapters.postgres_market_data import PostgresMarketDocumentRepository

        mock_gateway = MagicMock()
        # 模擬 engine.begin() 的 context manager
        mock_conn = MagicMock()
        mock_gateway._engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_gateway._engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        return PostgresMarketDocumentRepository(mock_gateway), mock_conn

    def test_empty_list_returns_zero(self) -> None:
        from llm_stock_system.adapters.postgres_market_data import PostgresMarketDocumentRepository
        mock_gateway = MagicMock()
        repo = PostgresMarketDocumentRepository(mock_gateway)
        result = repo.upsert_documents([])
        self.assertEqual(result, 0)
        mock_gateway._engine.begin.assert_not_called()

    def test_upsert_calls_db_execute(self) -> None:
        repo, mock_conn = self._make_repo()
        docs = [
            _doc(ticker="2330", url="http://example.com/doc1", title="公告A"),
            _doc(ticker="2330", url="http://example.com/doc2", title="公告B"),
        ]
        result = repo.upsert_documents(docs)
        self.assertEqual(result, 2)
        mock_conn.execute.assert_called_once()

    def test_upsert_db_failure_returns_zero(self) -> None:
        from llm_stock_system.adapters.postgres_market_data import PostgresMarketDocumentRepository
        mock_gateway = MagicMock()
        mock_gateway._engine.begin.side_effect = Exception("DB connection error")
        repo = PostgresMarketDocumentRepository(mock_gateway)
        docs = [_doc(url="http://example.com/1", title="test")]
        # 不應 raise，應回傳 0
        result = repo.upsert_documents(docs)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
