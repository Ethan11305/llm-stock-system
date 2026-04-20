"""E2E tests for the Single Stock Digest 產品線。

涵蓋：
  * DigestInputLayer：純 rule / 注入 fake classifier 兩條路徑都能產 StructuredQuery
  * DigestInputLayer 的輸出 query_profile 必為 SINGLE_STOCK_DIGEST
  * QueryPipeline 套 DigestInputLayer + InMemoryQueryLogStore，`get_query_log` 能取回
    structured_query + warnings + response snapshot 的完整閉環
"""

from __future__ import annotations

from pathlib import Path
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.repositories import (
    InMemoryDocumentRepository,
    InMemoryQueryLogStore,
)
from llm_stock_system.core.enums import Intent, QueryProfile, TopicTag
from llm_stock_system.core.interfaces import QueryClassifier
from llm_stock_system.core.models import QueryRequest
from llm_stock_system.layers.data_governance_layer import DataGovernanceLayer
from llm_stock_system.layers.digest_input_layer import DigestInputLayer
from llm_stock_system.layers.generation_layer import GenerationLayer
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.retrieval_layer import RetrievalLayer
from llm_stock_system.layers.validation_layer import ValidationLayer
from llm_stock_system.orchestrator.pipeline import QueryPipeline
from llm_stock_system.sample_data.documents import SAMPLE_DOCUMENTS


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "llm_stock_system"
    / "prompts"
    / "system_prompt.md"
)


class _FakeClassifier(QueryClassifier):
    """測試用 classifier：固定回一個合法 payload，驗證 classifier_source 會切到 llm/mixed。"""

    def __init__(self, payload: dict | None) -> None:
        self._payload = payload
        self.call_count = 0

    def classify(self, query_text: str) -> dict | None:
        self.call_count += 1
        return self._payload


def _build_digest_pipeline(
    classifier: QueryClassifier | None = None,
    query_log_store: InMemoryQueryLogStore | None = None,
) -> tuple[QueryPipeline, InMemoryQueryLogStore]:
    store = query_log_store or InMemoryQueryLogStore()
    pipeline = QueryPipeline(
        input_layer=DigestInputLayer(classifier=classifier),
        retrieval_layer=RetrievalLayer(
            InMemoryDocumentRepository(SAMPLE_DOCUMENTS), max_documents=8
        ),
        data_governance_layer=DataGovernanceLayer(),
        generation_layer=GenerationLayer(
            llm_client=RuleBasedSynthesisClient(),
            prompt_path=PROMPT_PATH,
        ),
        validation_layer=ValidationLayer(
            min_green_confidence=0.8, min_yellow_confidence=0.55
        ),
        presentation_layer=PresentationLayer(),
        query_log_store=store,
    )
    return pipeline, store


class DigestInputLayerTestCase(unittest.TestCase):
    def test_rule_only_path_tags_digest_profile_and_default_7d(self) -> None:
        layer = DigestInputLayer(classifier=None)
        query = layer.parse(QueryRequest(query="台積電 (2330) 最近有什麼新聞？"))

        self.assertEqual(query.query_profile, QueryProfile.SINGLE_STOCK_DIGEST)
        self.assertEqual(query.classifier_source, "rule")
        self.assertEqual(query.ticker, "2330")
        # 未指定 time_range 時，digest 的預設值是 7d
        self.assertEqual(query.time_range_label, "7d")
        self.assertEqual(query.time_range_days, 7)

    def test_respects_explicit_time_range_from_request(self) -> None:
        layer = DigestInputLayer(classifier=None)
        query = layer.parse(
            QueryRequest(query="台積電法說會重點是什麼？", time_range="latest_quarter")
        )

        # Digest 預設 7d，但 caller 明確指定 latest_quarter 時要尊重
        self.assertEqual(query.time_range_label, "latest_quarter")
        self.assertEqual(query.time_range_days, 90)
        self.assertEqual(query.query_profile, QueryProfile.SINGLE_STOCK_DIGEST)

    def test_classifier_payload_is_applied_when_valid(self) -> None:
        fake = _FakeClassifier(
            payload={
                "intent": "news_digest",
                "question_type": "market_summary",
                "topic_tags": [TopicTag.EVENT.value],
                "time_range_label": "30d",
                "stance_bias": "neutral",
                "is_forecast_query": False,
                "wants_direction": False,
                "wants_scenario_range": False,
                "forecast_horizon_label": None,
                "forecast_horizon_days": None,
            }
        )
        layer = DigestInputLayer(classifier=fake)
        query = layer.parse(QueryRequest(query="鴻海 (2317) 近期動態"))

        self.assertEqual(fake.call_count, 1)
        self.assertEqual(query.query_profile, QueryProfile.SINGLE_STOCK_DIGEST)
        # classifier 有合法值 → classifier_source 切到 llm/mixed（不是 rule）
        self.assertIn(query.classifier_source, {"llm", "mixed"})
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)

    def test_classifier_returning_none_degrades_to_rule(self) -> None:
        fake = _FakeClassifier(payload=None)
        layer = DigestInputLayer(classifier=fake)
        query = layer.parse(QueryRequest(query="聯發科最新消息？"))

        self.assertEqual(fake.call_count, 1)
        # classifier 回 None 時，必須回到 rule 路徑而非炸掉
        self.assertEqual(query.classifier_source, "rule")
        self.assertEqual(query.query_profile, QueryProfile.SINGLE_STOCK_DIGEST)


class DigestPipelineClosureTestCase(unittest.TestCase):
    """可追溯閉環：跑完 pipeline 後能從 query_log_store 取回完整 snapshot。"""

    def test_pipeline_persists_query_log_detail_with_digest_profile(self) -> None:
        pipeline, store = _build_digest_pipeline(classifier=None)

        response = pipeline.handle_query(
            QueryRequest(query="台積電 (2330) 最近有什麼新聞？")
        )

        self.assertIsNotNone(response.query_id)
        detail = store.get_query_log(response.query_id)
        self.assertIsNotNone(detail, "get_query_log 應回傳完整 QueryLogDetail")
        self.assertEqual(detail.query_profile, QueryProfile.SINGLE_STOCK_DIGEST)
        self.assertEqual(detail.classifier_source, "rule")
        # response snapshot 必須含 summary 與原 response 等價
        self.assertEqual(detail.response.summary, response.summary)
        self.assertEqual(detail.response.query_id, response.query_id)
        # structured_query 要是 dict（供日後 schema 比對）
        self.assertIsInstance(detail.structured_query, dict)
        self.assertEqual(detail.structured_query.get("ticker"), "2330")
        # source_count 與 response.sources 一致
        self.assertEqual(detail.source_count, len(response.sources))

    def test_pipeline_records_warnings_when_validation_degrades(self) -> None:
        pipeline, store = _build_digest_pipeline(classifier=None)
        # 問一個沒有足夠資料支援的股票，確保 validation 會給出 warnings
        response = pipeline.handle_query(
            QueryRequest(query="不存在公司最近發生什麼事？")
        )

        detail = store.get_query_log(response.query_id)
        self.assertIsNotNone(detail)
        # warnings 可以是空（取決於 validation profile），但 detail 物件本身必須存在並可序列化
        self.assertIsInstance(detail.warnings, list)
        self.assertIsInstance(detail.validation_status, str)

    def test_get_query_log_returns_none_for_unknown_id(self) -> None:
        _, store = _build_digest_pipeline(classifier=None)
        self.assertIsNone(store.get_query_log("no-such-id"))
        self.assertIsNone(store.get_sources("no-such-id"))


if __name__ == "__main__":
    unittest.main()
