"""P3 UI parity 測試。

驗證即時回應（QueryResponse）與回查（QueryLogDetail）兩條路徑的資料契約一致：
  * 即時 QueryResponse 帶有 warnings / classifierSource / queryProfile 三個新欄位
  * digest 路徑：即時即可看到 queryProfile=single_stock_digest，不必打 /query-log
  * legacy 路徑：queryProfile=legacy（避免既有 caller 誤判）
  * QueryLogDetail 頂層欄位序列化為 camelCase，與 QueryResponse alias 風格一致
  * /query-log/{id}.response 與當場 POST /query 的 response 主體欄位等價
"""

from __future__ import annotations

from pathlib import Path
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.repositories import (
    InMemoryDocumentRepository,
    InMemoryQueryLogStore,
)
from llm_stock_system.core.enums import QueryProfile
from llm_stock_system.core.models import QueryLogDetail, QueryRequest, QueryResponse
from llm_stock_system.layers.data_governance_layer import DataGovernanceLayer
from llm_stock_system.layers.digest_input_layer import DigestInputLayer
from llm_stock_system.layers.generation_layer import GenerationLayer
from llm_stock_system.layers.input_layer import InputLayer
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


def _make_pipeline(
    input_layer, store: InMemoryQueryLogStore | None = None
) -> tuple[QueryPipeline, InMemoryQueryLogStore]:
    store = store or InMemoryQueryLogStore()
    pipeline = QueryPipeline(
        input_layer=input_layer,
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


class QueryResponseFieldPresenceTestCase(unittest.TestCase):
    """即時 QueryResponse 要直接帶有 P3 新增的三個欄位。"""

    def test_response_exposes_warnings_classifier_source_query_profile(self) -> None:
        pipeline, _ = _make_pipeline(InputLayer())
        response = pipeline.handle_query(
            QueryRequest(query="台積電 (2330) 最近有什麼新聞？")
        )

        # 三個欄位都必須存在（即使是空 list / 預設值）
        self.assertIsInstance(response.warnings, list)
        self.assertIsInstance(response.classifier_source, str)
        self.assertIsInstance(response.query_profile, QueryProfile)

    def test_legacy_path_marks_response_query_profile_as_legacy(self) -> None:
        pipeline, _ = _make_pipeline(InputLayer())
        response = pipeline.handle_query(
            QueryRequest(query="聯發科 (2454) 最近法說重點？")
        )
        self.assertEqual(response.query_profile, QueryProfile.LEGACY)

    def test_digest_path_marks_response_query_profile_as_digest(self) -> None:
        pipeline, _ = _make_pipeline(DigestInputLayer(classifier=None))
        response = pipeline.handle_query(
            QueryRequest(query="鴻海 (2317) 近期動態")
        )
        # 即時回應就能看出這條是 digest 路徑，不必二次查詢 /query-log
        self.assertEqual(response.query_profile, QueryProfile.SINGLE_STOCK_DIGEST)

    def test_warnings_propagate_from_validation_to_response(self) -> None:
        """validation_result.warnings 必須出現在即時 response.warnings。"""
        pipeline, store = _make_pipeline(DigestInputLayer(classifier=None))
        response = pipeline.handle_query(
            QueryRequest(query="不存在公司最近發生什麼事？")
        )

        detail = store.get_query_log(response.query_id)
        self.assertIsNotNone(detail)
        # 即時 warnings 應與 log 內記錄的一致（UI parity 的核心承諾）
        self.assertEqual(response.warnings, detail.warnings)


class QueryResponseSerializationTestCase(unittest.TestCase):
    """JSON 序列化必須是 camelCase（與既有 dataStatus / confidenceLight 一致）。"""

    def test_response_serializes_new_fields_as_camel_case(self) -> None:
        pipeline, _ = _make_pipeline(DigestInputLayer(classifier=None))
        response = pipeline.handle_query(QueryRequest(query="鴻海 (2317) 近期動態"))

        dumped = response.model_dump(by_alias=True)
        # 既有欄位仍保持 camelCase
        self.assertIn("dataStatus", dumped)
        self.assertIn("confidenceLight", dumped)
        self.assertIn("confidenceScore", dumped)
        # P3 新增欄位也要 camelCase
        self.assertIn("classifierSource", dumped)
        self.assertIn("queryProfile", dumped)
        # warnings 是複數名詞，本來就是一個字，不需 alias
        self.assertIn("warnings", dumped)
        # 不應同時出現 snake_case 與 camelCase
        self.assertNotIn("classifier_source", dumped)
        self.assertNotIn("query_profile", dumped)

    def test_query_log_detail_top_level_serializes_as_camel_case(self) -> None:
        pipeline, store = _make_pipeline(DigestInputLayer(classifier=None))
        response = pipeline.handle_query(QueryRequest(query="鴻海 (2317) 近期動態"))

        detail = store.get_query_log(response.query_id)
        self.assertIsNotNone(detail)
        dumped = detail.model_dump(by_alias=True)

        expected_keys = {
            "queryId",
            "queryProfile",
            "classifierSource",
            "validationStatus",
            "warnings",
            "sourceCount",
            "schemaVersion",
            "structuredQuery",
            "response",
        }
        self.assertEqual(expected_keys, set(dumped.keys()))
        # 舊的 snake_case key 不該再出現在 by_alias 輸出
        for snake_key in (
            "query_id",
            "query_profile",
            "classifier_source",
            "validation_status",
            "source_count",
            "schema_version",
            "structured_query",
        ):
            self.assertNotIn(snake_key, dumped)

    def test_query_log_detail_populate_by_name_still_works(self) -> None:
        """給 populate_by_name=True 後，用 snake_case 字典建構也應成功。

        讓既有的 Python caller（repositories.save()）不因為加 alias 而壞掉。
        """
        pipeline, store = _make_pipeline(InputLayer())
        response = pipeline.handle_query(
            QueryRequest(query="台積電 (2330) 最近有什麼新聞？")
        )
        detail = store.get_query_log(response.query_id)
        self.assertIsInstance(detail, QueryLogDetail)
        # 能透過 snake_case 屬性存取
        self.assertEqual(detail.query_id, response.query_id)
        self.assertEqual(detail.query_profile, response.query_profile)
        self.assertEqual(detail.classifier_source, response.classifier_source)


class LiveVsLogParityTestCase(unittest.TestCase):
    """即時 response 與 /query-log/{id}.response 的主體內容必須等價。"""

    def test_log_response_matches_live_response(self) -> None:
        pipeline, store = _make_pipeline(InputLayer())
        response = pipeline.handle_query(
            QueryRequest(query="台積電 (2330) 最近有什麼新聞？")
        )

        detail = store.get_query_log(response.query_id)
        self.assertIsNotNone(detail)

        # response 嵌入的物件必須是 QueryResponse（而非 dict、避免欄位漂移）
        self.assertIsInstance(detail.response, QueryResponse)
        # 主體欄位逐一核對
        self.assertEqual(detail.response.query_id, response.query_id)
        self.assertEqual(detail.response.summary, response.summary)
        self.assertEqual(detail.response.highlights, response.highlights)
        self.assertEqual(detail.response.facts, response.facts)
        self.assertEqual(detail.response.impacts, response.impacts)
        self.assertEqual(detail.response.risks, response.risks)
        self.assertEqual(
            detail.response.confidence_score, response.confidence_score
        )
        self.assertEqual(detail.response.sources, response.sources)
        # 新的 UI parity 欄位也一致
        self.assertEqual(detail.response.warnings, response.warnings)
        self.assertEqual(
            detail.response.classifier_source, response.classifier_source
        )
        self.assertEqual(detail.response.query_profile, response.query_profile)

    def test_digest_log_response_keeps_digest_profile(self) -> None:
        pipeline, store = _make_pipeline(DigestInputLayer(classifier=None))
        response = pipeline.handle_query(QueryRequest(query="鴻海 (2317) 近期動態"))
        detail = store.get_query_log(response.query_id)
        self.assertIsNotNone(detail)
        # 回查路徑也要能立刻判斷這是 digest（兩個維度都標：頂層 + 嵌入 response）
        self.assertEqual(detail.query_profile, QueryProfile.SINGLE_STOCK_DIGEST)
        self.assertEqual(
            detail.response.query_profile, QueryProfile.SINGLE_STOCK_DIGEST
        )


if __name__ == "__main__":
    unittest.main()
