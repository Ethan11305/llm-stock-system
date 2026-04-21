from pathlib import Path
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.repositories import InMemoryDocumentRepository, InMemoryQueryLogStore
from llm_stock_system.layers.data_governance_layer import DataGovernanceLayer
from llm_stock_system.layers.generation_layer import GenerationLayer
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.retrieval_layer import RetrievalLayer
from llm_stock_system.layers.validation_layer import ValidationLayer
from llm_stock_system.orchestrator.pipeline import QueryPipeline
from llm_stock_system.core.enums import Intent
from llm_stock_system.core.models import QueryRequest
from llm_stock_system.sample_data.documents import SAMPLE_DOCUMENTS


def build_pipeline() -> QueryPipeline:
    return QueryPipeline(
        input_layer=InputLayer(),
        retrieval_layer=RetrievalLayer(InMemoryDocumentRepository(SAMPLE_DOCUMENTS), max_documents=8),
        data_governance_layer=DataGovernanceLayer(),
        generation_layer=GenerationLayer(
            llm_client=RuleBasedSynthesisClient(),
            prompt_path=Path(__file__).resolve().parents[1]
            / "src"
            / "llm_stock_system"
            / "prompts"
            / "system_prompt.md",
        ),
        validation_layer=ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55),
        presentation_layer=PresentationLayer(),
        query_log_store=InMemoryQueryLogStore(),
    )


class StockResolutionConflictTestCase(unittest.TestCase):
    def test_name_ticker_conflict_prefers_company_name(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="大 同 (2317)近期是否有處分土地或業外資產的計畫公佈？預計貢獻 EPS 多少？這筆帳款預計在哪一季入帳？"
            )
        )

        self.assertEqual(query.ticker, "2371")
        self.assertEqual(query.company_name, "大同")
        self.assertIsNone(query.comparison_ticker)
        self.assertEqual(query.intent, Intent.INVESTMENT_ASSESSMENT)

    def test_pipeline_does_not_leak_conflicting_ticker_documents(self) -> None:
        pipeline = build_pipeline()

        response = pipeline.handle_query(
            QueryRequest(
                query="大 同 (2317)近期是否有處分土地或業外資產的計畫公佈？預計貢獻 EPS 多少？這筆帳款預計在哪一季入帳？"
            )
        )

        self.assertEqual(response.summary, "資料不足，無法確認。")
        self.assertEqual(len(response.sources), 0)


if __name__ == "__main__":
    unittest.main()
