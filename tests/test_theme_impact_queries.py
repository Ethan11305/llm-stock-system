from pathlib import Path
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.repositories import InMemoryDocumentRepository, InMemoryQueryLogStore
from llm_stock_system.core.enums import Intent
from llm_stock_system.core.models import QueryRequest
from llm_stock_system.layers.data_governance_layer import DataGovernanceLayer
from llm_stock_system.layers.generation_layer import GenerationLayer
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.presentation_layer import PresentationLayer
from llm_stock_system.layers.retrieval_layer import RetrievalLayer
from llm_stock_system.layers.validation_layer import ValidationLayer
from llm_stock_system.orchestrator.pipeline import QueryPipeline
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


class ThemeImpactQueryTestCase(unittest.TestCase):
    def test_input_layer_detects_semiconductor_equipment_theme_query(self) -> None:
        query = InputLayer().parse(
            QueryRequest(
                query="如果 ASML（艾司摩爾）最新展望不如預期，搜尋這對台灣半導體設備族群（如 3680 家登、6187 萬潤）的最新利空分析與情緒影響"
            )
        )

        self.assertEqual(query.ticker, "3680")
        self.assertEqual(query.company_name, "家登")
        self.assertEqual(query.comparison_ticker, "6187")
        self.assertEqual(query.comparison_company_name, "萬潤")
        self.assertEqual(query.intent, Intent.NEWS_DIGEST)
        self.assertEqual(query.time_range_days, 30)

    def test_pipeline_returns_grounded_theme_impact_summary(self) -> None:
        pipeline = build_pipeline()

        response = pipeline.handle_query(
            QueryRequest(
                query="如果 ASML（艾司摩爾）最新展望不如預期，搜尋這對台灣半導體設備族群（如 3680 家登、6187 萬潤）的最新利空分析與情緒影響"
            )
        )

        self.assertIn("家登、萬潤", response.summary)
        self.assertIn("ASML", response.summary)
        self.assertGreaterEqual(len(response.sources), 2)
        self.assertNotEqual(response.confidence_light.value, "red")


if __name__ == "__main__":
    unittest.main()
