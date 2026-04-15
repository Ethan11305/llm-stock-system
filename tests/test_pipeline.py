from pathlib import Path
import unittest

from llm_stock_system.adapters.llm import RuleBasedSynthesisClient
from llm_stock_system.adapters.repositories import InMemoryDocumentRepository, InMemoryQueryLogStore
from llm_stock_system.core.enums import ConfidenceLight, ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, QueryRequest
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


class QueryPipelineTestCase(unittest.TestCase):
    def test_input_layer_detects_eps_dividend_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="長榮 (2603) 去年全年的 EPS 表現，以及市場對其今年股利發放的預期？")
        )

        self.assertEqual(query.ticker, "2603")
        self.assertEqual(query.question_type, "eps_dividend_review")
        self.assertEqual(query.time_range_days, 365)

    def test_input_layer_detects_dividend_yield_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="聯發科 (2454) 最新的配息政策及換算目前的現金殖利率是多少？")
        )

        self.assertEqual(query.ticker, "2454")
        self.assertEqual(query.question_type, "dividend_yield_review")
        self.assertEqual(query.time_range_days, 365)

    def test_input_layer_detects_ex_dividend_performance(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="長榮航(2618)在除權息當天的填息表現與市場反應？")
        )

        self.assertEqual(query.ticker, "2618")
        self.assertEqual(query.question_type, "ex_dividend_performance")
        self.assertEqual(query.time_range_days, 365)

    def test_input_layer_detects_technical_indicator_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="欣興 (3037) 目前的 MACD、布林通道與均線乖離是否顯示過熱？")
        )

        self.assertEqual(query.ticker, "3037")
        self.assertEqual(query.question_type, "technical_indicator_review")
        self.assertEqual(query.time_range_days, 30)

    def test_input_layer_detects_season_line_margin_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="緯穎 (6669) 的股價近期是否跌破季線？市場對其融資餘額過高的看法。")
        )

        self.assertEqual(query.ticker, "6669")
        self.assertEqual(query.question_type, "season_line_margin_review")
        self.assertEqual(query.time_range_days, 90)

    def test_input_layer_detects_theme_impact_review_with_spaced_company_name(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="電動車普及率放緩，這對電池正極材料商（如美 琪瑪）的短線資訊有哪些？")
        )

        self.assertEqual(query.ticker, "4721")
        self.assertEqual(query.company_name, "美琪瑪")
        self.assertEqual(query.question_type, "theme_impact_review")
        self.assertEqual(query.time_range_days, 30)

    def test_input_layer_detects_revenue_growth_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="鴻海 (2317) 目前的 AI 伺服器營收占比與 2026 年的成長預測？？")
        )

        self.assertEqual(query.ticker, "2317")
        self.assertEqual(query.question_type, "revenue_growth_review")
        self.assertEqual(query.time_range_days, 90)

    def test_query_returns_grounded_response(self) -> None:
        pipeline = build_pipeline()
        response = pipeline.handle_query(QueryRequest(query="2330 最近 7 天有什麼重點？"))

        self.assertIsNotNone(response.query_id)
        self.assertGreaterEqual(len(response.sources), 1)
        self.assertIn(response.confidence_light.value, {"green", "yellow", "red"})

    def test_company_alias_routes_to_matching_stock(self) -> None:
        # Use a market-summary query so routing is tested without triggering
        # the directional-price cap (which correctly returns low confidence for
        # price-prediction queries backed by a single announcement document).
        pipeline = build_pipeline()
        response = pipeline.handle_query(QueryRequest(query="鴻海最近有什麼重要消息？"))

        self.assertTrue(all("台積電" not in source.title for source in response.sources))

    def test_price_range_query_returns_matching_stock_data(self) -> None:
        pipeline = build_pipeline()
        response = pipeline.handle_query(QueryRequest(query="華邦電最近30天最高點與最低點股價？"))

        self.assertIn("華邦電", response.summary)
        self.assertIn("最高價", response.summary)
        self.assertIn("最低價", response.summary)
        self.assertTrue(all("華邦電" in source.title for source in response.sources))

    def test_unknown_stock_does_not_fall_back_to_other_companies(self) -> None:
        pipeline = build_pipeline()
        response = pipeline.handle_query(QueryRequest(query="不存在公司最近30天最高點與最低點股價？"))

        self.assertEqual(response.summary, "資料不足，無法確認。")
        self.assertEqual(len(response.sources), 0)

    def test_validation_downgrades_insufficient_summary(self) -> None:
        validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)
        query = InputLayer().parse(
            QueryRequest(query="長榮航(2618)在除權息當天的填息表現與市場反應？")
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="長榮航股利政策",
                    excerpt="最新現金股利約 1.80 元。",
                    source_name="FinMind TaiwanStockDividend",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/1",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="長榮航股利政策補充",
                    excerpt="資料仍不足以確認當天填息率。",
                    source_name="FinMind TaiwanStockDividend",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/2",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.MOSTLY_CONSISTENT,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=1.0,
        )
        answer_draft = AnswerDraft(
            summary="資料不足，無法確認。",
            highlights=["現有證據不足"],
            facts=[],
            impacts=[],
            risks=["風險 1", "風險 2", "風險 3"],
            sources=[],
        )

        result = validation.validate(query, governance_report, answer_draft)

        self.assertEqual(result.confidence_light, ConfidenceLight.RED)
        self.assertLessEqual(result.confidence_score, 0.25)

    def test_rule_based_summary_mentions_new_technical_indicators(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="欣興 (3037) 的 MACD、布林通道與均線乖離是否顯示過熱？")
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="欣興技術指標快照",
                    excerpt=(
                        "最新收盤價約 210.00 元。"
                        "RSI14 約 71.20。"
                        "K 值約 82.10，D 值約 78.30。"
                        "MACD 線約 3.20，Signal 線約 2.70，Histogram 約 0.50。"
                        "布林通道上軌約 215.00 元，中軌約 205.00 元，下軌約 195.00 元。"
                        "MA5 約 209.00 元，MA20 約 205.00 元。"
                        "MA5 乖離率約 3.25%，MA20 乖離率約 2.44%。"
                        "已進入超買區。"
                    ),
                    source_name="FinMind TaiwanStockPrice",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/technical-snapshot",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="欣興技術指標判讀",
                    excerpt="最新交易日為 2026-04-07。MACD 動能偏多。股價接近布林上軌。均線乖離維持溫和偏熱。",
                    source_name="FinMind TaiwanStockPrice",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/technical-assessment",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=1.0,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("MACD 動能偏多", draft.summary)
        self.assertIn("接近布林上軌", draft.summary)
        self.assertIn("MA5 乖離率約 3.25%", draft.summary)

    def test_rule_based_summary_mentions_season_line_and_margin(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="緯穎 (6669) 的股價近期是否跌破季線？市場對其融資餘額過高的看法。")
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="緯穎季線位置觀察",
                    excerpt=(
                        "最新交易日為 2026-04-07。"
                        "最新收盤價約 2450.00 元。"
                        "季線(MA60)約 2488.00 元。"
                        "與季線乖離約 -1.53%。"
                        "目前近期跌破季線。"
                    ),
                    source_name="FinMind TaiwanStockPrice",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/season-line",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="緯穎融資餘額觀察",
                    excerpt=(
                        "最新資料日為 2026-04-07。"
                        "最新融資餘額約 3200 張。"
                        "融資限額約 9800 張。"
                        "融資使用率約 32.65%。"
                        "近 20 個交易日平均融資餘額約 2780 張。"
                        "相較近 20 日平均變動約 15.11%。"
                        "近 5 個交易日融資餘額變動約 +180 張。"
                        "若以融資使用率與近 20 日均值推估，籌碼面屬中性偏高。"
                    ),
                    source_name="FinMind TaiwanStockMarginPurchaseShortSale",
                    source_tier=SourceTier.HIGH,
                    url="https://example.com/margin-balance",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=1.0,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=1.0,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("近期跌破季線", draft.summary)
        self.assertIn("融資使用率約 32.65%", draft.summary)
        self.assertIn("中性偏高", draft.summary)

    def test_rule_based_summary_mentions_theme_impact(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="電動車普及率放緩，這對電池正極材料商（如美 琪瑪）的短線資訊有哪些？")
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="油價破百、鈷價飆升！美琪瑪憑城市採礦技術躍升能源轉型核心",
                    excerpt="近期公開資訊仍偏向能源轉型與材料題材，並提到鈷價上升。",
                    source_name="理財周刊",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/theme-1",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=0.75,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="美琪瑪受益於產品組合優化與產能利用率提升",
                    excerpt="短線訊號仍以產品組合、庫存效應與產能利用率為主。",
                    source_name="cmoney.tw",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/theme-2",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=0.75,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=0.5,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("能源轉型", draft.summary)
        self.assertIn("產品組合", draft.summary)
        self.assertIn("原料價格", draft.summary)

    def test_rule_based_summary_mentions_revenue_growth_review(self) -> None:
        query = InputLayer().parse(
            QueryRequest(query="鴻海 (2317) 目前的 AI 伺服器營收占比與 2026 年的成長預測？？")
        )
        governance_report = GovernanceReport(
            evidence=[
                Evidence(
                    document_id="1",
                    title="鴻海 AI 伺服器出貨強勁 3月營收創同期新高",
                    excerpt="公開資訊顯示 AI 伺服器仍是 2026 年的重要成長動能。",
                    source_name="自由財經",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/revenue-1",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=0.75,
                    corroboration_count=1,
                ),
                Evidence(
                    document_id="2",
                    title="鴻海去年獲利遜於預期 AI伺服器成2026成長關鍵動能",
                    excerpt="現有來源未一致揭露明確營收占比，但多個來源都將 AI 伺服器視為 2026 成長關鍵。",
                    source_name="yesmedia.com.tw",
                    source_tier=SourceTier.MEDIUM,
                    url="https://example.com/revenue-2",
                    published_at=SAMPLE_DOCUMENTS[0].published_at,
                    support_score=0.75,
                    corroboration_count=1,
                ),
            ],
            sufficiency=SufficiencyStatus.SUFFICIENT,
            consistency=ConsistencyStatus.CONSISTENT,
            freshness=FreshnessStatus.RECENT,
            high_trust_ratio=0.5,
        )

        draft = RuleBasedSynthesisClient().synthesize(query, governance_report, "")

        self.assertIn("AI 伺服器", draft.summary)
        self.assertIn("2026 年的重要成長動能", draft.summary)
        self.assertIn("未一致揭露明確營收占比", draft.summary)


if __name__ == "__main__":
    unittest.main()
