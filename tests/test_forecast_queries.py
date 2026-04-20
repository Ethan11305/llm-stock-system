"""Tests for the forecast credibility (預測型查詢可信化) feature.

Covers:
1. Input layer classification — forecast vs. historical routing.
2. Forecast semantic field population.
3. ForecastBlock schema and guardrails.
4. Validation layer confidence caps per forecast mode.
5. Integration: forward-looking vs. backward-looking distinction.
"""

import unittest
from datetime import date, datetime

from llm_stock_system.core.enums import (
    ConfidenceLight,
    ConsistencyStatus,
    ForecastDirection,
    ForecastMode,
    FreshnessStatus,
    SourceTier,
    SufficiencyStatus,
    Topic,
)
from llm_stock_system.core.forecast import (
    apply_forecast_guardrail,
    apply_forecast_guardrail_to_list,
    build_forecast_block,
)
from llm_stock_system.core.models import (
    AnswerDraft,
    Evidence,
    ForecastBlock,
    ForecastWindow,
    GovernanceReport,
    QueryRequest,
    ScenarioRange,
    SourceCitation,
    StructuredQuery,
)
from llm_stock_system.layers.input_layer import InputLayer
from llm_stock_system.layers.validation_layer import ValidationLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evidence(title: str, excerpt: str, source_name: str = "moneydj") -> Evidence:
    return Evidence(
        document_id=f"doc-{abs(hash((title, excerpt))) % 100000}",
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=SourceTier.MEDIUM,
        url="https://example.com/article",
        published_at=datetime(2026, 4, 14, 10, 0, 0),
        support_score=0.8,
        corroboration_count=1,
    )


def make_report(*evidence: Evidence) -> GovernanceReport:
    return GovernanceReport(
        evidence=list(evidence),
        sufficiency=SufficiencyStatus.SUFFICIENT if evidence else SufficiencyStatus.INSUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT if evidence else ConsistencyStatus.CONFLICTING,
        freshness=FreshnessStatus.RECENT if evidence else FreshnessStatus.OUTDATED,
        high_trust_ratio=0.9 if evidence else 0.0,
    )


def make_empty_report() -> GovernanceReport:
    return make_report()


def make_forecast_query(
    user_query: str,
    ticker: str = "2330",
    company: str = "台積電",
    **kwargs,
) -> StructuredQuery:
    defaults = dict(
        user_query=user_query,
        ticker=ticker,
        company_name=company,
        topic=Topic.COMPOSITE,
        question_type="price_outlook",
        is_forecast_query=True,
        wants_direction=True,
        wants_scenario_range=True,
        forecast_horizon_label="未來一週",
        forecast_horizon_days=7,
    )
    defaults.update(kwargs)
    return StructuredQuery(**defaults)


def _empty_draft(forecast: ForecastBlock | None = None) -> AnswerDraft:
    return AnswerDraft(
        summary="資料不足，無法確認。",
        highlights=[],
        facts=[],
        impacts=[],
        risks=[],
        sources=[],
        forecast=forecast,
    )


def _draft_with_forecast(forecast: ForecastBlock, evidence: Evidence | None = None) -> AnswerDraft:
    sources = []
    if evidence:
        sources = [
            SourceCitation(
                title=evidence.title,
                source_name=evidence.source_name,
                source_tier=evidence.source_tier,
                url=evidence.url,
                published_at=evidence.published_at,
                excerpt=evidence.excerpt,
                support_score=evidence.support_score,
                corroboration_count=evidence.corroboration_count,
            )
        ]
    return AnswerDraft(
        summary="根據情境推估，方向偏多。",
        highlights=["情境推估"],
        facts=[],
        impacts=[],
        risks=["風險一", "風險二", "風險三"],
        sources=sources,
        forecast=forecast,
    )


# ===========================================================================
# 1. Input layer classification
# ===========================================================================

class ForecastClassificationTestCase(unittest.TestCase):
    """Forward-looking queries should route to price_outlook with forecast fields."""

    def setUp(self) -> None:
        self.layer = InputLayer()

    def _parse(self, query: str) -> StructuredQuery:
        return self.layer.parse(QueryRequest(query=query))

    def test_future_week_with_direction_and_range(self) -> None:
        """這一個星期台積電股價預期波動如何？ → price_outlook + is_forecast_query."""
        result = self._parse("這一個星期台積電股價預期波動如何？會上漲還是下跌？預估區間是多少？")
        self.assertEqual(result.question_type, "price_outlook")
        self.assertTrue(result.is_forecast_query)
        self.assertTrue(result.wants_direction)
        self.assertTrue(result.wants_scenario_range)

    def test_historical_range_stays_price_range(self) -> None:
        """近 7 天台積電股價區間是多少？ → price_range (NOT price_outlook)."""
        result = self._parse("近 7 天台積電股價區間是多少？")
        self.assertEqual(result.question_type, "price_range")
        self.assertFalse(result.is_forecast_query)

    def test_next_week_forecast(self) -> None:
        """下週台積電會漲嗎？ → price_outlook + forecast."""
        result = self._parse("下週台積電會漲嗎？")
        self.assertEqual(result.question_type, "price_outlook")
        self.assertTrue(result.is_forecast_query)
        self.assertTrue(result.wants_direction)

    def test_future_estimate_range(self) -> None:
        """接下來台積電預估區間 → price_outlook (not price_range)."""
        result = self._parse("接下來台積電預估區間是多少？")
        self.assertEqual(result.question_type, "price_outlook")
        self.assertTrue(result.is_forecast_query)
        self.assertTrue(result.wants_scenario_range)

    def test_this_week_forecast_demand(self) -> None:
        """這週台積電可能偏多還是偏空？"""
        result = self._parse("這週台積電可能偏多還是偏空？")
        self.assertEqual(result.question_type, "price_outlook")
        self.assertTrue(result.is_forecast_query)

    def test_pure_historical_high_low(self) -> None:
        """最近 30 天台積電最高價多少？ → price_range."""
        result = self._parse("最近 30 天台積電最高價多少？")
        self.assertEqual(result.question_type, "price_range")
        self.assertFalse(result.is_forecast_query)

    def test_horizon_label_one_week(self) -> None:
        result = self._parse("這一個星期台積電預期波動如何？")
        self.assertIn(result.forecast_horizon_label, ("本週", "未來一週"))
        self.assertEqual(result.forecast_horizon_days, 7)

    def test_horizon_label_one_month(self) -> None:
        result = self._parse("下個月台積電會漲嗎？預估區間？")
        self.assertEqual(result.forecast_horizon_label, "未來一個月")
        self.assertEqual(result.forecast_horizon_days, 30)

    # --- Regression: existing directional patterns still work ---

    def test_existing_will_rise_still_works(self) -> None:
        result = self._parse("台積電明天會漲嗎？")
        self.assertEqual(result.question_type, "price_outlook")
        self.assertTrue(result.is_forecast_query)

    def test_existing_target_price_still_works(self) -> None:
        result = self._parse("華邦電 2344 未來半年目標價是多少？")
        self.assertEqual(result.question_type, "price_outlook")
        self.assertTrue(result.is_forecast_query)


# ===========================================================================
# 2. ForecastBlock schema and guardrails
# ===========================================================================

class ForecastBlockTestCase(unittest.TestCase):
    """ForecastBlock construction and guardrail tests."""

    def test_scenario_estimate_with_analyst_range(self) -> None:
        """Analyst target prices → mode=scenario_estimate, has range."""
        query = make_forecast_query("下週台積電目標價多少？")
        evidence = make_evidence(
            "法人看好台積電", "外資目標價 950-1050 元",
        )
        report = make_report(evidence)

        block = build_forecast_block(query, report, reference_date=date(2026, 4, 16))
        self.assertEqual(block.mode, ForecastMode.SCENARIO_ESTIMATE)
        self.assertIsNotNone(block.scenario_range)
        self.assertEqual(block.scenario_range.low, 950.0)
        self.assertEqual(block.scenario_range.high, 1050.0)
        self.assertEqual(block.scenario_range.basis_type, "analyst_target")
        self.assertEqual(block.forecast_window.start_date, "2026-04-16")
        self.assertEqual(block.forecast_window.end_date, "2026-04-23")

    def test_historical_proxy_mode(self) -> None:
        """Only price data, no analyst context → mode=historical_proxy."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "台積電近期走勢", "收盤價在 900 元到 950 元之間波動",
        )
        report = make_report(evidence)

        block = build_forecast_block(query, report)
        self.assertEqual(block.mode, ForecastMode.HISTORICAL_PROXY)
        self.assertIsNotNone(block.scenario_range)
        self.assertEqual(block.scenario_range.basis_type, "historical_proxy")
        self.assertTrue(any("歷史波動代理" in b for b in block.forecast_basis))

    def test_unsupported_mode(self) -> None:
        """No evidence at all → mode=unsupported."""
        query = make_forecast_query("下週台積電會漲嗎？")
        report = make_empty_report()

        block = build_forecast_block(query, report)
        self.assertEqual(block.mode, ForecastMode.UNSUPPORTED)
        self.assertIsNone(block.scenario_range)
        self.assertEqual(block.direction, ForecastDirection.UNDETERMINED)

    def test_scenario_estimate_no_range_with_direction(self) -> None:
        """Analyst context but no numeric range → scenario_estimate, no range, direction set."""
        query = make_forecast_query("下週台積電偏多還是偏空？")
        evidence = make_evidence(
            "分析師看多台積電", "法人觀點偏多，看多台積電，認為支撐仍在。看多看多。",
        )
        report = make_report(evidence)

        block = build_forecast_block(query, report)
        self.assertEqual(block.mode, ForecastMode.SCENARIO_ESTIMATE)
        self.assertIsNone(block.scenario_range)
        self.assertIn(block.direction, (ForecastDirection.BULLISH_BIAS, ForecastDirection.RANGE_BOUND))

    def test_guardrail_rewrites_overconfident_language(self) -> None:
        """Guardrail rewrites '一定會漲' → scenario language."""
        block = ForecastBlock(
            mode=ForecastMode.SCENARIO_ESTIMATE,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.BULLISH_BIAS,
        )
        result = apply_forecast_guardrail("台積電一定會漲。", block)
        self.assertNotIn("一定會漲", result)
        self.assertIn("情境推估", result)

    def test_guardrail_adds_historical_disclaimer(self) -> None:
        block = ForecastBlock(
            mode=ForecastMode.HISTORICAL_PROXY,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
        )
        result = apply_forecast_guardrail("近期高低區間約 900–950 元。", block)
        self.assertIn("歷史波動代理", result)


# ===========================================================================
# 3. Validation layer confidence caps
# ===========================================================================

class ForecastValidationCapTestCase(unittest.TestCase):
    """Forecast modes have specific confidence ceilings."""

    def setUp(self) -> None:
        self.validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    def _make_scenario_estimate_forecast(self) -> ForecastBlock:
        return ForecastBlock(
            mode=ForecastMode.SCENARIO_ESTIMATE,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.BULLISH_BIAS,
            scenario_range=ScenarioRange(low=950, high=1050, basis_type="analyst_target"),
        )

    def _make_historical_proxy_forecast(self) -> ForecastBlock:
        return ForecastBlock(
            mode=ForecastMode.HISTORICAL_PROXY,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.RANGE_BOUND,
            scenario_range=ScenarioRange(low=900, high=950, basis_type="historical_proxy"),
        )

    def _make_unsupported_forecast(self) -> ForecastBlock:
        return ForecastBlock(
            mode=ForecastMode.UNSUPPORTED,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.UNDETERMINED,
        )

    def test_scenario_estimate_never_green(self) -> None:
        """scenario_estimate capped at 0.65 → never GREEN (min_green=0.8)."""
        query = make_forecast_query("下週台積電目標價區間？")
        forecast = self._make_scenario_estimate_forecast()
        evidence = make_evidence("法人看好", "目標價 950 至 1050 元")
        report = make_report(evidence)
        draft = _draft_with_forecast(forecast, evidence)

        result = self.validation.validate(query, report, draft)
        self.assertLessEqual(result.confidence_score, 0.65)
        self.assertNotEqual(result.confidence_light, ConfidenceLight.GREEN)
        self.assertTrue(any("情境推估" in w for w in result.warnings))

    def test_historical_proxy_max_045(self) -> None:
        """historical_proxy capped at 0.45."""
        query = make_forecast_query("這週台積電預估區間？")
        forecast = self._make_historical_proxy_forecast()
        evidence = make_evidence("近期走勢", "收盤價 900–950 元")
        report = make_report(evidence)
        draft = _draft_with_forecast(forecast, evidence)

        result = self.validation.validate(query, report, draft)
        self.assertLessEqual(result.confidence_score, 0.45)
        self.assertTrue(any("歷史波動代理" in w for w in result.warnings))

    def test_unsupported_max_025(self) -> None:
        """unsupported capped at 0.25."""
        query = make_forecast_query("下週台積電會漲嗎？")
        forecast = self._make_unsupported_forecast()
        report = make_empty_report()
        draft = _empty_draft(forecast)

        result = self.validation.validate(query, report, draft)
        self.assertLessEqual(result.confidence_score, 0.25)
        self.assertTrue(any("前瞻依據" in w for w in result.warnings))

    def test_non_forecast_query_unaffected(self) -> None:
        """Non-forecast queries should not be subject to forecast caps."""
        query = StructuredQuery(
            user_query="近 7 天台積電股價區間是多少？",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="price_range",
            is_forecast_query=False,
        )
        evidence = make_evidence("台積電近期走勢", "近 7 天高低區間為 900–950 元")
        report = make_report(evidence)
        draft = AnswerDraft(
            summary="近 7 天高低區間為 900–950 元。",
            highlights=["近 7 天區間"],
            facts=[],
            impacts=[],
            risks=["風險一", "風險二", "風險三"],
            sources=[
                SourceCitation(
                    title=evidence.title,
                    source_name=evidence.source_name,
                    source_tier=evidence.source_tier,
                    url=evidence.url,
                    published_at=evidence.published_at,
                    excerpt=evidence.excerpt,
                    support_score=evidence.support_score,
                    corroboration_count=evidence.corroboration_count,
                )
            ],
        )

        result = self.validation.validate(query, report, draft)
        # Should NOT have any forecast-related warnings
        self.assertFalse(any("情境推估" in w or "歷史波動代理" in w or "前瞻依據" in w for w in result.warnings))


# ===========================================================================
# 4. Integration: forward vs. backward distinction
# ===========================================================================

class ForecastIntegrationTestCase(unittest.TestCase):
    """End-to-end: the same 'one week' phrasing, different temporal framing."""

    def setUp(self) -> None:
        self.input_layer = InputLayer()

    def test_future_week_is_forecast(self) -> None:
        """'這一個星期預估區間' = forward looking."""
        result = self.input_layer.parse(QueryRequest(query="這一個星期台積電預估區間是多少？"))
        self.assertEqual(result.question_type, "price_outlook")
        self.assertTrue(result.is_forecast_query)

    def test_past_week_is_historical(self) -> None:
        """'近 7 天區間' = backward looking."""
        result = self.input_layer.parse(QueryRequest(query="近 7 天台積電股價區間是多少？"))
        self.assertEqual(result.question_type, "price_range")
        self.assertFalse(result.is_forecast_query)

    def test_this_week_forecast_vs_past_one_week(self) -> None:
        """'這週可能' = forecast; '最近一週' = historical."""
        forecast_result = self.input_layer.parse(QueryRequest(query="這週台積電可能會漲嗎？"))
        self.assertEqual(forecast_result.question_type, "price_outlook")
        self.assertTrue(forecast_result.is_forecast_query)

        historical_result = self.input_layer.parse(QueryRequest(query="最近一週台積電股價區間是多少？"))
        self.assertEqual(historical_result.question_type, "price_range")
        self.assertFalse(historical_result.is_forecast_query)


# ===========================================================================
# 5. Forecast facet override: PRICE_HISTORY as required, not PE_VALUATION
# ===========================================================================

class ForecastFacetOverrideTestCase(unittest.TestCase):
    """is_forecast_query price_outlook should require PRICE_HISTORY, not PE_VALUATION."""

    def test_forecast_requires_price_history(self) -> None:
        from llm_stock_system.core.enums import DataFacet
        query = StructuredQuery(
            user_query="下週台積電會漲嗎？",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="price_outlook",
            is_forecast_query=True,
        )
        self.assertIn(DataFacet.PRICE_HISTORY, query.required_facets)
        self.assertNotIn(DataFacet.PE_VALUATION, query.required_facets)

    def test_forecast_demotes_pe_to_preferred(self) -> None:
        from llm_stock_system.core.enums import DataFacet
        query = StructuredQuery(
            user_query="這週台積電預估區間？",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="price_outlook",
            is_forecast_query=True,
        )
        self.assertIn(DataFacet.PE_VALUATION, query.preferred_facets)
        self.assertIn(DataFacet.NEWS, query.preferred_facets)

    def test_non_forecast_price_outlook_still_requires_pe(self) -> None:
        from llm_stock_system.core.enums import DataFacet
        query = StructuredQuery(
            user_query="台積電目標價多少？",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="price_outlook",
            is_forecast_query=False,
        )
        self.assertIn(DataFacet.PE_VALUATION, query.required_facets)

    def test_forecast_with_price_data_not_red(self) -> None:
        """Forecast query with price evidence should not be capped at 0.25 by facet miss."""
        validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)
        query = make_forecast_query("下週台積電預估區間？")
        # Evidence that would satisfy PRICE_HISTORY but not PE_VALUATION
        evidence = make_evidence(
            "台積電近期走勢",
            "收盤價在 900 元到 950 元之間波動",
            source_name="TaiwanStockPrice",
        )
        report = make_report(evidence)
        forecast = ForecastBlock(
            mode=ForecastMode.HISTORICAL_PROXY,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.RANGE_BOUND,
            scenario_range=ScenarioRange(low=900, high=950, basis_type="historical_proxy"),
        )
        draft = _draft_with_forecast(forecast, evidence)

        result = validation.validate(query, report, draft)
        # Should NOT be capped at 0.25 because of missing PE_VALUATION
        # (historical_proxy cap of 0.45 is acceptable, but not 0.25)
        self.assertGreater(result.confidence_score, 0.25)
        # Should not have "All required facets failed" warning
        self.assertFalse(any("All required facets failed" in w for w in result.warnings))


# ===========================================================================
# 6. Historical proxy range extraction: tightened to price-only evidence
# ===========================================================================

class HistoricalProxyExtractionTestCase(unittest.TestCase):
    """Historical proxy range should only pick up genuine stock price data."""

    def test_eps_not_treated_as_price(self) -> None:
        """EPS 22 元 should NOT be extracted as a price range anchor."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "台積電財報",
            "台積電去年 EPS 為 22 元，毛利率 55%。",
            source_name="FinancialStatements",
        )
        report = make_report(evidence)
        block = build_forecast_block(query, report)
        # Should not build a range from EPS 22
        if block.scenario_range is not None:
            # If a range somehow formed, it shouldn't include 22
            self.assertNotEqual(block.scenario_range.low, 22.0)

    def test_price_source_extracts_range(self) -> None:
        """TaiwanStockPrice evidence should have prices extracted."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "台積電收盤走勢",
            "近五日收盤價分別為 900 元、910 元、920 元、905 元、915 元",
            source_name="TaiwanStockPrice",
        )
        report = make_report(evidence)
        block = build_forecast_block(query, report)
        self.assertEqual(block.mode, ForecastMode.HISTORICAL_PROXY)
        self.assertIsNotNone(block.scenario_range)
        self.assertEqual(block.scenario_range.low, 900.0)
        self.assertEqual(block.scenario_range.high, 920.0)

    def test_contextual_keyword_extraction(self) -> None:
        """Non-price source but with '收盤價' context should extract."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "台積電分析",
            "台積電收盤價約在 880 元到 930 元之間，支撐 870 元附近。",
            source_name="moneydj",
        )
        report = make_report(evidence)
        block = build_forecast_block(query, report)
        self.assertEqual(block.mode, ForecastMode.HISTORICAL_PROXY)
        self.assertIsNotNone(block.scenario_range)
        # Should pick up 870-930 range (not some random number)
        self.assertGreaterEqual(block.scenario_range.low, 870.0)
        self.assertLessEqual(block.scenario_range.high, 930.0)

    def test_absurd_spread_rejected(self) -> None:
        """Spread > 5x (e.g., 22 and 2100) should be rejected."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "混合資料",
            "收盤價 22 元的標的與收盤價 2100 元的標的",
            source_name="TaiwanStockPrice",
        )
        report = make_report(evidence)
        block = build_forecast_block(query, report)
        # The 5x sanity check should reject this pair
        self.assertIsNone(block.scenario_range)

    def test_news_without_price_context_ignored(self) -> None:
        """News mentioning 營收 130 億元 should not be treated as price."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "台積電營收快報",
            "台積電月營收達 2300 億元，年增 35%，分析師預估全年營收 2 兆 7000 億元。",
            source_name="cnyes_news",
        )
        report = make_report(evidence)
        block = build_forecast_block(query, report)
        # Revenue numbers should not be used as proxy price range
        self.assertIsNone(block.scenario_range)


# ===========================================================================
# 7. SourceResponse regression: /api/sources must include sources list
# ===========================================================================

class SourceResponseRegressionTestCase(unittest.TestCase):
    """/api/sources/{query_id} must return the full sources list."""

    def test_source_response_includes_sources_field(self) -> None:
        """SourceResponse must serialize a 'sources' list in its JSON output."""
        from llm_stock_system.core.models import SourceResponse

        citation = SourceCitation(
            title="法人看好台積電",
            source_name="moneydj",
            source_tier=SourceTier.MEDIUM,
            url="https://example.com/article",
            published_at=datetime(2026, 4, 14),
            excerpt="外資目標價 950-1050 元",
            support_score=0.8,
            corroboration_count=1,
        )
        sr = SourceResponse(
            query_id="q-test-123",
            ticker="2330",
            topic=Topic.COMPOSITE,
            source_count=1,
            sources=[citation],
        )
        data = sr.model_dump()
        self.assertIn("sources", data)
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["title"], "法人看好台積電")

    def test_source_response_default_empty_list(self) -> None:
        """SourceResponse without explicit sources should default to []."""
        from llm_stock_system.core.models import SourceResponse

        sr = SourceResponse(
            query_id="q-test-456",
            ticker="2330",
            topic=Topic.COMPOSITE,
            source_count=0,
        )
        data = sr.model_dump()
        self.assertIn("sources", data)
        self.assertEqual(data["sources"], [])

    def test_query_log_store_saves_sources(self) -> None:
        """InMemoryQueryLogStore.save must persist sources into SourceResponse."""
        from llm_stock_system.adapters.repositories import InMemoryQueryLogStore
        from llm_stock_system.core.models import ValidationResult

        store = InMemoryQueryLogStore()
        evidence = make_evidence("法人看好", "目標價 950-1050 元")
        report = make_report(evidence)

        query = StructuredQuery(
            user_query="台積電目標價",
            ticker="2330",
            company_name="台積電",
            topic=Topic.COMPOSITE,
            question_type="price_outlook",
        )
        from llm_stock_system.core.enums import ConfidenceLight
        vr = ValidationResult(
            confidence_score=0.6,
            confidence_light=ConfidenceLight.YELLOW,
            validation_status="review",
        )
        citation = SourceCitation(
            title=evidence.title,
            source_name=evidence.source_name,
            source_tier=evidence.source_tier,
            url=evidence.url,
            published_at=evidence.published_at,
            excerpt=evidence.excerpt,
            support_score=evidence.support_score,
            corroboration_count=evidence.corroboration_count,
        )
        from llm_stock_system.core.models import DataStatus, QueryResponse
        response = QueryResponse(
            query_id="placeholder",
            summary="test",
            highlights=[],
            facts=[],
            impacts=[],
            risks=[],
            dataStatus=DataStatus(
                sufficiency=report.sufficiency,
                consistency=report.consistency,
                freshness=report.freshness,
            ),
            confidenceLight=vr.confidence_light,
            confidenceScore=vr.confidence_score,
            sources=[citation],
            disclaimer="test",
        )

        query_id = store.save(query, response, report, vr)
        source_response = store.get_sources(query_id)

        self.assertIsNotNone(source_response)
        self.assertEqual(len(source_response.sources), 1)
        self.assertEqual(source_response.sources[0].title, evidence.title)


# ===========================================================================
# 8. Number cross-check guardrail: prevent LLM-fabricated price numbers
# ===========================================================================

class FabricatedNumberGuardrailTestCase(unittest.TestCase):
    """LLM-generated numbers not grounded in ForecastBlock should be sanitized."""

    def _make_block_with_range(self, low: float, high: float) -> ForecastBlock:
        return ForecastBlock(
            mode=ForecastMode.SCENARIO_ESTIMATE,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.BULLISH_BIAS,
            scenario_range=ScenarioRange(low=low, high=high, basis_type="analyst_target"),
            forecast_basis=[
                f"分析師目標價區間 {low}–{high} 元",
                "此為情境推估，不是確定預測",
            ],
        )

    def _make_block_no_range(self) -> ForecastBlock:
        return ForecastBlock(
            mode=ForecastMode.UNSUPPORTED,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.UNDETERMINED,
            forecast_basis=["無可用的前瞻或價格依據"],
        )

    def test_fabricated_range_phrase_replaced(self) -> None:
        """A range phrase with numbers not in ForecastBlock gets replaced."""
        block = self._make_block_with_range(950.0, 1050.0)
        text = "預估區間 800~850 元，偏多操作。"
        result = apply_forecast_guardrail(text, block)
        self.assertNotIn("800", result)
        self.assertNotIn("850", result)
        self.assertIn("系統無法提供具體預估區間數值", result)

    def test_legitimate_range_phrase_preserved(self) -> None:
        """A range phrase matching ForecastBlock numbers is kept intact."""
        block = self._make_block_with_range(950.0, 1050.0)
        text = "預估區間 950~1050 元，情境推估偏多。"
        result = apply_forecast_guardrail(text, block)
        self.assertIn("950", result)
        self.assertIn("1050", result)

    def test_fabricated_prediction_number_replaced(self) -> None:
        """'預估 1200 元' with 1200 not in ForecastBlock is replaced."""
        block = self._make_block_with_range(950.0, 1050.0)
        text = "預估 1200 元的價位有機會挑戰。"
        result = apply_forecast_guardrail(text, block)
        self.assertNotIn("1200", result)
        self.assertIn("無可靠依據的具體數值", result)

    def test_legitimate_prediction_number_preserved(self) -> None:
        """'預估 950 元' matching ForecastBlock is kept."""
        block = self._make_block_with_range(950.0, 1050.0)
        text = "預估 950 元為下方支撐。"
        result = apply_forecast_guardrail(text, block)
        self.assertIn("950", result)

    def test_no_range_block_strips_all_predictions(self) -> None:
        """When ForecastBlock has no range, all prediction numbers are stripped."""
        block = self._make_block_no_range()
        text = "預估區間 900~950 元，可能上看 980 元。"
        result = apply_forecast_guardrail(text, block)
        self.assertNotIn("900", result)
        self.assertNotIn("950", result)
        self.assertNotIn("980", result)

    def test_tolerance_allows_close_numbers(self) -> None:
        """Numbers within 5% of ForecastBlock values should pass."""
        block = self._make_block_with_range(950.0, 1050.0)
        # 997.5 is within 5% of 1050 (1050 * 0.95 = 997.5)
        text = "預估 998 元附近有壓力。"
        result = apply_forecast_guardrail(text, block)
        self.assertIn("998", result)

    def test_guardrail_to_list_sanitizes_highlights(self) -> None:
        """apply_forecast_guardrail_to_list should sanitize fabricated numbers in highlights."""
        block = self._make_block_with_range(950.0, 1050.0)
        items = [
            "目前偏多格局",
            "預估 1200 元的目標有望達成",
            "分析師目標價 950 至 1050 元",
        ]
        result = apply_forecast_guardrail_to_list(items, block)
        self.assertEqual(len(result), 3)
        # Item 0: no numbers, unchanged
        self.assertIn("偏多格局", result[0])
        # Item 1: fabricated 1200 should be replaced
        self.assertNotIn("1200", result[1])
        self.assertIn("無可靠依據的具體數值", result[1])
        # Item 2: legitimate 950/1050 should be preserved
        self.assertIn("950", result[2])
        self.assertIn("1050", result[2])

    def test_guardrail_to_list_no_range_strips_predictions(self) -> None:
        """With unsupported block (no range), prediction-style numbers in list items are stripped."""
        block = self._make_block_no_range()
        items = [
            "上看 980 元",
            "預估 900 元為支撐",
        ]
        result = apply_forecast_guardrail_to_list(items, block)
        self.assertNotIn("980", result[0])
        self.assertNotIn("900", result[1])


# ===========================================================================
# 9. Presentation layer: forecast RED preserves original summary
# ===========================================================================

class ForecastRedPresentationTestCase(unittest.TestCase):
    """RED confidence with a ForecastBlock should NOT replace summary with generic fallback."""

    def test_historical_proxy_red_preserves_forecast_summary(self) -> None:
        from llm_stock_system.layers.presentation_layer import PresentationLayer
        from llm_stock_system.core.models import ValidationResult

        forecast = ForecastBlock(
            mode=ForecastMode.HISTORICAL_PROXY,
            forecast_window=ForecastWindow(label="未來一週", start_date="2026-04-16", end_date="2026-04-23"),
            direction=ForecastDirection.RANGE_BOUND,
            scenario_range=ScenarioRange(low=1760, high=2100, basis_type="historical_proxy"),
            forecast_basis=[
                "近期價格波動區間約 1760–2100 元（歷史波動代理，不是明確目標價）",
            ],
        )
        draft = AnswerDraft(
            summary="近期價格波動區間約 1760–2100 元（僅以歷史波動代理，非明確目標價）",
            highlights=["近期股價在 1760 至 2100 元之間波動"],
            facts=["收盤價資料來自 TaiwanStockPrice"],
            impacts=[],
            risks=["此為歷史波動代理，不代表未來走勢"],
            sources=[],
            forecast=forecast,
        )
        report = make_empty_report()
        validation_result = ValidationResult(
            confidence_score=0.40,
            confidence_light=ConfidenceLight.RED,
            validation_status="review",
        )

        layer = PresentationLayer()
        response = layer.present(None, draft, report, validation_result)

        # Summary should be the original forecast text, NOT "資料不足，無法確認。"
        self.assertNotEqual(response.summary, "資料不足，無法確認。")
        self.assertIn("1760", response.summary)
        self.assertIn("2100", response.summary)
        self.assertIn("歷史波動代理", response.summary)
        # Highlights/facts should also be preserved
        self.assertTrue(len(response.highlights) >= 1)
        self.assertIn("1760", response.highlights[0])

    def test_non_forecast_red_still_uses_fallback(self) -> None:
        """Non-forecast RED queries should still get the generic fallback."""
        from llm_stock_system.layers.presentation_layer import PresentationLayer
        from llm_stock_system.core.models import ValidationResult

        draft = AnswerDraft(
            summary="台積電近期表現不錯。",
            highlights=[],
            facts=[],
            impacts=[],
            risks=[],
            sources=[],
            forecast=None,
        )
        report = make_empty_report()
        validation_result = ValidationResult(
            confidence_score=0.20,
            confidence_light=ConfidenceLight.RED,
            validation_status="blocked",
        )

        layer = PresentationLayer()
        response = layer.present(None, draft, report, validation_result)

        # Should be replaced with generic fallback
        self.assertEqual(response.summary, "資料不足，無法確認。")


# ===========================================================================
# 10. Price source extraction: dates and day counts excluded
# ===========================================================================

class PriceSourceExtractionUnitTestCase(unittest.TestCase):
    """Strategy 1 should only extract numbers with explicit price units (元/塊)."""

    def test_date_and_day_count_excluded_price_units_kept(self) -> None:
        """Evidence with '2026-04-16', '30 天', '2100 元', '1760 元' → range 1760–2100."""
        query = make_forecast_query("這一個星期台積電預估區間是多少？")
        evidence = make_evidence(
            "台積電近期走勢 2026-04-16",
            "近 30 天台積電股價在 1760 元到 2100 元之間波動，2026-04-16 收盤 2085 元",
            source_name="TaiwanStockPrice",
        )
        report = make_report(evidence)

        block = build_forecast_block(query, report)
        self.assertEqual(block.mode, ForecastMode.HISTORICAL_PROXY)
        self.assertIsNotNone(block.scenario_range)
        self.assertEqual(block.scenario_range.low, 1760.0)
        self.assertEqual(block.scenario_range.high, 2100.0)

    def test_single_price_unit_number_no_range(self) -> None:
        """Only one number with 元 → not enough to form a range."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "台積電收盤",
            "2026-04-16 台積電收盤 1760 元，成交量 30000 張",
            source_name="TaiwanStockPrice",
        )
        report = make_report(evidence)

        block = build_forecast_block(query, report)
        # Only one price value (1760), can't form a range
        self.assertIsNone(block.scenario_range)

    def test_bare_numbers_without_unit_ignored(self) -> None:
        """Numbers without 元/塊 in price source should be ignored."""
        query = make_forecast_query("這週台積電預估區間？")
        evidence = make_evidence(
            "台積電交易明細",
            "日期 20260416 成交量 45000 張，外資買超 2300 張，融資餘額 15000 張",
            source_name="TaiwanStockPrice",
        )
        report = make_report(evidence)

        block = build_forecast_block(query, report)
        # No price-unit numbers → no range
        self.assertIsNone(block.scenario_range)


if __name__ == "__main__":
    unittest.main()
