from datetime import datetime, timezone
import unittest

from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, SourceCitation, StructuredQuery
from llm_stock_system.core.target_price import is_forward_price_question
from llm_stock_system.core.validation_profiles import get_profile
from llm_stock_system.layers.validation_layer import ValidationLayer


UTC_NOW = datetime(2026, 4, 16, tzinfo=timezone.utc)


def build_query(question_type: str, **overrides) -> StructuredQuery:
    defaults = {
        "user_query": f"validate {question_type}",
        "ticker": "2330",
        "company_name": "TSMC",
        "question_type": question_type,
        "time_range_label": "1y",
        "time_range_days": 365,
    }
    defaults.update(overrides)
    return StructuredQuery(**defaults)


def build_evidence(source_name: str, title: str, excerpt: str) -> Evidence:
    return Evidence(
        document_id=f"{source_name}-{title}",
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=SourceTier.HIGH,
        url="https://example.com/evidence",
        published_at=UTC_NOW,
        support_score=1.0,
        corroboration_count=1,
    )


def build_governance_report(evidence: list[Evidence]) -> GovernanceReport:
    return GovernanceReport(
        evidence=evidence,
        sufficiency=SufficiencyStatus.SUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT,
        freshness=FreshnessStatus.RECENT,
        high_trust_ratio=1.0,
    )


def build_answer_draft(summary: str, evidence: list[Evidence]) -> AnswerDraft:
    sources = [
        SourceCitation(
            title=item.title,
            source_name=item.source_name,
            source_tier=item.source_tier,
            url=item.url,
            published_at=item.published_at,
            excerpt=item.excerpt,
            support_score=item.support_score,
            corroboration_count=item.corroboration_count,
        )
        for item in evidence
    ]
    return AnswerDraft(
        summary=summary,
        highlights=[],
        facts=[],
        impacts=[],
        risks=["risk 1", "risk 2", "risk 3"],
        sources=sources,
    )


class ValidationProfileParityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    def test_profile_score_parity(self) -> None:
        cases = [
            ("ex_dividend_performance", "happy", 1.0),
            ("ex_dividend_performance", "degraded", 1.0),
            ("technical_indicator_review", "happy", 1.0),
            ("technical_indicator_review", "degraded", 1.0),
            ("monthly_revenue_yoy_review", "happy", 1.0),
            ("monthly_revenue_yoy_review", "degraded", 1.0),
            ("pe_valuation_review", "happy", 1.0),
            ("pe_valuation_review", "degraded", 1.0),
            ("profitability_stability_review", "happy", 1.0),
            ("profitability_stability_review", "degraded", 1.0),
            ("margin_turnaround_review", "happy", 1.0),
            ("margin_turnaround_review", "degraded", 1.0),
            ("fundamental_pe_review", "happy", 1.0),
            ("fundamental_pe_review", "degraded", 1.0),
            ("investment_support", "happy", 1.0),
            ("investment_support", "degraded", 1.0),
            ("season_line_margin_review", "happy", 1.0),
            ("season_line_margin_review", "one_missing", 0.5),
            ("season_line_margin_review", "both_missing", 0.25),
            ("fcf_dividend_sustainability_review", "happy", 1.0),
            ("fcf_dividend_sustainability_review", "one_missing", 0.5),
            ("fcf_dividend_sustainability_review", "both_missing", 0.25),
            ("debt_dividend_safety_review", "happy", 1.0),
            ("debt_dividend_safety_review", "one_missing", 0.5),
            ("debt_dividend_safety_review", "both_missing", 0.25),
            ("gross_margin_comparison_review", "happy", 1.0),
            ("gross_margin_comparison_review", "one_missing", 0.5),
            ("gross_margin_comparison_review", "both_missing", 0.25),
            ("price_outlook", "numeric_target", 0.55),
            ("price_outlook", "directional_only", 0.55),
            ("price_outlook", "no_context", 0.25),
        ]

        for question_type, scenario, expected_score in cases:
            with self.subTest(question_type=question_type, scenario=scenario):
                query, governance_report, answer_draft = self._build_case(question_type, scenario)
                profile = get_profile(query.intent)
                self.assertIsNotNone(profile)

                legacy_warnings: list[str] = []
                profile_warnings: list[str] = []

                legacy_score = self._run_legacy(
                    question_type,
                    query,
                    governance_report,
                    answer_draft,
                    1.0,
                    legacy_warnings,
                )
                profile_score = self.validation._evaluate_profile(
                    profile,
                    query,
                    governance_report,
                    answer_draft,
                    1.0,
                    profile_warnings,
                )

                self.assertEqual(legacy_score, expected_score)
                self.assertEqual(profile_score, expected_score)
                self.assertEqual(profile_score, legacy_score)

    def _build_case(
        self,
        question_type: str,
        scenario: str,
    ) -> tuple[StructuredQuery, GovernanceReport, AnswerDraft]:
        if question_type == "ex_dividend_performance":
            evidence = [
                build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot", "cash dividend 12.0"),
                build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price history remains stable"),
            ]
            if scenario == "degraded":
                evidence = [item for item in evidence if "Price" not in item.source_name]
            return self._case_from_evidence(question_type, evidence, "Ex-dividend review.")

        if question_type == "technical_indicator_review":
            evidence = [build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price trend remains constructive")]
            if scenario == "degraded":
                evidence = [build_evidence("Generic source", "Context", "market commentary")]
            return self._case_from_evidence(question_type, evidence, "Technical review.")

        if question_type == "monthly_revenue_yoy_review":
            evidence = [build_evidence("TWSE monthly revenue", "Revenue snapshot", "monthly revenue year over year improved")]
            summary = "Monthly revenue review."
            if scenario == "degraded":
                summary = f"Monthly revenue review. {self.validation._PARTIAL_MONTHLY_REVENUE_TOKENS[0]}"
            return self._case_from_evidence(question_type, evidence, summary)

        if question_type == "pe_valuation_review":
            evidence = [build_evidence("TWSE valuation", "Valuation snapshot", "valuation percentile 62% and PE ratio 18.2")]
            if scenario == "degraded":
                evidence = [build_evidence("Generic source", "Context", "valuation discussion")]
            return self._case_from_evidence(question_type, evidence, "PE valuation review.")

        if question_type == "profitability_stability_review":
            evidence = [
                build_evidence("FinMind TaiwanStockFinancialStatements", "Financial snapshot 1", "EPS 10.0"),
                build_evidence("FinMind TaiwanStockFinancialStatements", "Financial snapshot 2", "EPS 11.0"),
                build_evidence("FinMind TaiwanStockFinancialStatements", "Financial snapshot 3", "EPS 12.0"),
            ]
            if scenario == "degraded":
                evidence = evidence[:2]
            return self._case_from_evidence(question_type, evidence, "Profitability stability review.")

        if question_type == "margin_turnaround_review":
            evidence = [
                build_evidence(
                    "FinMind TaiwanStockFinancialStatements",
                    "Financial snapshot 1",
                    "gross margin improved and operating income recovered",
                ),
                build_evidence(
                    "FinMind TaiwanStockFinancialStatements",
                    "Financial snapshot 2",
                    "gross margin stabilized and operating margin expanded",
                ),
                build_evidence(
                    "FinMind TaiwanStockFinancialStatements",
                    "Financial snapshot 3",
                    "operating income remained positive with stable gross margin",
                ),
            ]
            if scenario == "degraded":
                evidence = evidence[:2]
            return self._case_from_evidence(question_type, evidence, "Margin turnaround review.")

        if question_type in {"fundamental_pe_review", "investment_support"}:
            evidence = [
                build_evidence("FinMind TaiwanStockFinancialStatements", "EPS snapshot", "latest EPS 32.1 and gross margin stable"),
                build_evidence("TWSE valuation", "Valuation snapshot", "valuation percentile 62% and PE ratio 18.2"),
            ]
            if scenario == "degraded":
                evidence = [build_evidence("TWSE valuation", "Valuation snapshot", "valuation percentile 62% and PE ratio 18.2")]
            return self._case_from_evidence(question_type, evidence, "Fundamental valuation review.")

        if question_type == "season_line_margin_review":
            if scenario == "happy":
                evidence = [
                    build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price remains near season line"),
                    build_evidence("FinMind TaiwanStockMarginPurchaseShortSale", "Margin snapshot", "margin balance elevated"),
                ]
            elif scenario == "one_missing":
                evidence = [build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price remains near season line")]
            else:
                evidence = [build_evidence("Generic source", "Context", "market commentary")]
            return self._case_from_evidence(question_type, evidence, "Season line margin review.")

        if question_type == "fcf_dividend_sustainability_review":
            if scenario == "happy":
                evidence = [
                    build_evidence("FinMind TaiwanStockCashFlowsStatement", "Cash flow snapshot", "operating cash flow remains positive"),
                    build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot", "cash dividend 12.0"),
                ]
            elif scenario == "one_missing":
                evidence = [build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot", "cash dividend 12.0")]
            else:
                evidence = [build_evidence("Generic source", "Context", "market commentary")]
            return self._case_from_evidence(question_type, evidence, "FCF dividend sustainability review.")

        if question_type == "debt_dividend_safety_review":
            if scenario == "happy":
                evidence = [
                    build_evidence("FinMind TaiwanStockBalanceSheet", "Balance sheet snapshot", "debt ratio remained stable"),
                    build_evidence("FinMind TaiwanStockDividend", "Dividend snapshot", "cash dividend 12.0"),
                ]
            elif scenario == "one_missing":
                evidence = [build_evidence("FinMind TaiwanStockBalanceSheet", "Balance sheet snapshot", "debt ratio remained stable")]
            else:
                evidence = [build_evidence("Generic source", "Context", "market commentary")]
            return self._case_from_evidence(question_type, evidence, "Debt dividend safety review.")

        if question_type == "gross_margin_comparison_review":
            query = build_query(
                question_type,
                comparison_ticker="2303",
                comparison_company_name="UMC",
            )
            if scenario == "happy":
                evidence = [
                    build_evidence("FinMind TaiwanStockFinancialStatements", "Comparison 1", "TSMC gross margin 52%, UMC gross margin 34%"),
                    build_evidence("FinMind TaiwanStockFinancialStatements", "Comparison 2", "TSMC profitability stayed ahead of UMC"),
                    build_evidence("FinMind TaiwanStockFinancialStatements", "Comparison 3", "Both TSMC and UMC remained profitable"),
                ]
            elif scenario == "one_missing":
                evidence = [
                    build_evidence("FinMind TaiwanStockFinancialStatements", "Comparison 1", "TSMC gross margin 52%"),
                    build_evidence("FinMind TaiwanStockFinancialStatements", "Comparison 2", "TSMC profitability stayed stable"),
                ]
            else:
                evidence = [
                    build_evidence("Generic source", "Comparison 1", "TSMC gross margin 52%, UMC gross margin 34%"),
                    build_evidence("Generic source", "Comparison 2", "Both TSMC and UMC remained profitable"),
                    build_evidence("Generic source", "Comparison 3", "Comparison overview"),
                ]
            return query, build_governance_report(evidence), build_answer_draft("Gross margin comparison review.", evidence)

        if question_type == "price_outlook":
            if scenario == "numeric_target":
                query = build_query("price_outlook", user_query="TSMC 未來一年目標價是多少？")
                evidence = [
                    build_evidence("Broker research", "Analyst update", "法人目標價上看 1280 元"),
                    build_evidence("Broker research", "Valuation note", "target price 1280 based on demand recovery"),
                    build_evidence("TWSE valuation", "Valuation snapshot", "valuation percentile 62% and PE ratio 18.2"),
                    build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price history remains stable"),
                ]
                return query, build_governance_report(evidence), build_answer_draft("Forward-looking price view.", evidence)
            if scenario == "directional_only":
                query = build_query("price_outlook", user_query="TSMC 未來股價會不會上漲？")
                evidence = [
                    build_evidence("Broker research", "Analyst update", "法人調高評等，技術面支撐轉強"),
                    build_evidence("Market note", "Trading note", "外資看法偏多，股價站上季線"),
                    build_evidence("FinMind TaiwanStockPrice", "Price snapshot", "price history remains stable"),
                ]
                return query, build_governance_report(evidence), build_answer_draft("Directional price view.", evidence)
            query = build_query("price_outlook", user_query="TSMC 未來股價會不會上漲？")
            evidence = [
                build_evidence("Generic source", "Context 1", "market commentary"),
                build_evidence("Generic source", "Context 2", "industry update"),
            ]
            return query, build_governance_report(evidence), build_answer_draft("Directional price view.", evidence)

        raise ValueError(f"Unsupported parity case: {question_type}/{scenario}")

    def _case_from_evidence(
        self,
        question_type: str,
        evidence: list[Evidence],
        summary: str,
    ) -> tuple[StructuredQuery, GovernanceReport, AnswerDraft]:
        query = build_query(question_type)
        return query, build_governance_report(evidence), build_answer_draft(summary, evidence)

    def _run_legacy(
        self,
        question_type: str,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        source_names = self.validation._source_names(governance_report)
        combined_text = self.validation._combined_text(governance_report)
        normalized_text = combined_text.lower()

        if question_type == "price_outlook":
            if is_forward_price_question(query):
                return self.validation._apply_price_outlook_rules(
                    query,
                    governance_report,
                    confidence_score,
                    warnings,
                )
            return self.validation._apply_directional_price_rules(
                query,
                governance_report,
                confidence_score,
                warnings,
            )
        if question_type == "ex_dividend_performance":
            return self.validation._apply_ex_dividend_rules(source_names, confidence_score, warnings)
        if question_type == "technical_indicator_review":
            return self.validation._apply_technical_indicator_rules(source_names, confidence_score, warnings)
        if question_type == "monthly_revenue_yoy_review":
            return self.validation._apply_monthly_revenue_rules(
                source_names,
                answer_draft,
                confidence_score,
                warnings,
            )
        if question_type == "pe_valuation_review":
            return self.validation._apply_pe_valuation_rules(source_names, confidence_score, warnings)
        if question_type == "profitability_stability_review":
            return self.validation._apply_profitability_stability_rules(
                governance_report,
                source_names,
                combined_text,
                confidence_score,
                warnings,
            )
        if question_type == "margin_turnaround_review":
            return self.validation._apply_margin_turnaround_rules(
                governance_report,
                source_names,
                combined_text,
                confidence_score,
                warnings,
            )
        if question_type in {"fundamental_pe_review", "investment_support"}:
            return self.validation._apply_fundamental_valuation_rules(
                governance_report,
                confidence_score,
                warnings,
            )
        if question_type == "season_line_margin_review":
            return self.validation._apply_season_line_margin_rules(source_names, confidence_score, warnings)
        if question_type == "fcf_dividend_sustainability_review":
            return self.validation._apply_fcf_dividend_rules(source_names, confidence_score, warnings)
        if question_type == "debt_dividend_safety_review":
            return self.validation._apply_debt_dividend_rules(source_names, confidence_score, warnings)
        if question_type == "gross_margin_comparison_review":
            return self.validation._apply_gross_margin_comparison_rules(
                query,
                governance_report,
                source_names,
                normalized_text,
                confidence_score,
                warnings,
            )
        raise ValueError(f"Unsupported legacy parity branch: {question_type}")


if __name__ == "__main__":
    unittest.main()
