from llm_stock_system.core.enums import ConfidenceLight, ConsistencyStatus, FreshnessStatus, StanceBias
from llm_stock_system.core.fundamental_valuation import has_fundamental_evidence, has_valuation_evidence
from llm_stock_system.core.models import AnswerDraft, GovernanceReport, StructuredQuery, ValidationResult
from llm_stock_system.core.target_price import (
    extract_target_price_values,
    has_forward_price_context,
    has_price_level_context,
    has_target_price_context,
    is_forward_price_question,
    is_target_price_question,
)


class ValidationLayer:
    def __init__(self, min_green_confidence: float, min_yellow_confidence: float) -> None:
        self._min_green_confidence = min_green_confidence
        self._min_yellow_confidence = min_yellow_confidence

    def validate(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        facet_miss_list: list[str] | None = None,
    ) -> ValidationResult:
        evidence_score = min(len(governance_report.evidence) / 4, 1.0) * 0.30
        trust_score = governance_report.high_trust_ratio * 0.25
        freshness_score = self._freshness_weight(governance_report.freshness) * 0.20
        consistency_score = self._consistency_weight(governance_report.consistency) * 0.15
        citation_score = min(len(answer_draft.sources) / max(len(governance_report.evidence), 1), 1.0) * 0.10

        confidence_score = round(
            evidence_score + trust_score + freshness_score + consistency_score + citation_score,
            2,
        )

        warnings: list[str] = []
        if query.stance_bias != StanceBias.NEUTRAL and len(answer_draft.risks) < 3:
            warnings.append("Biased question should include at least three risks.")
            confidence_score = max(confidence_score - 0.10, 0.0)
        if not governance_report.evidence:
            warnings.append("No supporting evidence retrieved.")
        if answer_draft.summary.startswith("初步判讀："):
            warnings.append("Preliminary LLM answer returned without grounded local evidence.")
            confidence_score = min(confidence_score, 0.35)
        if governance_report.consistency == ConsistencyStatus.CONFLICTING:
            warnings.append("Evidence consistency is low.")
        if any(token in answer_draft.summary for token in ("\u8cc7\u6599\u4e0d\u8db3", "\u7121\u6cd5\u78ba\u8a8d")):
            warnings.append("Answer indicates insufficient data.")
            confidence_score = min(confidence_score, 0.25)
        if query.question_type == "price_outlook" and is_forward_price_question(query):
            has_direct_numeric_evidence = False
            has_directional_context = False

            if is_target_price_question(query):
                has_direct_numeric_evidence = bool(extract_target_price_values(governance_report))
                has_directional_context = has_target_price_context(governance_report)
            else:
                has_direct_numeric_evidence = has_price_level_context(query, governance_report)
                has_directional_context = bool(extract_target_price_values(governance_report)) or has_forward_price_context(
                    query,
                    governance_report,
                )

            if has_direct_numeric_evidence:
                warnings.append("Forward price answers remain scenario-dependent even when a numeric level is mentioned.")
                confidence_score = min(confidence_score, 0.55)
            elif has_directional_context:
                warnings.append(
                    "Forward price query is supported only by directional analyst or technical context without direct numeric evidence."
                )
                confidence_score = min(confidence_score, 0.55)
            else:
                warnings.append("Forward price query lacks direct target-price, threshold, or analyst evidence.")
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "ex_dividend_performance":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            has_dividend_evidence = any("dividend" in source_name for source_name in source_names)
            has_price_evidence = any("price" in source_name for source_name in source_names)
            if not (has_dividend_evidence and has_price_evidence):
                warnings.append("Ex-dividend performance requires both price and dividend evidence.")
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "technical_indicator_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            if not any("price" in source_name for source_name in source_names):
                warnings.append("Technical indicator review requires price evidence.")
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "season_line_margin_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            has_price_evidence = any("price" in source_name for source_name in source_names)
            has_margin_evidence = any("marginpurchase" in source_name or "margin_purchase" in source_name for source_name in source_names)
            if not (has_price_evidence and has_margin_evidence):
                warnings.append("Season line and margin review requires both price and margin evidence.")
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "fcf_dividend_sustainability_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            has_cash_flow_evidence = any("cashflows" in source_name or "cash flow" in source_name for source_name in source_names)
            has_dividend_evidence = any("dividend" in source_name for source_name in source_names)
            if not (has_cash_flow_evidence and has_dividend_evidence):
                warnings.append("FCF dividend sustainability review requires both cash flow and dividend evidence.")
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "monthly_revenue_yoy_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            if not any("twse" in source_name for source_name in source_names):
                warnings.append("Monthly revenue YoY review requires official revenue evidence.")
                confidence_score = min(confidence_score, 0.25)
            if "僅更新到今年前" in answer_draft.summary:
                warnings.append("Monthly revenue data is only partially available for the requested period.")
                confidence_score = min(confidence_score, 0.75)
        if query.question_type == "shipping_rate_impact_review":
            combined_text = " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)
            has_shipping_signal = any(token in combined_text for token in ("紅海", "SCFI", "運價", "航運"))
            has_target_signal = any(token in combined_text for token in ("目標價", "評等", "分析師", "法人", "外資"))
            if not (has_shipping_signal and has_target_signal and len(governance_report.evidence) >= 2):
                warnings.append("Shipping rate impact review requires both freight-rate and analyst-target evidence.")
                confidence_score = min(confidence_score, 0.25)
            elif query.comparison_ticker:
                primary_label = query.company_name or query.ticker or ""
                comparison_label = query.comparison_company_name or query.comparison_ticker or ""
                if primary_label and comparison_label and (
                    primary_label not in combined_text or comparison_label not in combined_text
                ):
                    warnings.append("Shipping rate impact comparison is only partially grounded for one of the companies.")
                    confidence_score = min(confidence_score, 0.75)
        if query.question_type == "electricity_cost_impact_review":
            combined_text = " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)
            has_cost_signal = any(token in combined_text for token in ("工業電價", "電價", "電費", "成本", "用電大戶"))
            has_response_signal = any(token in combined_text for token in ("因應", "對策", "節能", "節電", "轉嫁", "調價", "綠電", "自發電"))
            if not (has_cost_signal and len(governance_report.evidence) >= 2):
                warnings.append("Electricity cost impact review requires grounded electricity-cost evidence.")
                confidence_score = min(confidence_score, 0.25)
            elif not has_response_signal:
                warnings.append("Electricity cost impact review is missing concrete response evidence.")
                confidence_score = min(confidence_score, 0.75)
            elif query.comparison_ticker:
                primary_label = query.company_name or query.ticker or ""
                comparison_label = query.comparison_company_name or query.comparison_ticker or ""
                if primary_label and comparison_label and (
                    primary_label not in combined_text or comparison_label not in combined_text
                ):
                    warnings.append("Electricity cost impact comparison is only partially grounded for one of the companies.")
                    confidence_score = min(confidence_score, 0.75)
        if query.question_type == "macro_yield_sentiment_review":
            combined_text = " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)
            has_macro_signal = any(token in combined_text for token in ("CPI", "通膨", "利率", "殖利率", "高殖利率"))
            has_view_signal = any(token in combined_text for token in ("法人", "外資", "觀點", "看法", "保守", "防禦"))
            if not (has_macro_signal and len(governance_report.evidence) >= 2):
                warnings.append("Macro yield sentiment review requires grounded macro and yield-sentiment evidence.")
                confidence_score = min(confidence_score, 0.25)
            elif not has_view_signal:
                warnings.append("Macro yield sentiment review is missing concrete institutional views.")
                confidence_score = min(confidence_score, 0.75)
        if query.question_type == "gross_margin_comparison_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            combined_text = " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)
            primary_label = query.company_name or query.ticker or ""
            comparison_label = query.comparison_company_name or query.comparison_ticker or ""
            has_financial_evidence = any("financialstatements" in source_name for source_name in source_names)
            has_primary = bool(primary_label) and primary_label in combined_text
            has_comparison = bool(comparison_label) and comparison_label in combined_text
            if not (has_financial_evidence and has_primary and has_comparison and len(governance_report.evidence) >= 3):
                warnings.append(
                    "Gross margin comparison review requires comparable financial-statement evidence for both companies."
                )
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "margin_turnaround_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            combined_text = " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)
            has_financial_evidence = any("financialstatements" in source_name for source_name in source_names)
            has_margin_phrase = "毛利率" in combined_text
            has_operating_phrase = "營業利益" in combined_text
            if not (has_financial_evidence and has_margin_phrase and has_operating_phrase and len(governance_report.evidence) >= 3):
                warnings.append(
                    "Margin turnaround review requires latest-quarter financial-statement evidence for both gross margin and operating income."
                )
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "profitability_stability_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            has_financial_evidence = any("financialstatements" in source_name for source_name in source_names)
            if not has_financial_evidence or len(governance_report.evidence) < 3:
                warnings.append("Profitability stability review requires multi-year financial statement evidence.")
                confidence_score = min(confidence_score, 0.25)
            if "財報結構推估" in answer_draft.summary and not any("news" in source_name for source_name in source_names):
                warnings.append("Loss-year reason is inferred from financial structure without direct news corroboration.")
                confidence_score = min(confidence_score, 0.75)
        if query.question_type == "debt_dividend_safety_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            has_balance_sheet_evidence = any("balancesheet" in source_name for source_name in source_names)
            has_dividend_evidence = any("dividend" in source_name for source_name in source_names)
            if not (has_balance_sheet_evidence and has_dividend_evidence):
                warnings.append("Debt dividend safety review requires both balance-sheet and dividend evidence.")
                confidence_score = min(confidence_score, 0.25)
        if query.question_type == "pe_valuation_review":
            source_names = {item.source_name.lower() for item in governance_report.evidence}
            if not any("twse" in source_name for source_name in source_names):
                warnings.append("PE valuation review requires official valuation evidence.")
                confidence_score = min(confidence_score, 0.25)
        if query.question_type in {"fundamental_pe_review", "investment_support"}:
            if not has_fundamental_evidence(governance_report):
                warnings.append("Combined fundamental and valuation review is missing direct fundamental evidence.")
                confidence_score = min(confidence_score, 0.25)
            if not has_valuation_evidence(governance_report):
                warnings.append("Combined fundamental and valuation review is missing direct valuation evidence.")
                confidence_score = min(confidence_score, 0.25)
            if has_fundamental_evidence(governance_report) and has_valuation_evidence(governance_report):
                confidence_score = min(confidence_score, 0.75)

        if confidence_score >= self._min_green_confidence:
            light = ConfidenceLight.GREEN
            status = "pass"
        elif confidence_score >= self._min_yellow_confidence:
            light = ConfidenceLight.YELLOW
            status = "review"
        else:
            light = ConfidenceLight.RED
            status = "blocked"

        return ValidationResult(
            confidence_score=confidence_score,
            confidence_light=light,
            validation_status=status,
            warnings=warnings,
            facet_miss_list=list(facet_miss_list or []),
        )

    def _freshness_weight(self, status: FreshnessStatus) -> float:
        return {
            FreshnessStatus.RECENT: 1.0,
            FreshnessStatus.STALE: 0.6,
            FreshnessStatus.OUTDATED: 0.2,
        }[status]

    def _consistency_weight(self, status: ConsistencyStatus) -> float:
        return {
            ConsistencyStatus.CONSISTENT: 1.0,
            ConsistencyStatus.MOSTLY_CONSISTENT: 0.75,
            ConsistencyStatus.CONFLICTING: 0.2,
        }[status]
nsistencyStatus.MOSTLY_CONSISTENT: 0.7,
            ConsistencyStatus.CONFLICTING: 0.3,
        }[status]
