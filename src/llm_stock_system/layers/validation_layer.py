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
from llm_stock_system.core.validation_profiles import ConditionKind, ValidationProfile, ValidationRule


class ValidationLayer:
    _PRELIMINARY_PREFIXES = ("初步判讀：", "preliminary")
    _INSUFFICIENT_DATA_TOKENS = ("資料不足", "無法確認")
    _PARTIAL_MONTHLY_REVENUE_TOKENS = ("部分月份", "僅能確認", "資料僅到", "尚未公告")

    def __init__(self, min_green_confidence: float, min_yellow_confidence: float) -> None:
        self._min_green_confidence = min_green_confidence
        self._min_yellow_confidence = min_yellow_confidence

    def validate(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        facet_miss_list: list[str] | None = None,
        preferred_miss_list: list[str] | None = None,
    ) -> ValidationResult:
        confidence_score = self._calculate_base_confidence(governance_report, answer_draft)
        warnings: list[str] = []

        confidence_score = self._apply_general_checks(
            query,
            governance_report,
            answer_draft,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_required_facet_cap(
            query,
            facet_miss_list,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_preferred_facet_penalty(
            query,
            preferred_miss_list,
            confidence_score,
            warnings,
        )
        confidence_score = self._apply_question_type_rules(
            query,
            governance_report,
            answer_draft,
            confidence_score,
            warnings,
        )
        confidence_score = round(max(confidence_score, 0.0), 2)

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

    def _calculate_base_confidence(
        self,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
    ) -> float:
        evidence_score = min(len(governance_report.evidence) / 4, 1.0) * 0.30
        trust_score = governance_report.high_trust_ratio * 0.25
        freshness_score = self._freshness_weight(governance_report.freshness) * 0.20
        consistency_score = self._consistency_weight(governance_report.consistency) * 0.15
        citation_score = min(len(answer_draft.sources) / max(len(governance_report.evidence), 1), 1.0) * 0.10
        return round(
            evidence_score + trust_score + freshness_score + consistency_score + citation_score,
            2,
        )

    def _apply_general_checks(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        summary = answer_draft.summary.strip()

        if query.stance_bias != StanceBias.NEUTRAL and len(answer_draft.risks) < 3:
            warnings.append("Biased question should include at least three risks.")
            confidence_score = max(confidence_score - 0.10, 0.0)
        if not governance_report.evidence:
            warnings.append("No supporting evidence retrieved.")
        if self._is_preliminary_summary(summary):
            warnings.append("Preliminary LLM answer returned without grounded local evidence.")
            confidence_score = min(confidence_score, 0.35)
        if governance_report.consistency == ConsistencyStatus.CONFLICTING:
            warnings.append("Evidence consistency is low.")
        if any(token in summary for token in self._INSUFFICIENT_DATA_TOKENS):
            warnings.append("Answer indicates insufficient data.")
            confidence_score = min(confidence_score, 0.25)

        return confidence_score

    def _apply_required_facet_cap(
        self,
        query: StructuredQuery,
        facet_miss_list: list[str] | None,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        required_facets = {facet.value for facet in query.required_facets}
        required_misses = sorted(set(facet_miss_list or []) & required_facets)

        if not required_facets or not required_misses:
            return confidence_score

        if len(required_misses) == len(required_facets):
            warnings.append(f"All required facets failed to sync: {required_misses}")
            return min(confidence_score, 0.25)

        warnings.append(f"Required facet sync failed (partial): {required_misses}")
        return min(confidence_score, 0.50)

    def _apply_preferred_facet_penalty(
        self,
        query: StructuredQuery,
        preferred_miss_list: list[str] | None,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        preferred_facets = {facet.value for facet in query.preferred_facets}
        preferred_misses = sorted(set(preferred_miss_list or []) & preferred_facets)

        if not preferred_misses:
            return confidence_score

        penalty = min(len(preferred_misses) * 0.10, 0.30)
        warnings.append(f"Preferred facets not synced ({len(preferred_misses)}): {preferred_misses}")
        return max(confidence_score - penalty, 0.0)

    def _evaluate_profile(
        self,
        profile: ValidationProfile,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        """Evaluate a declarative validation profile without changing call flow.

        PR1 keeps the legacy ``_apply_question_type_rules()`` path in production.
        This evaluator exists so we can add direct tests and dual-run parity checks
        before switching runtime behavior in a later PR.
        """
        source_names = self._source_names(governance_report)
        normalized_text = self._combined_text(governance_report).lower()

        for rule in profile.rules:
            confidence_score = self._apply_profile_rule(
                rule,
                query,
                governance_report,
                answer_draft,
                source_names,
                normalized_text,
                confidence_score,
                warnings,
            )

        if profile.custom_validator is not None:
            confidence_score = profile.custom_validator(
                query,
                governance_report,
                answer_draft,
                confidence_score,
                warnings,
            )

        return confidence_score

    def _apply_profile_rule(
        self,
        rule: ValidationRule,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        source_names: set[str],
        normalized_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        if rule.condition == ConditionKind.DUAL_SIGNAL_MISSING:
            return self._apply_dual_signal_rule(rule, source_names, confidence_score, warnings)

        if not self._is_profile_rule_triggered(
            rule,
            query,
            governance_report,
            answer_draft,
            source_names,
            normalized_text,
        ):
            return confidence_score

        if rule.warning:
            warnings.append(rule.warning)
        if rule.cap is not None:
            confidence_score = min(confidence_score, rule.cap)
        if rule.penalty is not None:
            confidence_score = max(confidence_score - rule.penalty, 0.0)
        return confidence_score

    def _is_profile_rule_triggered(
        self,
        rule: ValidationRule,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        source_names: set[str],
        normalized_text: str,
    ) -> bool:
        if rule.condition == ConditionKind.SOURCE_FRAGMENT_MISSING:
            fragments = tuple(rule.params.get("fragments", ()))
            return not self._has_source_fragment(source_names, *fragments)

        if rule.condition == ConditionKind.CONTENT_KEYWORD_MISSING:
            keywords = [str(keyword).lower() for keyword in rule.params.get("keywords", ())]
            match_mode = str(rule.params.get("match_mode", "any")).lower()
            if match_mode == "all":
                return any(keyword not in normalized_text for keyword in keywords)
            return not any(keyword in normalized_text for keyword in keywords)

        if rule.condition == ConditionKind.EVIDENCE_COUNT_BELOW:
            threshold = int(rule.params.get("threshold", 0))
            return len(governance_report.evidence) < threshold

        if rule.condition == ConditionKind.COMPARISON_COMPANY_MISSING:
            if not query.comparison_ticker:
                return False
            primary_label = (query.company_name or query.ticker or "").lower()
            comparison_label = (query.comparison_company_name or query.comparison_ticker or "").lower()
            if not (primary_label and comparison_label):
                return False
            return primary_label not in normalized_text or comparison_label not in normalized_text

        if rule.condition == ConditionKind.ANSWER_CONTAINS_TOKEN:
            summary = answer_draft.summary.lower()
            tokens = [str(token).lower() for token in rule.params.get("tokens", ())]
            return any(token in summary for token in tokens)

        raise ValueError(f"Unsupported condition kind: {rule.condition}")

    def _apply_dual_signal_rule(
        self,
        rule: ValidationRule,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        signal_a_fragments = tuple(rule.params.get("signal_a_fragments", ()))
        signal_b_fragments = tuple(rule.params.get("signal_b_fragments", ()))
        has_signal_a = self._has_source_fragment(source_names, *signal_a_fragments)
        has_signal_b = self._has_source_fragment(source_names, *signal_b_fragments)

        if has_signal_a and has_signal_b:
            return confidence_score

        if not has_signal_a and not has_signal_b:
            warning = str(rule.params.get("both_missing_warning", "")).strip()
            cap = rule.params.get("both_missing_cap")
        else:
            warning = str(rule.params.get("one_missing_warning", "")).strip()
            cap = rule.params.get("one_missing_cap")

        if warning:
            warnings.append(warning)
        if cap is not None:
            confidence_score = min(confidence_score, float(cap))
        return confidence_score

    def _apply_question_type_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        source_names = self._source_names(governance_report)
        combined_text = self._combined_text(governance_report)
        normalized_text = combined_text.lower()

        if query.question_type == "price_outlook":
            if is_forward_price_question(query):
                return self._apply_price_outlook_rules(query, governance_report, confidence_score, warnings)
            return self._apply_directional_price_rules(query, governance_report, confidence_score, warnings)
        if query.question_type == "ex_dividend_performance":
            return self._apply_ex_dividend_rules(source_names, confidence_score, warnings)
        if query.question_type == "technical_indicator_review":
            return self._apply_technical_indicator_rules(source_names, confidence_score, warnings)
        if query.question_type == "season_line_margin_review":
            return self._apply_season_line_margin_rules(source_names, confidence_score, warnings)
        if query.question_type == "fcf_dividend_sustainability_review":
            return self._apply_fcf_dividend_rules(source_names, confidence_score, warnings)
        if query.question_type == "monthly_revenue_yoy_review":
            return self._apply_monthly_revenue_rules(
                source_names,
                answer_draft,
                confidence_score,
                warnings,
            )
        if query.question_type == "shipping_rate_impact_review":
            return self._apply_shipping_rate_rules(query, governance_report, normalized_text, confidence_score, warnings)
        if query.question_type == "electricity_cost_impact_review":
            return self._apply_electricity_cost_rules(query, governance_report, normalized_text, confidence_score, warnings)
        if query.question_type == "macro_yield_sentiment_review":
            return self._apply_macro_yield_rules(query, governance_report, normalized_text, confidence_score, warnings)
        if query.question_type == "gross_margin_comparison_review":
            return self._apply_gross_margin_comparison_rules(
                query,
                governance_report,
                source_names,
                normalized_text,
                confidence_score,
                warnings,
            )
        if query.question_type == "margin_turnaround_review":
            return self._apply_margin_turnaround_rules(
                governance_report,
                source_names,
                combined_text,
                confidence_score,
                warnings,
            )
        if query.question_type == "profitability_stability_review":
            return self._apply_profitability_stability_rules(
                governance_report,
                source_names,
                combined_text,
                confidence_score,
                warnings,
            )
        if query.question_type == "debt_dividend_safety_review":
            return self._apply_debt_dividend_rules(source_names, confidence_score, warnings)
        if query.question_type == "pe_valuation_review":
            return self._apply_pe_valuation_rules(source_names, confidence_score, warnings)
        if query.question_type in {"fundamental_pe_review", "investment_support"}:
            return self._apply_fundamental_valuation_rules(governance_report, confidence_score, warnings)

        return confidence_score

    def _apply_directional_price_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        """Cap for price_outlook queries that lack a numeric target or price-level action.

        These are directional questions such as "還會繼續漲嗎" or "續漲空間多少".
        They cannot be confirmed from public data alone, so confidence is capped
        conservatively even when supporting context exists.
        """
        if has_forward_price_context(query, governance_report):
            warnings.append(
                "Directional price query has related market context but cannot confirm"
                " specific future price movement from public data."
            )
            return min(confidence_score, 0.55)
        warnings.append(
            "Directional price query lacks analyst, technical, or price-level evidence"
            " to support a directional view."
        )
        return min(confidence_score, 0.25)

    def _apply_price_outlook_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
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
            return min(confidence_score, 0.55)
        if has_directional_context:
            warnings.append(
                "Forward price query is supported only by directional analyst or technical context without direct numeric evidence."
            )
            return min(confidence_score, 0.55)

        warnings.append("Forward price query lacks direct target-price, threshold, or analyst evidence.")
        return min(confidence_score, 0.25)

    def _apply_ex_dividend_rules(
        self,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_dividend_evidence = self._has_source_fragment(source_names, "dividend")
        has_price_evidence = self._has_source_fragment(source_names, "price")
        if not (has_dividend_evidence and has_price_evidence):
            warnings.append(
                "Ex-dividend performance: missing price or dividend evidence (facet-based cap already applied if sync failed)."
            )
        return confidence_score

    def _apply_technical_indicator_rules(
        self,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        if not self._has_source_fragment(source_names, "price"):
            warnings.append(
                "Technical indicator review: no price evidence in governance report (facet-based cap already applied if sync failed)."
            )
        return confidence_score

    def _apply_monthly_revenue_rules(
        self,
        source_names: set[str],
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        if not self._has_source_fragment(source_names, "twse"):
            warnings.append(
                "Monthly revenue YoY review: no TWSE revenue evidence in governance report (facet-based cap already applied if sync failed)."
            )
        if any(token in answer_draft.summary for token in self._PARTIAL_MONTHLY_REVENUE_TOKENS):
            warnings.append("Monthly revenue data is only partially available for the requested period.")
        return confidence_score

    def _apply_pe_valuation_rules(
        self,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        if not self._has_source_fragment(source_names, "twse"):
            warnings.append(
                "PE valuation review: no official valuation evidence in governance report (facet-based cap already applied if sync failed)."
            )
        return confidence_score

    def _apply_profitability_stability_rules(
        self,
        governance_report: GovernanceReport,
        source_names: set[str],
        combined_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_financial_evidence = self._has_source_fragment(
            source_names,
            "financialstatements",
            "financial_statements",
        )
        if not has_financial_evidence or len(governance_report.evidence) < 3:
            warnings.append(
                "Profitability stability review: missing multi-year financial statement evidence (facet-based cap already applied if sync failed)."
            )
        if "若只看財報結構推估" in combined_text and not self._has_source_fragment(source_names, "news"):
            warnings.append("Loss-year reason is inferred from financial structure without direct news corroboration.")
            confidence_score = min(confidence_score, 0.75)
        return confidence_score

    def _apply_margin_turnaround_rules(
        self,
        governance_report: GovernanceReport,
        source_names: set[str],
        combined_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_financial_evidence = self._has_source_fragment(
            source_names,
            "financialstatements",
            "financial_statements",
        )
        has_margin_phrase = any(token in combined_text for token in ("毛利", "毛利率", "gross margin"))
        has_operating_phrase = any(
            token in combined_text for token in ("營業利益", "營益率", "operating income", "operating margin")
        )
        if not (has_financial_evidence and has_margin_phrase and has_operating_phrase and len(governance_report.evidence) >= 3):
            warnings.append(
                "Margin turnaround review: incomplete gross-margin or operating-income evidence (facet-based cap already applied if sync failed)."
            )
        return confidence_score

    def _apply_fundamental_valuation_rules(
        self,
        governance_report: GovernanceReport,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        if not has_fundamental_evidence(governance_report):
            warnings.append(
                "Combined fundamental and valuation review is missing direct fundamental evidence (facet-based cap already applied if sync failed)."
            )
        if not has_valuation_evidence(governance_report):
            warnings.append(
                "Combined fundamental and valuation review is missing direct valuation evidence (facet-based cap already applied if sync failed)."
            )
        return confidence_score

    def _apply_season_line_margin_rules(
        self,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_price_evidence = self._has_source_fragment(source_names, "price")
        has_margin_evidence = self._has_source_fragment(source_names, "marginpurchase", "margin_purchase")
        if not has_price_evidence and not has_margin_evidence:
            warnings.append("Season line and margin review: missing both price and margin evidence.")
            return min(confidence_score, 0.25)
        if not has_price_evidence or not has_margin_evidence:
            warnings.append("Season line and margin review: missing one of price or margin evidence.")
            return min(confidence_score, 0.50)
        return confidence_score

    def _apply_fcf_dividend_rules(
        self,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_cash_flow_evidence = self._has_source_fragment(source_names, "cashflows", "cash_flow")
        has_dividend_evidence = self._has_source_fragment(source_names, "dividend")
        if not has_cash_flow_evidence and not has_dividend_evidence:
            warnings.append("FCF dividend sustainability review: missing both cash-flow and dividend evidence.")
            return min(confidence_score, 0.25)
        if not has_cash_flow_evidence or not has_dividend_evidence:
            warnings.append("FCF dividend sustainability review: missing one of cash flow or dividend evidence.")
            return min(confidence_score, 0.50)
        return confidence_score

    def _apply_shipping_rate_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        normalized_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_shipping_signal = any(token in normalized_text for token in ("紅海", "scfi", "運價", "航運"))
        has_target_signal = any(token in normalized_text for token in ("目標價", "評等", "分析師", "法人", "外資"))
        if not (has_shipping_signal and has_target_signal and len(governance_report.evidence) >= 2):
            warnings.append("Shipping rate impact review requires both freight-rate and analyst-target evidence.")
            return min(confidence_score, 0.25)
        if query.comparison_ticker:
            primary_label = (query.company_name or query.ticker or "").lower()
            comparison_label = (query.comparison_company_name or query.comparison_ticker or "").lower()
            if primary_label and comparison_label and (
                primary_label not in normalized_text or comparison_label not in normalized_text
            ):
                warnings.append("Shipping rate impact comparison is only partially grounded for one of the companies.")
                return min(confidence_score, 0.75)
        return confidence_score

    def _apply_electricity_cost_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        normalized_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_cost_signal = any(token in normalized_text for token in ("工業電價", "電價", "電費", "成本", "用電大戶"))
        has_response_signal = any(
            token in normalized_text for token in ("因應", "對策", "節能", "節電", "轉嫁", "調價", "綠電", "自發電")
        )
        if not (has_cost_signal and len(governance_report.evidence) >= 2):
            warnings.append("Electricity cost impact review requires grounded electricity-cost evidence.")
            return min(confidence_score, 0.25)
        if not has_response_signal:
            warnings.append("Electricity cost impact review is missing concrete response evidence.")
            return min(confidence_score, 0.75)
        if query.comparison_ticker:
            primary_label = (query.company_name or query.ticker or "").lower()
            comparison_label = (query.comparison_company_name or query.comparison_ticker or "").lower()
            if primary_label and comparison_label and (
                primary_label not in normalized_text or comparison_label not in normalized_text
            ):
                warnings.append("Electricity cost impact comparison is only partially grounded for one of the companies.")
                return min(confidence_score, 0.75)
        return confidence_score

    def _apply_macro_yield_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        normalized_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_macro_signal = any(token in normalized_text for token in ("cpi", "通膨", "利率", "殖利率", "高殖利率"))
        has_view_signal = any(token in normalized_text for token in ("法人", "外資", "觀點", "看法", "保守", "防禦"))
        if not (has_macro_signal and len(governance_report.evidence) >= 2):
            warnings.append("Macro yield sentiment review requires grounded macro and yield-sentiment evidence.")
            return min(confidence_score, 0.25)
        if not has_view_signal:
            warnings.append("Macro yield sentiment review is missing concrete institutional views.")
            return min(confidence_score, 0.75)
        return confidence_score

    def _apply_gross_margin_comparison_rules(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        source_names: set[str],
        normalized_text: str,
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        primary_label = (query.company_name or query.ticker or "").lower()
        comparison_label = (query.comparison_company_name or query.comparison_ticker or "").lower()
        has_financial_evidence = self._has_source_fragment(source_names, "financialstatements", "financial_statements")
        has_primary = bool(primary_label) and primary_label in normalized_text
        has_comparison = bool(comparison_label) and comparison_label in normalized_text
        if not has_financial_evidence:
            warnings.append(
                "Gross margin comparison review: missing financial-statement evidence (facet-based cap already applied if sync failed)."
            )
            return min(confidence_score, 0.25)
        if not (has_primary and has_comparison) or len(governance_report.evidence) < 3:
            warnings.append(
                "Gross margin comparison review: evidence only partially covers both companies."
            )
            return min(confidence_score, 0.50)
        return confidence_score

    def _apply_debt_dividend_rules(
        self,
        source_names: set[str],
        confidence_score: float,
        warnings: list[str],
    ) -> float:
        has_balance_sheet_evidence = self._has_source_fragment(source_names, "balancesheet", "balance_sheet")
        has_dividend_evidence = self._has_source_fragment(source_names, "dividend")
        if not has_balance_sheet_evidence and not has_dividend_evidence:
            warnings.append("Debt dividend safety review: missing both balance-sheet and dividend evidence.")
            return min(confidence_score, 0.25)
        if not has_balance_sheet_evidence or not has_dividend_evidence:
            warnings.append("Debt dividend safety review: missing one of balance-sheet or dividend evidence.")
            return min(confidence_score, 0.50)
        return confidence_score

    # --- Helper methods ---

    def _source_names(self, governance_report: GovernanceReport) -> set[str]:
        return {item.source_name.lower() for item in governance_report.evidence}

    def _combined_text(self, governance_report: GovernanceReport) -> str:
        return " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)

    def _has_source_fragment(self, source_names: set[str], *fragments: str) -> bool:
        return any(fragment in source_name for fragment in fragments for source_name in source_names)

    def _is_preliminary_summary(self, summary: str) -> bool:
        return any(summary.startswith(prefix) for prefix in self._PRELIMINARY_PREFIXES)

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
