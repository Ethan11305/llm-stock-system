"""Declarative validation profiles keyed by intent and topic-tag discriminators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from llm_stock_system.core.enums import Intent

if TYPE_CHECKING:
    from llm_stock_system.core.models import AnswerDraft, GovernanceReport, StructuredQuery


class ConditionKind(str, Enum):
    """Condition types supported by the rule engine."""

    SOURCE_FRAGMENT_MISSING = "source_fragment_missing"
    CONTENT_KEYWORD_MISSING = "content_keyword_missing"
    EVIDENCE_COUNT_BELOW = "evidence_count_below"
    DUAL_SIGNAL_MISSING = "dual_signal_missing"
    COMPARISON_COMPANY_MISSING = "comparison_company_missing"
    ANSWER_CONTAINS_TOKEN = "answer_contains_token"


@dataclass(frozen=True)
class ValidationRule:
    """A single declarative validation check."""

    condition: ConditionKind
    params: dict[str, Any] = field(default_factory=dict)
    cap: float | None = None
    penalty: float | None = None
    warning: str = ""


class CustomValidatorFn(Protocol):
    """Typed callback for logic that cannot be expressed declaratively."""

    def __call__(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        answer_draft: AnswerDraft,
        confidence_score: float,
        warnings: list[str],
    ) -> float: ...


@dataclass(frozen=True)
class TagConditionRule:
    """Run this rule group only when all required tags are present."""

    required_tags: frozenset[str]
    rules: tuple[ValidationRule, ...] = ()
    custom_validator: CustomValidatorFn | None = None


@dataclass(frozen=True)
class ValidationProfile:
    """Intent-level profile with optional topic-tag discriminators."""

    intent: Intent
    base_rules: tuple[ValidationRule, ...] = ()
    tag_rules: tuple[TagConditionRule, ...] = ()
    custom_validator: CustomValidatorFn | None = None


def _source_names(report: GovernanceReport) -> set[str]:
    return {item.source_name.lower() for item in report.evidence}


def _has_source_fragment(source_names: set[str], *fragments: str) -> bool:
    return any(
        fragment in source_name
        for fragment in fragments
        for source_name in source_names
    )


def _combined_text(report: GovernanceReport) -> str:
    return " ".join(f"{item.title} {item.excerpt}" for item in report.evidence)


def _validate_profitability_stability(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    combined = _combined_text(governance_report)
    names = _source_names(governance_report)
    if "若只看財報結構推估" in combined and not _has_source_fragment(names, "news"):
        warnings.append(
            "Loss-year reason is inferred from financial structure "
            "without direct news corroboration."
        )
        confidence_score = min(confidence_score, 0.75)
    return confidence_score


def _validate_fundamental_valuation(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    from llm_stock_system.core.fundamental_valuation import has_fundamental_evidence, has_valuation_evidence

    if not has_fundamental_evidence(governance_report):
        warnings.append(
            "Combined fundamental and valuation review is missing "
            "direct fundamental evidence "
            "(facet-based cap already applied if sync failed)."
        )
    if not has_valuation_evidence(governance_report):
        warnings.append(
            "Combined fundamental and valuation review is missing "
            "direct valuation evidence "
            "(facet-based cap already applied if sync failed)."
        )
    return confidence_score


def _validate_gross_margin_comparison(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    names = _source_names(governance_report)
    normalized = _combined_text(governance_report).lower()

    has_financial = _has_source_fragment(
        names,
        "financialstatements",
        "financial_statements",
    )
    if not has_financial:
        warnings.append(
            "Gross margin comparison review: missing financial-statement "
            "evidence (facet-based cap already applied if sync failed)."
        )
        return min(confidence_score, 0.25)

    primary_label = (query.company_name or query.ticker or "").lower()
    comparison_label = (query.comparison_company_name or query.comparison_ticker or "").lower()
    has_primary = bool(primary_label) and primary_label in normalized
    has_comparison = bool(comparison_label) and comparison_label in normalized

    if not (has_primary and has_comparison) or len(governance_report.evidence) < 3:
        warnings.append(
            "Gross margin comparison review: evidence only partially "
            "covers both companies."
        )
        return min(confidence_score, 0.50)

    return confidence_score


def _validate_price_outlook(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    from llm_stock_system.core.target_price import (
        extract_target_price_values,
        has_forward_price_context,
        has_price_level_context,
        has_target_price_context,
        is_forward_price_question,
        is_target_price_question,
    )

    # --- Forecast queries are governed by the ForecastBlock-based caps in
    # ValidationLayer._apply_forecast_cap, so skip the legacy directional
    # cap logic that would otherwise double-penalise them. ---
    if getattr(query, "is_forecast_query", False):
        # Still warn, but do NOT cap here — the forecast mode cap handles it.
        if governance_report.evidence:
            warnings.append(
                "Forecast query: direction/range assessment delegated to "
                "ForecastBlock mode-based confidence cap."
            )
        return confidence_score

    if not is_forward_price_question(query):
        if has_forward_price_context(query, governance_report):
            warnings.append(
                "Directional price query has related market context but "
                "cannot confirm specific future price movement from "
                "public data."
            )
            return min(confidence_score, 0.55)
        warnings.append(
            "Directional price query lacks analyst, technical, or "
            "price-level evidence to support a directional view."
        )
        return min(confidence_score, 0.25)

    if is_target_price_question(query):
        has_direct_numeric = bool(extract_target_price_values(governance_report))
        has_directional_ctx = has_target_price_context(governance_report)
    else:
        has_direct_numeric = has_price_level_context(query, governance_report)
        has_directional_ctx = bool(extract_target_price_values(governance_report)) or has_forward_price_context(
            query,
            governance_report,
        )

    if has_direct_numeric:
        warnings.append(
            "Forward price answers remain scenario-dependent even "
            "when a numeric level is mentioned."
        )
        return min(confidence_score, 0.55)
    if has_directional_ctx:
        warnings.append(
            "Forward price query is supported only by directional "
            "analyst or technical context without direct numeric evidence."
        )
        return min(confidence_score, 0.55)

    warnings.append(
        "Forward price query lacks direct target-price, threshold, "
        "or analyst evidence."
    )
    return min(confidence_score, 0.25)


TAG_SHIPPING = "航運"
TAG_ELECTRICITY = "電價"
TAG_CPI = "CPI"
TAG_MONTHLY_REVENUE = "月營收"
TAG_GROSS_MARGIN = "毛利率"
TAG_FUNDAMENTAL = "基本面"
TAG_VALUATION = "本益比"
TAG_PRICE = "股價"
TAG_EX_DIVIDEND = "除息"
TAG_CASH_FLOW = "現金流"
TAG_DEBT = "負債"
TAG_STABILITY = "穩定性"
TAG_COMPARISON = "比較"
TAG_TECHNICAL = "技術面"
TAG_MARGIN_FLOW = "籌碼"
TAG_INVESTMENT_ASSESSMENT = "投資評估"


VALIDATION_PROFILES: dict[Intent, ValidationProfile] = {
    Intent.NEWS_DIGEST: ValidationProfile(
        intent=Intent.NEWS_DIGEST,
        tag_rules=(
            TagConditionRule(
                required_tags=frozenset({TAG_SHIPPING}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={"keywords": ["紅海", "scfi", "運價", "航運"], "match_mode": "any"},
                        cap=0.25,
                        warning="Shipping rate impact review requires freight-rate evidence.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={
                            "keywords": ["目標價", "評等", "分析師", "法人", "外資"],
                            "match_mode": "any",
                        },
                        cap=0.25,
                        warning="Shipping rate impact review requires analyst-target evidence.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                        params={"threshold": 2},
                        cap=0.25,
                        warning="Shipping rate impact review requires at least 2 evidence items.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.COMPARISON_COMPANY_MISSING,
                        params={},
                        cap=0.75,
                        warning="Shipping rate impact comparison is only partially grounded for one of the companies.",
                    ),
                ),
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_ELECTRICITY}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={
                            "keywords": ["工業電價", "電價", "電費", "成本", "用電大戶"],
                            "match_mode": "any",
                        },
                        cap=0.25,
                        warning="Electricity cost impact review requires grounded electricity-cost evidence.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                        params={"threshold": 2},
                        cap=0.25,
                        warning="Electricity cost impact review requires at least 2 evidence items.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={
                            "keywords": ["因應", "對策", "節能", "節電", "轉嫁", "調價", "綠電", "自發電"],
                            "match_mode": "any",
                        },
                        cap=0.75,
                        warning="Electricity cost impact review is missing concrete response evidence.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.COMPARISON_COMPANY_MISSING,
                        params={},
                        cap=0.75,
                        warning="Electricity cost impact comparison is only partially grounded for one of the companies.",
                    ),
                ),
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_CPI}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={
                            "keywords": ["cpi", "通膨", "利率", "殖利率", "高殖利率"],
                            "match_mode": "any",
                        },
                        cap=0.25,
                        warning="Macro yield sentiment review requires grounded macro and yield-sentiment evidence.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                        params={"threshold": 2},
                        cap=0.25,
                        warning="Macro yield sentiment review requires at least 2 evidence items.",
                    ),
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={"keywords": ["法人", "外資", "觀點", "看法", "保守", "防禦"], "match_mode": "any"},
                        cap=0.75,
                        warning="Macro yield sentiment review is missing concrete institutional views.",
                    ),
                ),
            ),
        ),
    ),
    Intent.EARNINGS_REVIEW: ValidationProfile(
        intent=Intent.EARNINGS_REVIEW,
        tag_rules=(
            TagConditionRule(
                required_tags=frozenset({TAG_MONTHLY_REVENUE}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                        params={"fragments": ["twse"]},
                        warning=(
                            "Monthly revenue YoY review: no TWSE revenue evidence "
                            "in governance report (facet-based cap already applied if sync failed)."
                        ),
                    ),
                    ValidationRule(
                        condition=ConditionKind.ANSWER_CONTAINS_TOKEN,
                        params={"tokens": ["部分月份", "僅能確認", "資料僅到", "尚未公告"]},
                        warning="Monthly revenue data is only partially available for the requested period.",
                    ),
                ),
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_GROSS_MARGIN}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                        params={"fragments": ["financialstatements", "financial_statements"]},
                        warning=(
                            "Margin turnaround review: missing financial statement source "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={"keywords": ["毛利", "毛利率", "gross margin"], "match_mode": "any"},
                        warning=(
                            "Margin turnaround review: no gross-margin keyword found in evidence text "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                    ValidationRule(
                        condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                        params={
                            "keywords": ["營業利益", "營益率", "operating income", "operating margin"],
                            "match_mode": "any",
                        },
                        warning=(
                            "Margin turnaround review: no operating-income keyword found in evidence text "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                    ValidationRule(
                        condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                        params={"threshold": 3},
                        warning=(
                            "Margin turnaround review: fewer than 3 evidence items for multi-year analysis "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                ),
            ),
        ),
    ),
    Intent.VALUATION_CHECK: ValidationProfile(
        intent=Intent.VALUATION_CHECK,
        tag_rules=(
            TagConditionRule(
                required_tags=frozenset({TAG_FUNDAMENTAL, TAG_VALUATION}),
                custom_validator=_validate_fundamental_valuation,
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_VALUATION}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                        params={"fragments": ["twse"]},
                        warning=(
                            "PE valuation review: no official valuation evidence in governance report "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                ),
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_PRICE}),
                custom_validator=_validate_price_outlook,
            ),
        ),
    ),
    Intent.DIVIDEND_ANALYSIS: ValidationProfile(
        intent=Intent.DIVIDEND_ANALYSIS,
        tag_rules=(
            TagConditionRule(
                required_tags=frozenset({TAG_EX_DIVIDEND}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                        params={"fragments": ["dividend"]},
                        warning=(
                            "Ex-dividend performance: missing dividend evidence "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                    ValidationRule(
                        condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                        params={"fragments": ["price"]},
                        warning=(
                            "Ex-dividend performance: missing price evidence "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                ),
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_CASH_FLOW}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.DUAL_SIGNAL_MISSING,
                        params={
                            "signal_a_fragments": ["cashflows", "cash_flow"],
                            "signal_b_fragments": ["dividend"],
                            "both_missing_cap": 0.25,
                            "one_missing_cap": 0.50,
                            "both_missing_warning": (
                                "FCF dividend sustainability review: missing both cash-flow and dividend evidence."
                            ),
                            "one_missing_warning": (
                                "FCF dividend sustainability review: missing one of cash flow or dividend evidence."
                            ),
                        },
                    ),
                ),
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_DEBT}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.DUAL_SIGNAL_MISSING,
                        params={
                            "signal_a_fragments": ["balancesheet", "balance_sheet"],
                            "signal_b_fragments": ["dividend"],
                            "both_missing_cap": 0.25,
                            "one_missing_cap": 0.50,
                            "both_missing_warning": (
                                "Debt dividend safety review: missing both balance-sheet and dividend evidence."
                            ),
                            "one_missing_warning": (
                                "Debt dividend safety review: missing one of balance-sheet or dividend evidence."
                            ),
                        },
                    ),
                ),
            ),
        ),
    ),
    Intent.FINANCIAL_HEALTH: ValidationProfile(
        intent=Intent.FINANCIAL_HEALTH,
        tag_rules=(
            TagConditionRule(
                required_tags=frozenset({TAG_STABILITY}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                        params={"fragments": ["financialstatements", "financial_statements"]},
                        warning=(
                            "Profitability stability review: missing financial statement source in governance report "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                    ValidationRule(
                        condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                        params={"threshold": 3},
                        warning=(
                            "Profitability stability review: fewer than 3 evidence items for multi-year analysis "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                ),
                custom_validator=_validate_profitability_stability,
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_COMPARISON}),
                custom_validator=_validate_gross_margin_comparison,
            ),
        ),
    ),
    Intent.TECHNICAL_VIEW: ValidationProfile(
        intent=Intent.TECHNICAL_VIEW,
        tag_rules=(
            TagConditionRule(
                required_tags=frozenset({TAG_TECHNICAL}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                        params={"fragments": ["price"]},
                        warning=(
                            "Technical indicator review: no price evidence in governance report "
                            "(facet-based cap already applied if sync failed)."
                        ),
                    ),
                ),
            ),
            TagConditionRule(
                required_tags=frozenset({TAG_MARGIN_FLOW}),
                rules=(
                    ValidationRule(
                        condition=ConditionKind.DUAL_SIGNAL_MISSING,
                        params={
                            "signal_a_fragments": ["price"],
                            "signal_b_fragments": ["marginpurchase", "margin_purchase"],
                            "both_missing_cap": 0.25,
                            "one_missing_cap": 0.50,
                            "both_missing_warning": (
                                "Season line and margin review: missing both price and margin evidence."
                            ),
                            "one_missing_warning": (
                                "Season line and margin review: missing one of price or margin evidence."
                            ),
                        },
                    ),
                ),
            ),
        ),
    ),
    Intent.INVESTMENT_ASSESSMENT: ValidationProfile(
        intent=Intent.INVESTMENT_ASSESSMENT,
        tag_rules=(
            TagConditionRule(
                required_tags=frozenset({TAG_INVESTMENT_ASSESSMENT}),
                custom_validator=_validate_fundamental_valuation,
            ),
        ),
    ),
}


def get_profile(intent: Intent) -> ValidationProfile | None:
    """Return the profile for *intent*, or ``None`` if not registered."""
    return VALIDATION_PROFILES.get(intent)
