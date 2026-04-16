"""Declarative validation profiles for question-type confidence scoring.

Replaces the 17 if-branches in ``ValidationLayer._apply_question_type_rules()``
with data-driven profiles.  Each profile declares the evidence checks and
cap/warning rules for one ``question_type``.

Design principles
-----------------
* **FacetSpec** (``INTENT_FACET_SPECS``) = data-acquisition contract
  → drives ``_apply_required_facet_cap`` / ``_apply_preferred_facet_penalty``
  (steps 3–4 in ``validate()``).
* **ValidationProfile** = evidence-semantics check
  → drives ``_apply_question_type_rules`` (step 5 in ``validate()``).
* The two layers are complementary and must **not** overlap.

Migration plan
--------------
PR1  Define structures + registry + evaluator.  **No behaviour change.**
PR2  Migrate Group A (warning-only) + Group B (three-tier cap) to profiles.
PR3  Migrate Group C (content-keyword / custom) + delete legacy branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from llm_stock_system.core.models import (
        AnswerDraft,
        GovernanceReport,
        StructuredQuery,
    )


# ───────────────────────────────────────────────────────────────────────────
# 1. ConditionKind — the six supported rule-condition types
# ───────────────────────────────────────────────────────────────────────────

class ConditionKind(str, Enum):
    """Condition types supported by the rule engine.

    Anything more complex than these six should use ``custom_validator``.
    """

    SOURCE_FRAGMENT_MISSING = "source_fragment_missing"
    """No ``source_name`` (lowered) contains any of the given fragments.

    params::

        fragments: list[str]  — alternative spellings for the same source
                                (OR-semantics: any fragment match = present)
    """

    CONTENT_KEYWORD_MISSING = "content_keyword_missing"
    """Evidence combined text (lowered) does not contain required keywords.

    params::

        keywords:   list[str]
        match_mode: "any" | "all"
            "any" → triggered when **none** of the keywords appear
            "all" → triggered when **any** keyword is absent
    """

    EVIDENCE_COUNT_BELOW = "evidence_count_below"
    """Number of evidence items is strictly below the threshold.

    params::

        threshold: int
    """

    DUAL_SIGNAL_MISSING = "dual_signal_missing"
    """Two distinct source-fragment signals with three-tier cap semantics.

    params::

        signal_a_fragments:   list[str]
        signal_b_fragments:   list[str]
        both_missing_cap:     float | None
        one_missing_cap:      float | None
        both_missing_warning: str
        one_missing_warning:  str

    NOTE: ``cap``, ``penalty``, and ``warning`` on the ``ValidationRule``
    are **ignored** for this condition.  Caps and warnings are driven
    entirely by ``params`` to support the three-tier semantics.
    """

    COMPARISON_COMPANY_MISSING = "comparison_company_missing"
    """Dual-ticker comparison but one company name is absent from evidence.

    params::

        {}   (labels derived from ``query`` automatically)

    Only triggers when ``query.comparison_ticker`` is set.
    """

    ANSWER_CONTAINS_TOKEN = "answer_contains_token"
    """The LLM answer summary contains one of the specified tokens.

    params::

        tokens: list[str]
    """


# ───────────────────────────────────────────────────────────────────────────
# 2. ValidationRule
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationRule:
    """A single declarative validation check.

    For most condition kinds ``cap`` / ``penalty`` / ``warning`` on this
    dataclass control what happens when the condition triggers.

    Exception: ``DUAL_SIGNAL_MISSING`` — caps and warnings live inside
    ``params`` because the three-tier semantics need two distinct
    cap/warning pairs.
    """

    condition: ConditionKind
    params: dict[str, Any] = field(default_factory=dict)
    cap: float | None = None       # min(score, cap)  when triggered
    penalty: float | None = None   # score -= penalty  when triggered
    warning: str = ""              # appended to warnings when triggered


# ───────────────────────────────────────────────────────────────────────────
# 3. CustomValidatorFn — typed callback protocol
# ───────────────────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────────────────
# 4. ValidationProfile
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationProfile:
    """Declarative confidence-scoring configuration for one ``question_type``.

    The evaluator processes a profile in order:

    1. Evaluate each ``rule`` in sequence (cap/penalty/warning).
    2. If ``custom_validator`` is set, call it last (may override score).
    """

    question_type: str
    rules: tuple[ValidationRule, ...] = ()
    custom_validator: CustomValidatorFn | None = None


# ───────────────────────────────────────────────────────────────────────────
# 5. Standalone helpers (shared by custom validators)
# ───────────────────────────────────────────────────────────────────────────

def _source_names(report: GovernanceReport) -> set[str]:
    """Lowercased set of all source_name values in the report."""
    return {item.source_name.lower() for item in report.evidence}


def _has_source_fragment(source_names: set[str], *fragments: str) -> bool:
    """True if *any* source_name contains *any* of the fragments (OR)."""
    return any(
        fragment in source_name
        for fragment in fragments
        for source_name in source_names
    )


def _combined_text(report: GovernanceReport) -> str:
    """Concatenated title + excerpt of all evidence items."""
    return " ".join(f"{item.title} {item.excerpt}" for item in report.evidence)


# ───────────────────────────────────────────────────────────────────────────
# 6. Custom validators
# ───────────────────────────────────────────────────────────────────────────

# --- Group A (complex) ------------------------------------------------

def _validate_profitability_stability(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    """Compound check: loss-year reason inferred without news corroboration.

    The basic financial-statements / evidence-count warnings are handled
    by declarative rules on the profile.  This validator only handles the
    **cap at 0.75** for the compound condition:

        ``"若只看財報結構推估"`` present in evidence text
        AND no ``"news"`` source in governance report.

    Mirrors ``ValidationLayer._apply_profitability_stability_rules`` L336-356.
    """
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
    """Warning-only check via external fundamental/valuation helpers.

    Shared by ``fundamental_pe_review`` and ``investment_support``.
    Mirrors ``ValidationLayer._apply_fundamental_valuation_rules`` L381-395.
    """
    from llm_stock_system.core.fundamental_valuation import (
        has_fundamental_evidence,
        has_valuation_evidence,
    )

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


# --- Group B (complex) ------------------------------------------------

def _validate_gross_margin_comparison(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    """Sequential short-circuit: financial source → dual company check.

    This must stay as a custom validator because the original logic
    short-circuits (returns early on the first failing tier), and the
    second condition depends on ``query.company_name`` /
    ``query.comparison_company_name``.

    Mirrors ``ValidationLayer._apply_gross_margin_comparison_rules`` L498-522.
    """
    names = _source_names(governance_report)
    normalized = _combined_text(governance_report).lower()

    has_financial = _has_source_fragment(
        names, "financialstatements", "financial_statements",
    )
    if not has_financial:
        warnings.append(
            "Gross margin comparison review: missing financial-statement "
            "evidence (facet-based cap already applied if sync failed)."
        )
        return min(confidence_score, 0.25)

    primary_label = (query.company_name or query.ticker or "").lower()
    comparison_label = (
        query.comparison_company_name or query.comparison_ticker or ""
    ).lower()
    has_primary = bool(primary_label) and primary_label in normalized
    has_comparison = bool(comparison_label) and comparison_label in normalized

    if not (has_primary and has_comparison) or len(governance_report.evidence) < 3:
        warnings.append(
            "Gross margin comparison review: evidence only partially "
            "covers both companies."
        )
        return min(confidence_score, 0.50)

    return confidence_score


# --- Group C ----------------------------------------------------------

def _validate_price_outlook(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    answer_draft: AnswerDraft,
    confidence_score: float,
    warnings: list[str],
) -> float:
    """Full price-outlook validation (forward / directional / target-price).

    This is the most complex branch in the system.  It delegates to
    ``target_price.py`` helpers and has two distinct sub-paths:

    * **Forward** (``is_forward_price_question`` = True):
      Checks for numeric target-price values, then directional context.
    * **Directional** (``is_forward_price_question`` = False):
      Checks for forward-price context only.

    Mirrors ``ValidationLayer._apply_price_outlook_rules`` L254-281
    and ``ValidationLayer._apply_directional_price_rules`` L229-252.
    """
    from llm_stock_system.core.target_price import (
        extract_target_price_values,
        has_forward_price_context,
        has_price_level_context,
        has_target_price_context,
        is_forward_price_question,
        is_target_price_question,
    )

    # ── Sub-path 1: directional (non-forward) questions ──────────────
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

    # ── Sub-path 2: forward questions (target-price / price-level) ───
    if is_target_price_question(query):
        has_direct_numeric = bool(extract_target_price_values(governance_report))
        has_directional_ctx = has_target_price_context(governance_report)
    else:
        has_direct_numeric = has_price_level_context(query, governance_report)
        has_directional_ctx = bool(
            extract_target_price_values(governance_report)
        ) or has_forward_price_context(query, governance_report)

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


# ───────────────────────────────────────────────────────────────────────────
# 7. VALIDATION_PROFILES registry
# ───────────────────────────────────────────────────────────────────────────
#
# Key   = question_type (str)
# Value = ValidationProfile
#
# Every profile mirrors exactly one legacy branch from
# ``ValidationLayer._apply_question_type_rules()``.
#
# question_types that have NO validation branch (e.g. "market_summary",
# "dividend_yield_review") are intentionally absent — the evaluator
# returns the score unchanged when no profile is found.
# ───────────────────────────────────────────────────────────────────────────

VALIDATION_PROFILES: dict[str, ValidationProfile] = {

    # =================================================================
    # Group A — warning-only (9 profiles)
    #
    # Facet-based cap (steps 3/4) handles scoring.
    # These profiles only add diagnostic warnings.
    # =================================================================

    # -- A1: ex_dividend_performance ------------------------------------
    # Legacy: _apply_ex_dividend_rules (L283-295)
    # Checks: source "dividend" AND source "price"
    # Note:  Original emits one combined warning; profile emits per-signal
    #        warnings (more specific, score parity maintained — no cap).
    "ex_dividend_performance": ValidationProfile(
        question_type="ex_dividend_performance",
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

    # -- A2: technical_indicator_review ---------------------------------
    # Legacy: _apply_technical_indicator_rules (L297-307)
    "technical_indicator_review": ValidationProfile(
        question_type="technical_indicator_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                params={"fragments": ["price"]},
                warning=(
                    "Technical indicator review: no price evidence in "
                    "governance report "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
        ),
    ),

    # -- A3: monthly_revenue_yoy_review ---------------------------------
    # Legacy: _apply_monthly_revenue_rules (L309-322)
    # Two independent checks: TWSE source + partial-revenue tokens.
    "monthly_revenue_yoy_review": ValidationProfile(
        question_type="monthly_revenue_yoy_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                params={"fragments": ["twse"]},
                warning=(
                    "Monthly revenue YoY review: no TWSE revenue evidence "
                    "in governance report "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.ANSWER_CONTAINS_TOKEN,
                params={
                    "tokens": ["部分月份", "僅能確認", "資料僅到", "尚未公告"],
                },
                warning=(
                    "Monthly revenue data is only partially available "
                    "for the requested period."
                ),
            ),
        ),
    ),

    # -- A4: pe_valuation_review ----------------------------------------
    # Legacy: _apply_pe_valuation_rules (L324-334)
    "pe_valuation_review": ValidationProfile(
        question_type="pe_valuation_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                params={"fragments": ["twse"]},
                warning=(
                    "PE valuation review: no official valuation evidence "
                    "in governance report "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
        ),
    ),

    # -- A5: profitability_stability_review -----------------------------
    # Legacy: _apply_profitability_stability_rules (L336-356)
    # Rules handle the basic financial-source + evidence-count warning.
    # custom_validator handles the compound "若只看財報結構推估" cap.
    # Note:  Original fires one combined warning for (missing source OR
    #        evidence < 3).  Profile fires per-condition warnings.
    "profitability_stability_review": ValidationProfile(
        question_type="profitability_stability_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                params={
                    "fragments": [
                        "financialstatements",
                        "financial_statements",
                    ],
                },
                warning=(
                    "Profitability stability review: missing financial "
                    "statement source in governance report "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                params={"threshold": 3},
                warning=(
                    "Profitability stability review: fewer than 3 "
                    "evidence items for multi-year analysis "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
        ),
        custom_validator=_validate_profitability_stability,
    ),

    # -- A6: margin_turnaround_review -----------------------------------
    # Legacy: _apply_margin_turnaround_rules (L358-379)
    # Original: one compound (A AND B AND C AND D) → one warning.
    # Profile:  four independent rules → per-condition warnings.
    "margin_turnaround_review": ValidationProfile(
        question_type="margin_turnaround_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.SOURCE_FRAGMENT_MISSING,
                params={
                    "fragments": [
                        "financialstatements",
                        "financial_statements",
                    ],
                },
                warning=(
                    "Margin turnaround review: missing financial "
                    "statement source "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": ["毛利", "毛利率", "gross margin"],
                    "match_mode": "any",
                },
                warning=(
                    "Margin turnaround review: no gross-margin keyword "
                    "found in evidence text "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": [
                        "營業利益",
                        "營益率",
                        "operating income",
                        "operating margin",
                    ],
                    "match_mode": "any",
                },
                warning=(
                    "Margin turnaround review: no operating-income "
                    "keyword found in evidence text "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                params={"threshold": 3},
                warning=(
                    "Margin turnaround review: fewer than 3 evidence "
                    "items for multi-year analysis "
                    "(facet-based cap already applied if sync failed)."
                ),
            ),
        ),
    ),

    # -- A7: fundamental_pe_review --------------------------------------
    # Legacy: _apply_fundamental_valuation_rules (L381-395)
    # Delegates to has_fundamental_evidence / has_valuation_evidence.
    "fundamental_pe_review": ValidationProfile(
        question_type="fundamental_pe_review",
        custom_validator=_validate_fundamental_valuation,
    ),

    # -- A8/A9: investment_support (shares validator with A7) -----------
    "investment_support": ValidationProfile(
        question_type="investment_support",
        custom_validator=_validate_fundamental_valuation,
    ),

    # =================================================================
    # Group B — three-tier cap (4 profiles)
    #
    # Dual-signal source checks with graduated caps:
    #   both missing → cap 0.25 (RED)
    #   one missing  → cap 0.50 (YELLOW)
    #   both present → no cap
    # =================================================================

    # -- B1: season_line_margin_review ----------------------------------
    # Legacy: _apply_season_line_margin_rules (L397-411)
    "season_line_margin_review": ValidationProfile(
        question_type="season_line_margin_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.DUAL_SIGNAL_MISSING,
                params={
                    "signal_a_fragments": ["price"],
                    "signal_b_fragments": [
                        "marginpurchase",
                        "margin_purchase",
                    ],
                    "both_missing_cap": 0.25,
                    "one_missing_cap": 0.50,
                    "both_missing_warning": (
                        "Season line and margin review: "
                        "missing both price and margin evidence."
                    ),
                    "one_missing_warning": (
                        "Season line and margin review: "
                        "missing one of price or margin evidence."
                    ),
                },
            ),
        ),
    ),

    # -- B2: fcf_dividend_sustainability_review -------------------------
    # Legacy: _apply_fcf_dividend_rules (L413-427)
    "fcf_dividend_sustainability_review": ValidationProfile(
        question_type="fcf_dividend_sustainability_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.DUAL_SIGNAL_MISSING,
                params={
                    "signal_a_fragments": ["cashflows", "cash_flow"],
                    "signal_b_fragments": ["dividend"],
                    "both_missing_cap": 0.25,
                    "one_missing_cap": 0.50,
                    "both_missing_warning": (
                        "FCF dividend sustainability review: "
                        "missing both cash-flow and dividend evidence."
                    ),
                    "one_missing_warning": (
                        "FCF dividend sustainability review: "
                        "missing one of cash flow or dividend evidence."
                    ),
                },
            ),
        ),
    ),

    # -- B3: debt_dividend_safety_review --------------------------------
    # Legacy: _apply_debt_dividend_rules (L524-538)
    "debt_dividend_safety_review": ValidationProfile(
        question_type="debt_dividend_safety_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.DUAL_SIGNAL_MISSING,
                params={
                    "signal_a_fragments": [
                        "balancesheet",
                        "balance_sheet",
                    ],
                    "signal_b_fragments": ["dividend"],
                    "both_missing_cap": 0.25,
                    "one_missing_cap": 0.50,
                    "both_missing_warning": (
                        "Debt dividend safety review: "
                        "missing both balance-sheet and "
                        "dividend evidence."
                    ),
                    "one_missing_warning": (
                        "Debt dividend safety review: "
                        "missing one of balance-sheet or "
                        "dividend evidence."
                    ),
                },
            ),
        ),
    ),

    # -- B4: gross_margin_comparison_review -----------------------------
    # Legacy: _apply_gross_margin_comparison_rules (L498-522)
    # Uses custom_validator because the original logic short-circuits
    # and the second tier depends on query.company_name.
    "gross_margin_comparison_review": ValidationProfile(
        question_type="gross_margin_comparison_review",
        custom_validator=_validate_gross_margin_comparison,
    ),

    # =================================================================
    # Group C — content-keyword / custom (4 profiles)
    #
    # These check evidence *text content* (not just source names).
    # shipping / electricity / macro start declarative; can be upgraded
    # to custom_validator if parity tests show keyword noise.
    # =================================================================

    # -- C1: price_outlook ---------------------------------------------
    # Legacy: _apply_price_outlook_rules (L254-281)
    #       + _apply_directional_price_rules (L229-252)
    # Too complex for declarative rules (depends on target_price.py
    # helper functions and query-aware sub-path routing).
    "price_outlook": ValidationProfile(
        question_type="price_outlook",
        custom_validator=_validate_price_outlook,
    ),

    # -- C2: shipping_rate_impact_review --------------------------------
    # Legacy: _apply_shipping_rate_rules (L429-450)
    # Note:  Original short-circuits (cap 0.25 → return) before the
    #        comparison check.  Profile evaluates all rules sequentially;
    #        score parity is maintained via min() semantics, but
    #        an extra comparison warning may appear when the base
    #        condition already caps at 0.25.
    "shipping_rate_impact_review": ValidationProfile(
        question_type="shipping_rate_impact_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": ["紅海", "scfi", "運價", "航運"],
                    "match_mode": "any",
                },
                cap=0.25,
                warning=(
                    "Shipping rate impact review requires "
                    "freight-rate evidence."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": [
                        "目標價", "評等", "分析師", "法人", "外資",
                    ],
                    "match_mode": "any",
                },
                cap=0.25,
                warning=(
                    "Shipping rate impact review requires "
                    "analyst-target evidence."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                params={"threshold": 2},
                cap=0.25,
                warning=(
                    "Shipping rate impact review requires "
                    "at least 2 evidence items."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.COMPARISON_COMPANY_MISSING,
                params={},
                cap=0.75,
                warning=(
                    "Shipping rate impact comparison is only partially "
                    "grounded for one of the companies."
                ),
            ),
        ),
    ),

    # -- C3: electricity_cost_impact_review -----------------------------
    # Legacy: _apply_electricity_cost_rules (L452-478)
    # Same short-circuit caveat as C2.
    "electricity_cost_impact_review": ValidationProfile(
        question_type="electricity_cost_impact_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": [
                        "工業電價", "電價", "電費", "成本", "用電大戶",
                    ],
                    "match_mode": "any",
                },
                cap=0.25,
                warning=(
                    "Electricity cost impact review requires "
                    "grounded electricity-cost evidence."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                params={"threshold": 2},
                cap=0.25,
                warning=(
                    "Electricity cost impact review requires "
                    "at least 2 evidence items."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": [
                        "因應", "對策", "節能", "節電",
                        "轉嫁", "調價", "綠電", "自發電",
                    ],
                    "match_mode": "any",
                },
                cap=0.75,
                warning=(
                    "Electricity cost impact review is missing "
                    "concrete response evidence."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.COMPARISON_COMPANY_MISSING,
                params={},
                cap=0.75,
                warning=(
                    "Electricity cost impact comparison is only "
                    "partially grounded for one of the companies."
                ),
            ),
        ),
    ),

    # -- C4: macro_yield_sentiment_review -------------------------------
    # Legacy: _apply_macro_yield_rules (L480-496)
    "macro_yield_sentiment_review": ValidationProfile(
        question_type="macro_yield_sentiment_review",
        rules=(
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": [
                        "cpi", "通膨", "利率", "殖利率", "高殖利率",
                    ],
                    "match_mode": "any",
                },
                cap=0.25,
                warning=(
                    "Macro yield sentiment review requires "
                    "grounded macro and yield-sentiment evidence."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.EVIDENCE_COUNT_BELOW,
                params={"threshold": 2},
                cap=0.25,
                warning=(
                    "Macro yield sentiment review requires "
                    "at least 2 evidence items."
                ),
            ),
            ValidationRule(
                condition=ConditionKind.CONTENT_KEYWORD_MISSING,
                params={
                    "keywords": [
                        "法人", "外資", "觀點", "看法", "保守", "防禦",
                    ],
                    "match_mode": "any",
                },
                cap=0.75,
                warning=(
                    "Macro yield sentiment review is missing "
                    "concrete institutional views."
                ),
            ),
        ),
    ),
}


# ───────────────────────────────────────────────────────────────────────────
# 8. Convenience accessor
# ───────────────────────────────────────────────────────────────────────────

def get_profile(question_type: str) -> ValidationProfile | None:
    """Return the profile for *question_type*, or ``None`` if not registered."""
    return VALIDATION_PROFILES.get(question_type)
