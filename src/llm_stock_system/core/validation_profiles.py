"""Declarative validation profiles keyed by intent and topic-tag discriminators.

Wave 3 sunset：dividend / earnings 的 deep-dive 路徑（ex-dividend、FCF
sustainability、debt safety、profitability stability、gross margin comparison、
margin turnaround、season line + margin）已全部下架。剩下的 profile 只留：
  * NEWS_DIGEST 的三條產業 tag（航運 / 電價 / CPI）
  * EARNINGS_REVIEW 的 TAG_MONTHLY_REVENUE（營收年增）
  * TECHNICAL_VIEW 的 TAG_TECHNICAL（基本價格證據）
"""

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


TAG_SHIPPING = "航運"
TAG_ELECTRICITY = "電價"
TAG_CPI = "CPI"
TAG_MONTHLY_REVENUE = "月營收"
TAG_TECHNICAL = "技術面"


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
        ),
    ),
}


def get_profile(intent: Intent) -> ValidationProfile | None:
    """Return the profile for *intent*, or ``None`` if not registered."""
    return VALIDATION_PROFILES.get(intent)
