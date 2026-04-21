"""Digest policy 層的資料模型。

包含：
- `DigestQuery` (§6.1) — digest path 的封閉輸入 schema
- `ClassifierResult` — classifier / post-processing 輸出
- `PolicyDecision` — refusal / degraded / normal 的統一回傳型別

設計原則：
- DigestQuery 不含 `question_type`、`comparison_ticker`、forecast 等 legacy 欄位。
- `intent` 與 `query_profile` 固定為 digest 封閉集合。
- PolicyDecision 是 frozen dataclass，方便在函式間以不可變物件傳遞。

引用的 shared 型別只有 `Topic`, `TopicTag`, `QueryProfile`, `Intent` 與
`ConfidenceLight`，全部來自 `core.enums`——這些是純 enum，無 legacy 邏輯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...core.enums import (
    ConfidenceLight,
    Intent,
    QueryProfile,
    Topic,
    TopicTag,
)
from .enums import (
    DegradedReason,
    Outcome,
    RefusalCategory,
    RefusalReason,
)


# §5.5 Digest 封閉 intent 集合
_DIGEST_ALLOWED_INTENTS: frozenset[Intent] = frozenset({Intent.NEWS_DIGEST})

# §5.5 Digest 封閉 topic 集合
_DIGEST_ALLOWED_TOPICS: frozenset[Topic] = frozenset(
    {Topic.NEWS, Topic.ANNOUNCEMENT, Topic.COMPOSITE}
)


class DigestQuery(BaseModel):
    """§6.1 — digest path 的最低必要輸入欄位。

    此 model 刻意不含 `question_type` / `comparison_ticker` / forecast 欄位，
    避免 legacy 語意污染。所有欄位皆由 digest input layer 填入，不經
    `infer_intent_from_question_type()` 反推。
    """

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    user_query: str
    ticker: str | None
    company_name: str | None = None
    topic: Topic
    time_range_label: str = "7d"
    time_range_days: int = 7
    time_range_explicit: bool = False
    intent: Intent = Intent.NEWS_DIGEST
    controlled_tags: tuple[TopicTag, ...] = Field(default_factory=tuple)
    topic_tags: tuple[str, ...] = Field(default_factory=tuple)
    classifier_source: str = "llm"
    classifier_raw_score: float | None = None
    classifier_tag_coverage: str = "sufficient"
    query_profile: QueryProfile = QueryProfile.SINGLE_STOCK_DIGEST

    @field_validator("intent")
    @classmethod
    def _intent_must_be_in_digest_closed_set(cls, value: Intent) -> Intent:
        if value not in _DIGEST_ALLOWED_INTENTS:
            raise ValueError(
                f"DigestQuery.intent must be one of {_DIGEST_ALLOWED_INTENTS}; "
                f"got {value!r}. See spec §5.5."
            )
        return value

    @field_validator("topic")
    @classmethod
    def _topic_must_be_in_digest_closed_set(cls, value: Topic) -> Topic:
        if value not in _DIGEST_ALLOWED_TOPICS:
            raise ValueError(
                f"DigestQuery.topic must be one of {_DIGEST_ALLOWED_TOPICS}; "
                f"got {value!r}. See spec §5.5."
            )
        return value

    @field_validator("query_profile")
    @classmethod
    def _query_profile_must_be_digest(cls, value: QueryProfile) -> QueryProfile:
        if value is not QueryProfile.SINGLE_STOCK_DIGEST:
            raise ValueError(
                "DigestQuery.query_profile must be SINGLE_STOCK_DIGEST. "
                "Legacy path should not construct DigestQuery."
            )
        return value

    @field_validator("classifier_tag_coverage")
    @classmethod
    def _tag_coverage_enum_guard(cls, value: str) -> str:
        allowed = {"sufficient", "partial", "insufficient"}
        if value not in allowed:
            raise ValueError(
                f"classifier_tag_coverage must be one of {allowed}; got {value!r}."
            )
        return value


class ClassifierResult(BaseModel):
    """Classifier + deterministic post-processing 的輸出。

    `tag_coverage` 由規則引擎算出（見 §5.9），不是 LLM 自評結果。

    v1.1 `status` 值域：`"ok"` | `"unknown"`。
    若將來 classifier 輸出需要更豐富的狀態，再擴展此 enum。
    """

    model_config = ConfigDict(frozen=True)

    status: str = "ok"  # "ok" | "unknown"
    raw_score: float | None = None
    tag_coverage: str = "sufficient"  # "sufficient" | "partial" | "insufficient"
    predicted_topic: Topic | None = None
    predicted_tags: frozenset[TopicTag] = Field(default_factory=frozenset)

    @field_validator("status")
    @classmethod
    def _status_enum_guard(cls, value: str) -> str:
        allowed = {"ok", "unknown"}
        if value not in allowed:
            raise ValueError(f"status must be one of {allowed}; got {value!r}.")
        return value

    @field_validator("tag_coverage")
    @classmethod
    def _tag_coverage_enum_guard(cls, value: str) -> str:
        allowed = {"sufficient", "partial", "insufficient"}
        if value not in allowed:
            raise ValueError(
                f"tag_coverage must be one of {allowed}; got {value!r}."
            )
        return value


@dataclass(frozen=True)
class PolicyDecision:
    """統一 refusal / degraded / normal 決策結果。

    REFUSE: `refusal_category` 與 `refusal_reason` 必填，
            `confidence_light` / `confidence_score` 為 None (§7.1)。
    DEGRADED: `degraded_reasons` 至少 1 筆，
              `confidence_score ∈ [0.30, 0.69]`, `confidence_light ∈ {RED, YELLOW}` (§7.2)。
    NORMAL: `confidence_score ∈ [0.70, 1.00]`, `confidence_light = GREEN` (§7.3)。

    v1.1 `confidence_score` / `confidence_light` 於 DEGRADED / NORMAL 預設為 None；
    實際 scoring 由後續 validation / presentation 層決定。policy 層本身只
    提供結構化結論，不負責量化 score（保留此欄位便於後續層覆寫）。
    """

    outcome: Outcome
    refusal_category: Optional[RefusalCategory] = None
    refusal_reason: Optional[RefusalReason] = None
    degraded_reasons: tuple[DegradedReason, ...] = field(default_factory=tuple)
    confidence_light: Optional[ConfidenceLight] = None
    confidence_score: Optional[float] = None

    # ----- convenient constructors -----

    @classmethod
    def refuse(
        cls,
        category: RefusalCategory,
        reason: RefusalReason,
    ) -> "PolicyDecision":
        return cls(
            outcome=Outcome.REFUSE,
            refusal_category=category,
            refusal_reason=reason,
            degraded_reasons=(),
            confidence_light=None,  # §7.1
            confidence_score=None,  # §7.1
        )

    @classmethod
    def degraded(
        cls,
        reasons: list[DegradedReason] | tuple[DegradedReason, ...],
    ) -> "PolicyDecision":
        if not reasons:
            raise ValueError(
                "PolicyDecision.degraded() requires at least one reason. "
                "See spec §7.2."
            )
        return cls(
            outcome=Outcome.DEGRADED,
            degraded_reasons=tuple(reasons),
        )

    @classmethod
    def normal(cls) -> "PolicyDecision":
        return cls(outcome=Outcome.NORMAL)

    # ----- invariants -----

    def __post_init__(self) -> None:
        # Invariants per §7.
        if self.outcome is Outcome.REFUSE:
            if self.refusal_category is None or self.refusal_reason is None:
                raise ValueError(
                    "REFUSE decisions must carry refusal_category and refusal_reason."
                )
            if self.confidence_light is not None or self.confidence_score is not None:
                raise ValueError(
                    "REFUSE decisions must not carry confidence light/score (§7.1)."
                )
            if self.degraded_reasons:
                raise ValueError("REFUSE decisions must not carry degraded_reasons.")
        elif self.outcome is Outcome.DEGRADED:
            if not self.degraded_reasons:
                raise ValueError(
                    "DEGRADED decisions must carry at least one degraded_reason (§7.2)."
                )
            if self.refusal_category is not None or self.refusal_reason is not None:
                raise ValueError(
                    "DEGRADED decisions must not carry refusal_category/reason."
                )
        elif self.outcome is Outcome.NORMAL:
            if self.degraded_reasons:
                raise ValueError("NORMAL decisions must not carry degraded_reasons.")
            if self.refusal_category is not None or self.refusal_reason is not None:
                raise ValueError(
                    "NORMAL decisions must not carry refusal_category/reason."
                )


__all__ = [
    "ClassifierResult",
    "DigestQuery",
    "PolicyDecision",
]
