"""Digest refusal policy enums (§14)。

所有 enum 值對應 `digest_refusal_boundary_v1_1.md` §14。

v1.1 Dormant 標註：
- R9 INSUFFICIENT_TAG_COVERAGE (§18.5)
- D4 EVIDENCE_CONFLICT (§18.5)
- D5 PARTIAL_TAG_COVERAGE (§18.5)

Dormant enum 值仍保留於 schema，以保留 log / UI 欄位不變、利於 v1.2 啟用。
"""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """§14.1 — digest query 只允許三種結果。"""

    REFUSE = "refuse"
    DEGRADED = "degraded"
    NORMAL = "normal"


class RefusalCategory(str, Enum):
    """§14.2 — refusal 的來源層。"""

    PARSE = "parse"
    CLASSIFICATION = "classification"
    EVIDENCE = "evidence"
    GOVERNANCE = "governance"


class RefusalReason(str, Enum):
    """§14.3 — 具體 refusal 原因。

    v1.1 Dormant:
        - INSUFFICIENT_TAG_COVERAGE (§18.5)
    """

    # PARSE
    UNRESOLVED_TICKER = "unresolved_ticker"

    # CLASSIFICATION
    CLASSIFIER_UNKNOWN = "classifier_unknown"
    EXPLICIT_OUT_OF_SCOPE_TIME_RANGE = "explicit_out_of_scope_time_range"
    OUT_OF_SCOPE_QUERY_KIND = "out_of_scope_query_kind"

    # EVIDENCE
    NO_EVIDENCE = "no_evidence"
    STALE_EVIDENCE_ONLY = "stale_evidence_only"

    # GOVERNANCE
    LOW_TRUST_SINGLE_SOURCE = "low_trust_single_source"
    TOPIC_MISMATCH = "topic_mismatch"
    INSUFFICIENT_TAG_COVERAGE = "insufficient_tag_coverage"  # v1.1 dormant


class DegradedCategory(str, Enum):
    """§14.4 — degraded 訊號的來源層。"""

    EVIDENCE = "evidence"
    FRESHNESS = "freshness"
    CONSISTENCY = "consistency"
    CLASSIFICATION = "classification"


class DegradedReason(str, Enum):
    """§14.5 — 具體 degraded 原因。可同時多個（見 §12）。

    v1.1 Dormant:
        - EVIDENCE_CONFLICT (§18.5)
        - PARTIAL_TAG_COVERAGE (§18.5)
    """

    SINGLE_HIGH_TRUST_SOURCE = "single_high_trust_source"
    WEAK_CROSS_VALIDATION = "weak_cross_validation"
    NO_HIGH_TRUST_SOURCE = "no_high_trust_source"
    BORDERLINE_FRESHNESS = "borderline_freshness"
    EVIDENCE_CONFLICT = "evidence_conflict"  # v1.1 dormant
    PARTIAL_TAG_COVERAGE = "partial_tag_coverage"  # v1.1 dormant


# --- v1.1 Dormant set ---
# CI / runtime assertions can use this to confirm dormant rules never fire.
# See §18.5.

DORMANT_REFUSAL_REASONS: frozenset[RefusalReason] = frozenset(
    {RefusalReason.INSUFFICIENT_TAG_COVERAGE}
)

DORMANT_DEGRADED_REASONS: frozenset[DegradedReason] = frozenset(
    {DegradedReason.EVIDENCE_CONFLICT, DegradedReason.PARTIAL_TAG_COVERAGE}
)


CLASSIFIER_TAG_COVERAGE_VALUES: frozenset[str] = frozenset(
    {"sufficient", "partial", "insufficient"}
)
"""§5.9 — classifier_tag_coverage 的值域（保留字串而非 enum 以與 log schema 對齊）。"""
