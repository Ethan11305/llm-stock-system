"""三階段 Early-Exit Refusal Policy（§8、§19）。

本檔 **only** 實作三個 checkpoint 函式：

- `early_refusal(query, classifier_result)` — Checkpoint 1 (§9)
- `retrieval_refusal(query, retrieved_documents, query_time)` — Checkpoint 2 (§10)
- `governance_decision(query, governance_report, classifier_result,
                       query_time, evidence_source_types=None)` — Checkpoint 3 (§11 + §12)

每個 checkpoint 回傳：
- `PolicyDecision`：若規則命中（REFUSE 或 DEGRADED 或 NORMAL）
- `None`：若此階段未命中任何 refusal，呼叫端應繼續下一階段

此檔絕對不 import `legacy_qa.*`、`core.models.infer_intent_from_question_type`、
`core.models.QUESTION_TYPE_FALLBACK_TOPIC_TAGS`。
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ...core.models import Document, Evidence, GovernanceReport
from .enums import DegradedReason, RefusalCategory, RefusalReason
from .models import ClassifierResult, DigestQuery, PolicyDecision
from .terms import (
    all_stale,
    cross_validated,
    evidence_conflict,
    freshness_strong,
    high_trust_count,
    hits_out_of_scope_keywords,
    topic_mismatch,
)


# ---------------------------------------------------------------------------
# Checkpoint 1 — Early Refusal (§9)
# ---------------------------------------------------------------------------

def early_refusal(
    query: DigestQuery,
    classifier_result: ClassifierResult,
) -> PolicyDecision | None:
    """§9 — parse / scope guard / classifier 後，retrieval 前判斷。

    檢查順序（依成本由低到高）：

    1. R1 UNRESOLVED_TICKER
    2. R2 CLASSIFIER_UNKNOWN
    3. R4 EXPLICIT_OUT_OF_SCOPE_TIME_RANGE
    4. R4b OUT_OF_SCOPE_QUERY_KIND
    """
    # R1
    if query.ticker is None:
        return PolicyDecision.refuse(
            RefusalCategory.PARSE, RefusalReason.UNRESOLVED_TICKER
        )

    # R2
    if classifier_result.status == "unknown":
        return PolicyDecision.refuse(
            RefusalCategory.CLASSIFICATION, RefusalReason.CLASSIFIER_UNKNOWN
        )

    # R4
    if query.time_range_explicit and query.time_range_label != "7d":
        return PolicyDecision.refuse(
            RefusalCategory.CLASSIFICATION,
            RefusalReason.EXPLICIT_OUT_OF_SCOPE_TIME_RANGE,
        )

    # R4b
    if hits_out_of_scope_keywords(query.user_query) is not None:
        return PolicyDecision.refuse(
            RefusalCategory.CLASSIFICATION,
            RefusalReason.OUT_OF_SCOPE_QUERY_KIND,
        )

    return None


# ---------------------------------------------------------------------------
# Checkpoint 2 — Retrieval Refusal (§10)
# ---------------------------------------------------------------------------

def retrieval_refusal(
    query: DigestQuery,  # noqa: ARG001  — reserved for future topic-aware checks
    retrieved_documents: Sequence[Document],
    query_time: datetime,
) -> PolicyDecision | None:
    """§10 — retrieval 後，governance 前判斷。

    檢查：
    1. R5 NO_EVIDENCE
    2. R6 STALE_EVIDENCE_ONLY
    """
    # R5
    if len(retrieved_documents) == 0:
        return PolicyDecision.refuse(
            RefusalCategory.EVIDENCE, RefusalReason.NO_EVIDENCE
        )

    # R6
    # Document 具備 published_at，直接在 document list 上檢查。
    # 若所有 document 都不在近 7 天窗內，視為 STALE_EVIDENCE_ONLY。
    from .terms import within_7d  # 局部 import 避免循環

    if all(
        not within_7d(doc.published_at, query_time)
        for doc in retrieved_documents
    ):
        return PolicyDecision.refuse(
            RefusalCategory.EVIDENCE, RefusalReason.STALE_EVIDENCE_ONLY
        )

    return None


# ---------------------------------------------------------------------------
# Checkpoint 3 — Governance Decision (§11 + §12)
# ---------------------------------------------------------------------------

def governance_decision(
    query: DigestQuery,
    governance_report: GovernanceReport,
    classifier_result: ClassifierResult,
    query_time: datetime,
    evidence_source_types: list[str] | None = None,
) -> PolicyDecision:
    """§11 + §12 — governance 後判斷 refusal / degraded / normal。

    此 checkpoint **不會**回 None：governance 階段必定產出三態之一。

    檢查順序（依 §19 偽代碼）：

    Refusal（依序，first-match-wins）:
    - R7 LOW_TRUST_SINGLE_SOURCE
    - R8 TOPIC_MISMATCH
    - R9 INSUFFICIENT_TAG_COVERAGE (v1.1 dormant, 仍寫進 code 以便 v1.2 啟用)

    DEGRADED D0 （early return，§12 D0 說明）:
    - SINGLE_HIGH_TRUST_SOURCE

    DEGRADED D1-D5 （多 reason 收集）:
    - WEAK_CROSS_VALIDATION
    - NO_HIGH_TRUST_SOURCE
    - BORDERLINE_FRESHNESS
    - EVIDENCE_CONFLICT (dormant, 函式恆回 False)
    - PARTIAL_TAG_COVERAGE (dormant, 因 MINIMUM_TAG_SET 全空)

    參數 `evidence_source_types`: 見 `topic_mismatch()` 的說明。若 governance
    layer 能提供對應 evidence 的 source_type，應傳入此參數；否則 v1.1 會
    保守不觸發 TOPIC_MISMATCH。
    """
    evidence: list[Evidence] = list(governance_report.evidence)
    htc = high_trust_count(evidence)

    # ---- R7 LOW_TRUST_SINGLE_SOURCE ----
    if len(evidence) == 1 and htc == 0:
        return PolicyDecision.refuse(
            RefusalCategory.GOVERNANCE, RefusalReason.LOW_TRUST_SINGLE_SOURCE
        )

    # ---- R8 TOPIC_MISMATCH ----
    if topic_mismatch(query.topic, evidence, evidence_source_types):
        return PolicyDecision.refuse(
            RefusalCategory.GOVERNANCE, RefusalReason.TOPIC_MISMATCH
        )

    # ---- R9 INSUFFICIENT_TAG_COVERAGE (v1.1 dormant) ----
    # 在 v1.1 MINIMUM_TAG_SET_BY_TOPIC 全為空集合，classifier_tag_coverage
    # 不會回 "insufficient"；此分支保留以便 v1.2 啟用。
    if classifier_result.tag_coverage == "insufficient":
        return PolicyDecision.refuse(
            RefusalCategory.GOVERNANCE, RefusalReason.INSUFFICIENT_TAG_COVERAGE
        )

    # ---- D0 SINGLE_HIGH_TRUST_SOURCE — early return (§12 D0 說明) ----
    if len(evidence) == 1 and htc == 1:
        return PolicyDecision.degraded([DegradedReason.SINGLE_HIGH_TRUST_SOURCE])

    # ---- D1-D5 多 reason 收集 ----
    degraded_reasons: list[DegradedReason] = []

    # D1 WEAK_CROSS_VALIDATION — 僅對 len(evidence) >= 2 有意義
    if len(evidence) >= 2 and not cross_validated(evidence):
        degraded_reasons.append(DegradedReason.WEAK_CROSS_VALIDATION)

    # D2 NO_HIGH_TRUST_SOURCE
    # 注意：此處到達時 (1, 0) 已被 R7 攔截、(1, 1) 已被 D0 攔截，
    # 故此分支隱含 len(evidence) >= 2。
    if htc == 0:
        degraded_reasons.append(DegradedReason.NO_HIGH_TRUST_SOURCE)

    # D3 BORDERLINE_FRESHNESS
    if not freshness_strong(evidence, query_time, query.topic):
        degraded_reasons.append(DegradedReason.BORDERLINE_FRESHNESS)

    # D4 EVIDENCE_CONFLICT — v1.1 dormant (evidence_conflict() 恆回 False)
    if evidence_conflict(governance_report):
        degraded_reasons.append(DegradedReason.EVIDENCE_CONFLICT)

    # D5 PARTIAL_TAG_COVERAGE — v1.1 dormant (tag_coverage 在空 MINIMUM_TAG_SET
    # 下恆為 "sufficient"；此分支保留以便 v1.2 啟用)
    if classifier_result.tag_coverage == "partial":
        degraded_reasons.append(DegradedReason.PARTIAL_TAG_COVERAGE)

    if degraded_reasons:
        return PolicyDecision.degraded(degraded_reasons)

    return PolicyDecision.normal()


__all__ = [
    "early_refusal",
    "governance_decision",
    "retrieval_refusal",
]
