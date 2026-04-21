"""Digest refusal / degraded / normal 決策層。

對應 `digest_refusal_boundary_v1_1.md` §19 三階段 early-exit 流程。

公開 API：

- `early_refusal(query, classifier_result)`  → Checkpoint 1
- `retrieval_refusal(query, retrieved_documents, query_time)` → Checkpoint 2
- `governance_decision(query, governance_report, classifier_result, query_time,
                       evidence_source_types=None)` → Checkpoint 3

所有 checkpoint 函式回傳 `PolicyDecision | None`；`None` 代表繼續下一階段。
"""

from .enums import (
    DegradedCategory,
    DegradedReason,
    Outcome,
    RefusalCategory,
    RefusalReason,
)
from .models import ClassifierResult, DigestQuery, PolicyDecision
from .refusal_policy import (
    early_refusal,
    governance_decision,
    retrieval_refusal,
)

__all__ = [
    "ClassifierResult",
    "DegradedCategory",
    "DegradedReason",
    "DigestQuery",
    "Outcome",
    "PolicyDecision",
    "RefusalCategory",
    "RefusalReason",
    "early_refusal",
    "governance_decision",
    "retrieval_refusal",
]
