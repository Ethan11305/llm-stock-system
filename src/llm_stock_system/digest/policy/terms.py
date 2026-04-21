"""§5 術語定義的 deterministic 實作。

本檔封裝所有 refusal / degraded / normal 判斷依賴的 helper：

- `within_7d` (§5.4)
- `cross_validated` (§5.2)
- `topic_mismatch` (§5.3)
- `freshness_strong` (§5.7)
- `evidence_conflict` (§5.8) — **v1.1 dormant，固定回 False**
- `compute_classifier_tag_coverage` (§5.9)
- `hits_out_of_scope_keywords` (§6.2)

設計：
- 所有函式皆為 pure function，無副作用、無網路、無 LLM。
- 輸入輸出皆為資料結構，不引用 digest policy 之外的 service / adapter。
- 這樣可以在 unit test 中以構造資料完整覆蓋所有規則分支。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ...core.enums import SourceTier, Topic, TopicTag
from ...core.models import Evidence, GovernanceReport
from .closed_tables import (
    ALLOWED_KEYWORD_TAGS,
    MINIMUM_TAG_SET_BY_TOPIC,
    OUT_OF_SCOPE_KEYWORDS,
)


# §5.4 Asia/Taipei 時區常數。
_ASIA_TAIPEI = ZoneInfo("Asia/Taipei")


def _to_aware(dt: datetime) -> datetime:
    """若 datetime 是 naive，視為 Asia/Taipei；已帶 tz 則直接回傳。

    `within_7d` 的計算以絕對時間差 (7 * 24h) 為準，所以 tz 只影響比較的
    一致性：naive 與 aware 混用會在 Python 裡丟例外，此 helper 統一格式。
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ASIA_TAIPEI)
    return dt


# ---------------------------------------------------------------------------
# §5.4 within_7d
# ---------------------------------------------------------------------------

def within_7d(published_at: datetime, query_time: datetime) -> bool:
    """§5.4 — `published_at >= query_time - 7*24h`，Asia/Taipei calendar time。

    使用絕對時間差，不是日曆天切點。
    """
    published_aware = _to_aware(published_at)
    query_aware = _to_aware(query_time)
    return published_aware >= query_aware - timedelta(hours=7 * 24)


# ---------------------------------------------------------------------------
# §5.2 cross_validated
# ---------------------------------------------------------------------------

def cross_validated(evidence: list[Evidence]) -> bool:
    """§5.2 — v1.1 定義：存在至少 2 筆**不同 URL、來源組織不同**的 evidence。

    v1.1 以 `url` 與 `source_name` 去重；不嘗試辨識改寫稿或事件群組（此屬 v1.2+）。
    若 evidence < 2，結構上不可能交叉驗證，回 False。
    """
    if len(evidence) < 2:
        return False

    seen_keys: set[tuple[str, str]] = set()
    for item in evidence:
        seen_keys.add((item.url, item.source_name))
        if len(seen_keys) >= 2:
            # 只要看到兩筆不同 (url, source_name) 組合即判為交叉驗證通過。
            # 注意：同 source_name 不同 url 不算交叉驗證（組織相同）；
            # 此邏輯由下一步 distinct source_name 判斷補強。
            break

    distinct_urls = {item.url for item in evidence}
    distinct_sources = {item.source_name for item in evidence}
    return len(distinct_urls) >= 2 and len(distinct_sources) >= 2


# ---------------------------------------------------------------------------
# §5.3 topic_mismatch
# ---------------------------------------------------------------------------

# v1.1 heuristic：依 source_type 字串前綴判別文件類型。
# 若未來 Evidence 直接攜帶 topic label，可替換此 heuristic。
_ANNOUNCEMENT_SOURCE_TYPES: frozenset[str] = frozenset(
    {"official_disclosure", "announcement", "company_disclosure", "regulatory_filing"}
)
_NEWS_SOURCE_TYPES: frozenset[str] = frozenset(
    {"news_article", "news", "media_report"}
)


def topic_mismatch(
    query_topic: Topic,
    evidence: list[Evidence],
    evidence_source_types: list[str] | None = None,
) -> bool:
    """§5.3 — 低於 50% evidence 符合 query topic 時視為 mismatch。

    參數：
        query_topic: digest query 指定的 topic。
        evidence: 召回後的 evidence 列表。
        evidence_source_types: 可選；與 evidence 一對一對應的 source_type。
            若未提供，v1.1 fallback 為「永遠視為 COMPOSITE 可接受」，
            保守地不觸發 mismatch。這是明確的 v1.1 限制，governance layer
            應盡量傳入 source_types。

    Topic 對應（§5.3）：
        NEWS: 應主要來自新聞類文件
        ANNOUNCEMENT: 應主要來自公告 / 官方揭露
        COMPOSITE: 兩者皆可，永遠不觸發 mismatch
    """
    if query_topic is Topic.COMPOSITE:
        return False
    if not evidence:
        return False  # empty evidence 的情境由 R5 NO_EVIDENCE 處理，非 topic_mismatch 職責

    if evidence_source_types is None:
        # v1.1 限制：缺 source_type metadata 時保守不觸發 mismatch，
        # 避免誤拒。這個 fallback 應在 v1.2 governance 補完 metadata 後移除。
        return False

    if len(evidence_source_types) != len(evidence):
        raise ValueError(
            "evidence_source_types length must match evidence length; "
            f"got {len(evidence_source_types)} vs {len(evidence)}."
        )

    if query_topic is Topic.NEWS:
        expected = _NEWS_SOURCE_TYPES
    elif query_topic is Topic.ANNOUNCEMENT:
        expected = _ANNOUNCEMENT_SOURCE_TYPES
    else:  # pragma: no cover — enum 已限制
        return False

    matched = sum(1 for st in evidence_source_types if st in expected)
    ratio = matched / len(evidence_source_types)
    return ratio < 0.5


# ---------------------------------------------------------------------------
# §5.7 freshness_strong
# ---------------------------------------------------------------------------

def freshness_strong(
    evidence: list[Evidence],
    query_time: datetime,
    topic: Topic,
) -> bool:
    """§5.7 — strong freshness 判斷。

    必須同時滿足：
    1. 所有 evidence 在 7 天內。
    2. 依 topic 的 recency 條件：
        - ANNOUNCEMENT: 至少 1 筆 HIGH source 在近 72h 內
        - NEWS / COMPOSITE: 至少 1 筆 evidence 在近 48h 內
    """
    if not evidence:
        # 空 evidence 由 R5 處理；此處保守回 False。
        return False

    query_aware = _to_aware(query_time)

    # Condition 1
    for item in evidence:
        if not within_7d(item.published_at, query_aware):
            return False

    # Condition 2
    if topic is Topic.ANNOUNCEMENT:
        cutoff = query_aware - timedelta(hours=72)
        return any(
            item.source_tier is SourceTier.HIGH
            and _to_aware(item.published_at) >= cutoff
            for item in evidence
        )

    # NEWS / COMPOSITE
    cutoff = query_aware - timedelta(hours=48)
    return any(_to_aware(item.published_at) >= cutoff for item in evidence)


# ---------------------------------------------------------------------------
# §5.8 evidence_conflict — v1.1 DORMANT
# ---------------------------------------------------------------------------

def evidence_conflict(governance_report: GovernanceReport) -> bool:
    """§5.8 — v1.1 **dormant**；固定回 False。

    啟用條件見 §18.6：governance layer 具備 ExtractedFact 抽取能力後，
    本函式改為依 `governance_report.extracted_facts` 的確定性比較。
    改動**不需要**修改 refusal spec 契約。
    """
    # v1.1: dormant, see §5.8 and §18.5
    _ = governance_report  # silence linters
    return False


# ---------------------------------------------------------------------------
# §5.9 classifier_tag_coverage
# ---------------------------------------------------------------------------

def compute_classifier_tag_coverage(
    topic: Topic,
    predicted_tags: set[TopicTag] | frozenset[TopicTag],
) -> str:
    """§5.9 — deterministic 規則，回傳 "sufficient" | "partial" | "insufficient"。

    - `required_tags` 取自 `MINIMUM_TAG_SET_BY_TOPIC[topic]`
    - `coverage_ratio = |predicted ∩ required| / |required|`
    - 空 required_tags → "sufficient"（v1.1 所有 topic 皆如此；R9/D5 因而 dormant）
    """
    required_tags = MINIMUM_TAG_SET_BY_TOPIC.get(topic, frozenset())
    if not required_tags:
        return "sufficient"

    matched = predicted_tags & required_tags
    coverage_ratio = len(matched) / len(required_tags)

    if coverage_ratio == 1.0:
        return "sufficient"
    if coverage_ratio >= 0.5:
        return "partial"
    return "insufficient"


# ---------------------------------------------------------------------------
# §6.2 hits_out_of_scope_keywords
# ---------------------------------------------------------------------------

def hits_out_of_scope_keywords(user_query: str) -> str | None:
    """§6.2 — 若 user_query 命中 `OUT_OF_SCOPE_KEYWORDS`，回傳對應 scope code。

    用於 Checkpoint 1 R4b 的偵測。回傳：
        - 命中時：第一個命中 keyword 的 scope code
        - 未命中：None

    比對為子字串 case-sensitive match（關鍵字表已以大小寫敏感方式維護，
    例如 RSI / MACD 保留大寫，「比較」保留中文）。
    """
    if not user_query:
        return None
    for keyword, scope_code in OUT_OF_SCOPE_KEYWORDS.items():
        if keyword in user_query:
            return scope_code
    return None


# ---------------------------------------------------------------------------
# §6.3 apply_keyword_tag_fallback — 輔助函式
# ---------------------------------------------------------------------------

def apply_keyword_tag_fallback(user_query: str) -> tuple[TopicTag, ...]:
    """§6.3 — 從 user_query 挑出命中 ALLOWED_KEYWORD_TAGS 的 TopicTag。

    回傳 tuple，保留輸入順序，不重複。digest input layer 可以使用此函式
    補保守 TopicTag；不得擴大為完整 intent / question_type 推導。
    """
    if not user_query:
        return ()
    seen: set[TopicTag] = set()
    ordered: list[TopicTag] = []
    for keyword, tag in ALLOWED_KEYWORD_TAGS.items():
        if keyword in user_query and tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# helper for governance_decision
# ---------------------------------------------------------------------------

def high_trust_count(evidence: list[Evidence]) -> int:
    """回傳 evidence 中 source_tier == HIGH 的筆數。"""
    return sum(1 for item in evidence if item.source_tier is SourceTier.HIGH)


def all_stale(evidence: list[Evidence], query_time: datetime) -> bool:
    """§10 R6 — 是否所有 evidence 皆落在 7 天時間窗外。"""
    if not evidence:
        return False
    return all(not within_7d(item.published_at, query_time) for item in evidence)


__all__ = [
    "all_stale",
    "apply_keyword_tag_fallback",
    "compute_classifier_tag_coverage",
    "cross_validated",
    "evidence_conflict",
    "freshness_strong",
    "high_trust_count",
    "hits_out_of_scope_keywords",
    "topic_mismatch",
    "within_7d",
]
