"""Digest path 封閉表（§5.9.1、§6.2、§6.3）。

此檔三張 map 的共同特徵：

1. **封閉**：值的集合固定，新增需要 PR review。
2. **有硬上限**：避免任何一張表慢慢長成新版的
   `QUESTION_TYPE_FALLBACK_TOPIC_TAGS`。
3. **不含 legacy 語意**：不允許 `question_type` 相關 key。

CI 測試應強制以下不等式（見 tests/digest/policy/test_closed_tables.py）：

- `len(OUT_OF_SCOPE_KEYWORDS) <= 30`
- `len(ALLOWED_KEYWORD_TAGS) <= 10`
- `sum(len(v) for v in MINIMUM_TAG_SET_BY_TOPIC.values()) <= 5`
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ...core.enums import Topic, TopicTag


# ---------------------------------------------------------------------------
# §6.2 OUT_OF_SCOPE_KEYWORDS
#
# 命中任一 keyword 的 query 會在 Checkpoint 1 被 R4b 拒絕。
# v1.1 為封閉清單；長度硬上限 30。
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_KEYWORDS: Mapping[str, str] = {
    # VALUATION
    "估值": "VALUATION",
    "本益比": "VALUATION",
    "合理價": "VALUATION",
    # PRICE_TARGET
    "目標價": "PRICE_TARGET",
    # TECHNICAL
    "技術線": "TECHNICAL",
    "K線": "TECHNICAL",
    "均線": "TECHNICAL",
    "RSI": "TECHNICAL",
    "MACD": "TECHNICAL",
    # FORECAST
    "預測": "FORECAST",
    "會漲嗎": "FORECAST",
    "股價會到": "FORECAST",
    # EARNINGS_DEEP_DIVE
    "EPS": "EARNINGS_DEEP_DIVE",
    "毛利率": "EARNINGS_DEEP_DIVE",
    "現金流": "EARNINGS_DEEP_DIVE",
    # MULTI_STOCK
    "vs": "MULTI_STOCK",
    "比較": "MULTI_STOCK",
    "哪個比較好": "MULTI_STOCK",
}

OUT_OF_SCOPE_KEYWORDS: Mapping[str, str] = MappingProxyType(_OUT_OF_SCOPE_KEYWORDS)
"""§6.2 封閉關鍵字表，讀取專用。"""


# ---------------------------------------------------------------------------
# §6.3 ALLOWED_KEYWORD_TAGS
#
# digest 合法 rule fallback：根據 query 出現的保守關鍵字補 TopicTag。
# 硬上限 10；不允許以此繞道恢復 question_type -> intent 推導。
# ---------------------------------------------------------------------------

_ALLOWED_KEYWORD_TAGS: Mapping[str, TopicTag] = {
    "法說": TopicTag.GUIDANCE,
    "指引": TopicTag.GUIDANCE,
    "AI": TopicTag.AI,
    "電動車": TopicTag.EV,
    "航運": TopicTag.SHIPPING,
    "電價": TopicTag.ELECTRICITY,
    "殖利率": TopicTag.MACRO,
}

ALLOWED_KEYWORD_TAGS: Mapping[str, TopicTag] = MappingProxyType(_ALLOWED_KEYWORD_TAGS)
"""§6.3 封閉 keyword → TopicTag 映射，讀取專用。"""


# ---------------------------------------------------------------------------
# §5.9.1 MINIMUM_TAG_SET_BY_TOPIC
#
# v1.1 所有 topic 皆為空集合，對應 §18.5 把 R9 / D5 標為 Dormant。
# `compute_classifier_tag_coverage()` 在空集合下恆回 "sufficient"。
#
# 未來啟用時機：參見 spec §18.6。
# ---------------------------------------------------------------------------

_MINIMUM_TAG_SET_BY_TOPIC: dict[Topic, frozenset[TopicTag]] = {
    Topic.NEWS: frozenset(),
    Topic.ANNOUNCEMENT: frozenset(),
    Topic.COMPOSITE: frozenset(),
}

MINIMUM_TAG_SET_BY_TOPIC: Mapping[Topic, frozenset[TopicTag]] = MappingProxyType(
    _MINIMUM_TAG_SET_BY_TOPIC
)
"""§5.9.1 封閉 topic → required tags 映射，讀取專用。v1.1 全部為空集合（Dormant）。"""


# ---------------------------------------------------------------------------
# CI gate constants — tests assert these limits.
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_KEYWORDS_LIMIT = 30
ALLOWED_KEYWORD_TAGS_LIMIT = 10
MINIMUM_TAG_SET_BY_TOPIC_LIMIT = 5


__all__ = [
    "ALLOWED_KEYWORD_TAGS",
    "ALLOWED_KEYWORD_TAGS_LIMIT",
    "MINIMUM_TAG_SET_BY_TOPIC",
    "MINIMUM_TAG_SET_BY_TOPIC_LIMIT",
    "OUT_OF_SCOPE_KEYWORDS",
    "OUT_OF_SCOPE_KEYWORDS_LIMIT",
]
