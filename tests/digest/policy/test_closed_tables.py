"""Tests for `digest.policy.closed_tables` — §5.9.1、§6.2、§6.3 封閉表 CI 閘門。

這些是結構性測試，確保：

1. 三張 map 皆為 read-only（`MappingProxyType`），無法運行時被污染。
2. 三張 map 的大小都在 spec §18 硬上限之內。
3. v1.1 MINIMUM_TAG_SET_BY_TOPIC 維持全空集合（R9 / D5 dormant）。
4. OUT_OF_SCOPE_KEYWORDS 的 scope code 屬於封閉集合。
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from llm_stock_system.core.enums import Topic, TopicTag
from llm_stock_system.digest.policy.closed_tables import (
    ALLOWED_KEYWORD_TAGS,
    ALLOWED_KEYWORD_TAGS_LIMIT,
    MINIMUM_TAG_SET_BY_TOPIC,
    MINIMUM_TAG_SET_BY_TOPIC_LIMIT,
    OUT_OF_SCOPE_KEYWORDS,
    OUT_OF_SCOPE_KEYWORDS_LIMIT,
)


# ---------------------------------------------------------------------------
# Read-only invariants
# ---------------------------------------------------------------------------


def test_out_of_scope_keywords_is_mapping_proxy():
    assert isinstance(OUT_OF_SCOPE_KEYWORDS, MappingProxyType)


def test_allowed_keyword_tags_is_mapping_proxy():
    assert isinstance(ALLOWED_KEYWORD_TAGS, MappingProxyType)


def test_minimum_tag_set_by_topic_is_mapping_proxy():
    assert isinstance(MINIMUM_TAG_SET_BY_TOPIC, MappingProxyType)


def test_out_of_scope_keywords_cannot_be_mutated_at_runtime():
    with pytest.raises(TypeError):
        OUT_OF_SCOPE_KEYWORDS["新關鍵字"] = "X"  # type: ignore[index]


def test_allowed_keyword_tags_cannot_be_mutated_at_runtime():
    with pytest.raises(TypeError):
        ALLOWED_KEYWORD_TAGS["foo"] = TopicTag.AI  # type: ignore[index]


def test_minimum_tag_set_cannot_be_mutated_at_runtime():
    with pytest.raises(TypeError):
        MINIMUM_TAG_SET_BY_TOPIC[Topic.NEWS] = frozenset()  # type: ignore[index]


# ---------------------------------------------------------------------------
# Size / CI gates — §18 硬上限
# ---------------------------------------------------------------------------


def test_out_of_scope_keywords_hard_limit():
    assert len(OUT_OF_SCOPE_KEYWORDS) <= OUT_OF_SCOPE_KEYWORDS_LIMIT
    assert OUT_OF_SCOPE_KEYWORDS_LIMIT == 30


def test_allowed_keyword_tags_hard_limit():
    assert len(ALLOWED_KEYWORD_TAGS) <= ALLOWED_KEYWORD_TAGS_LIMIT
    assert ALLOWED_KEYWORD_TAGS_LIMIT == 10


def test_minimum_tag_set_total_size_limit():
    total = sum(len(v) for v in MINIMUM_TAG_SET_BY_TOPIC.values())
    assert total <= MINIMUM_TAG_SET_BY_TOPIC_LIMIT
    assert MINIMUM_TAG_SET_BY_TOPIC_LIMIT == 5


# ---------------------------------------------------------------------------
# §5.9.1 — v1.1 Dormant: all empty frozensets
# ---------------------------------------------------------------------------


def test_minimum_tag_set_by_topic_all_empty_in_v1_1():
    """v1.1 所有 topic 的 minimum tag set 必為空集合，保證 R9 / D5 永不觸發。"""
    for topic in (Topic.NEWS, Topic.ANNOUNCEMENT, Topic.COMPOSITE):
        assert topic in MINIMUM_TAG_SET_BY_TOPIC
        assert MINIMUM_TAG_SET_BY_TOPIC[topic] == frozenset()


def test_minimum_tag_set_covers_all_digest_topics():
    """封閉集合必須涵蓋 digest 合法的三個 Topic。"""
    expected = {Topic.NEWS, Topic.ANNOUNCEMENT, Topic.COMPOSITE}
    assert expected.issubset(set(MINIMUM_TAG_SET_BY_TOPIC.keys()))


# ---------------------------------------------------------------------------
# §6.2 — OUT_OF_SCOPE_KEYWORDS scope 封閉
# ---------------------------------------------------------------------------


def test_out_of_scope_keyword_scope_codes_are_closed_set():
    """§6.2 scope code 目前封閉為 6 類；新增需要 PR review。"""
    allowed = {
        "VALUATION",
        "PRICE_TARGET",
        "TECHNICAL",
        "FORECAST",
        "EARNINGS_DEEP_DIVE",
        "MULTI_STOCK",
    }
    observed = set(OUT_OF_SCOPE_KEYWORDS.values())
    assert observed.issubset(allowed)


def test_out_of_scope_keywords_nonempty():
    assert len(OUT_OF_SCOPE_KEYWORDS) > 0


# ---------------------------------------------------------------------------
# §6.3 — ALLOWED_KEYWORD_TAGS 值域限定在 TopicTag
# ---------------------------------------------------------------------------


def test_allowed_keyword_tags_values_are_topic_tag():
    for keyword, tag in ALLOWED_KEYWORD_TAGS.items():
        assert isinstance(keyword, str)
        assert isinstance(tag, TopicTag)


def test_allowed_keyword_tags_does_not_include_legacy_question_type_key():
    """禁止以 question_type 相關字樣作 key，避免回歸 legacy fallback。"""
    banned_substrings = ("question_type", "intent=", "QT_")
    for keyword in ALLOWED_KEYWORD_TAGS.keys():
        for banned in banned_substrings:
            assert banned not in keyword
