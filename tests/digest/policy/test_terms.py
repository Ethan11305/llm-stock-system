"""Tests for `digest.policy.terms` — §5 術語的 deterministic 實作。

覆蓋：
- within_7d (§5.4)
- cross_validated (§5.2)
- topic_mismatch (§5.3)
- freshness_strong (§5.7)
- evidence_conflict (§5.8) — v1.1 dormant 永遠回 False
- compute_classifier_tag_coverage (§5.9)
- hits_out_of_scope_keywords (§6.2)
- apply_keyword_tag_fallback (§6.3)
- high_trust_count / all_stale 輔助
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from llm_stock_system.core.enums import SourceTier, Topic, TopicTag
from llm_stock_system.core.models import Evidence, GovernanceReport
from llm_stock_system.digest.policy.terms import (
    all_stale,
    apply_keyword_tag_fallback,
    compute_classifier_tag_coverage,
    cross_validated,
    evidence_conflict,
    freshness_strong,
    high_trust_count,
    hits_out_of_scope_keywords,
    topic_mismatch,
    within_7d,
)


TAIPEI = ZoneInfo("Asia/Taipei")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_evidence(
    *,
    published_at: datetime,
    url: str = "https://example.com/a",
    source_name: str = "ExampleNews",
    source_tier: SourceTier = SourceTier.MEDIUM,
    document_id: str = "doc-1",
    title: str = "title",
    excerpt: str = "excerpt",
    support_score: float = 0.8,
    corroboration_count: int = 1,
) -> Evidence:
    return Evidence(
        document_id=document_id,
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=source_tier,
        url=url,
        published_at=published_at,
        support_score=support_score,
        corroboration_count=corroboration_count,
    )


# ---------------------------------------------------------------------------
# §5.4 within_7d
# ---------------------------------------------------------------------------


class TestWithin7d:
    def test_published_exactly_at_boundary(self):
        q = datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)
        published = q - timedelta(hours=7 * 24)
        assert within_7d(published, q) is True

    def test_published_just_outside_boundary(self):
        q = datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)
        published = q - timedelta(hours=7 * 24, seconds=1)
        assert within_7d(published, q) is False

    def test_published_within_window(self):
        q = datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)
        assert within_7d(q - timedelta(hours=48), q) is True

    def test_naive_datetime_treated_as_taipei(self):
        q = datetime(2026, 4, 21, 10, 0)  # naive
        published = datetime(2026, 4, 20, 10, 0)  # naive
        assert within_7d(published, q) is True

    def test_mixed_naive_and_aware(self):
        q = datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)
        published = datetime(2026, 4, 20, 10, 0)  # naive
        assert within_7d(published, q) is True

    def test_cross_timezone_same_instant(self):
        # Same absolute instant, different tz representations.
        q_tpe = datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)
        q_utc = q_tpe.astimezone(timezone.utc)
        published = q_tpe - timedelta(hours=72)
        assert within_7d(published, q_utc) is True


# ---------------------------------------------------------------------------
# §5.2 cross_validated
# ---------------------------------------------------------------------------


class TestCrossValidated:
    def test_empty_is_false(self):
        assert cross_validated([]) is False

    def test_single_is_false(self):
        ev = make_evidence(published_at=datetime(2026, 4, 20, tzinfo=TAIPEI))
        assert cross_validated([ev]) is False

    def test_two_distinct_urls_and_sources_is_true(self):
        a = make_evidence(
            url="https://a.example/1",
            source_name="A",
            published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
        )
        b = make_evidence(
            url="https://b.example/1",
            source_name="B",
            published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
        )
        assert cross_validated([a, b]) is True

    def test_same_source_different_urls_is_false(self):
        a = make_evidence(
            url="https://a.example/1",
            source_name="A",
            published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
        )
        b = make_evidence(
            url="https://a.example/2",
            source_name="A",
            published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
        )
        assert cross_validated([a, b]) is False

    def test_same_url_different_sources_is_false(self):
        a = make_evidence(
            url="https://shared.example/article",
            source_name="A",
            published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
        )
        b = make_evidence(
            url="https://shared.example/article",
            source_name="B",
            published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
        )
        assert cross_validated([a, b]) is False


# ---------------------------------------------------------------------------
# §5.3 topic_mismatch
# ---------------------------------------------------------------------------


class TestTopicMismatch:
    def _make_set(self, n: int = 2) -> list[Evidence]:
        return [
            make_evidence(
                document_id=f"doc-{i}",
                url=f"https://x.example/{i}",
                source_name=f"S{i}",
                published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
            )
            for i in range(n)
        ]

    def test_composite_topic_never_mismatches(self):
        ev = self._make_set(3)
        assert topic_mismatch(Topic.COMPOSITE, ev, ["news_article"] * 3) is False

    def test_empty_evidence_returns_false(self):
        # Empty evidence is handled by R5 NO_EVIDENCE, not topic_mismatch.
        assert topic_mismatch(Topic.NEWS, [], None) is False

    def test_missing_source_types_is_conservative_false(self):
        # v1.1 fallback: without source_types, don't trigger mismatch.
        ev = self._make_set(3)
        assert topic_mismatch(Topic.NEWS, ev, None) is False

    def test_length_mismatch_raises(self):
        ev = self._make_set(2)
        with pytest.raises(ValueError):
            topic_mismatch(Topic.NEWS, ev, ["news_article"])

    def test_news_topic_mostly_news_is_fine(self):
        ev = self._make_set(4)
        types = ["news_article", "news_article", "news_article", "announcement"]
        assert topic_mismatch(Topic.NEWS, ev, types) is False

    def test_news_topic_mostly_announcement_is_mismatch(self):
        ev = self._make_set(4)
        types = [
            "announcement",
            "announcement",
            "announcement",
            "news_article",
        ]
        assert topic_mismatch(Topic.NEWS, ev, types) is True

    def test_announcement_topic_mostly_disclosures_is_fine(self):
        ev = self._make_set(2)
        types = ["official_disclosure", "regulatory_filing"]
        assert topic_mismatch(Topic.ANNOUNCEMENT, ev, types) is False

    def test_announcement_topic_all_news_is_mismatch(self):
        ev = self._make_set(3)
        types = ["news_article", "news_article", "news_article"]
        assert topic_mismatch(Topic.ANNOUNCEMENT, ev, types) is True

    def test_exactly_half_is_not_mismatch(self):
        # ratio < 0.5 triggers mismatch; exactly 0.5 should be fine.
        ev = self._make_set(4)
        types = ["news_article", "news_article", "announcement", "announcement"]
        assert topic_mismatch(Topic.NEWS, ev, types) is False


# ---------------------------------------------------------------------------
# §5.7 freshness_strong
# ---------------------------------------------------------------------------


class TestFreshnessStrong:
    def _base_query_time(self) -> datetime:
        return datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)

    def test_empty_evidence_returns_false(self):
        q = self._base_query_time()
        assert freshness_strong([], q, Topic.NEWS) is False

    def test_any_stale_item_disqualifies(self):
        q = self._base_query_time()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.HIGH,
                published_at=q - timedelta(hours=24),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.MEDIUM,
                published_at=q - timedelta(days=8),  # outside 7d
            ),
        ]
        assert freshness_strong(ev, q, Topic.NEWS) is False

    def test_news_requires_any_48h(self):
        q = self._base_query_time()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.MEDIUM,
                published_at=q - timedelta(hours=72),  # inside 7d but outside 48h
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.MEDIUM,
                published_at=q - timedelta(hours=120),  # inside 7d but outside 48h
            ),
        ]
        assert freshness_strong(ev, q, Topic.NEWS) is False

    def test_news_within_48h_is_strong(self):
        q = self._base_query_time()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.LOW,
                published_at=q - timedelta(hours=24),  # inside 48h
            ),
        ]
        assert freshness_strong(ev, q, Topic.NEWS) is True

    def test_composite_uses_news_rule(self):
        q = self._base_query_time()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.LOW,
                published_at=q - timedelta(hours=12),
            ),
        ]
        assert freshness_strong(ev, q, Topic.COMPOSITE) is True

    def test_announcement_needs_high_tier_within_72h(self):
        q = self._base_query_time()
        # Has an item at 48h but MEDIUM tier — doesn't satisfy ANNOUNCEMENT rule.
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.MEDIUM,
                published_at=q - timedelta(hours=48),
            ),
        ]
        assert freshness_strong(ev, q, Topic.ANNOUNCEMENT) is False

    def test_announcement_high_tier_within_72h_is_strong(self):
        q = self._base_query_time()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="TWSE",
                source_tier=SourceTier.HIGH,
                published_at=q - timedelta(hours=36),
            ),
        ]
        assert freshness_strong(ev, q, Topic.ANNOUNCEMENT) is True

    def test_announcement_high_tier_outside_72h_is_weak(self):
        q = self._base_query_time()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="TWSE",
                source_tier=SourceTier.HIGH,
                published_at=q - timedelta(hours=96),  # inside 7d, outside 72h
            ),
        ]
        assert freshness_strong(ev, q, Topic.ANNOUNCEMENT) is False


# ---------------------------------------------------------------------------
# §5.8 evidence_conflict — v1.1 dormant
# ---------------------------------------------------------------------------


class TestEvidenceConflict:
    def test_always_false_regardless_of_input(self):
        report = GovernanceReport()
        assert evidence_conflict(report) is False

    def test_false_even_with_rich_governance_report(self):
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                published_at=datetime(2026, 4, 20, tzinfo=TAIPEI),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        assert evidence_conflict(report) is False


# ---------------------------------------------------------------------------
# §5.9 compute_classifier_tag_coverage
# ---------------------------------------------------------------------------


class TestComputeClassifierTagCoverage:
    def test_empty_required_returns_sufficient(self):
        # v1.1 path: MINIMUM_TAG_SET_BY_TOPIC all empty → always sufficient.
        for topic in (Topic.NEWS, Topic.ANNOUNCEMENT, Topic.COMPOSITE):
            assert (
                compute_classifier_tag_coverage(topic, frozenset()) == "sufficient"
            )

    def test_empty_required_is_sufficient_even_with_predicted_tags(self):
        assert (
            compute_classifier_tag_coverage(Topic.NEWS, frozenset({TopicTag.AI}))
            == "sufficient"
        )


# ---------------------------------------------------------------------------
# §6.2 hits_out_of_scope_keywords
# ---------------------------------------------------------------------------


class TestHitsOutOfScopeKeywords:
    def test_empty_string_returns_none(self):
        assert hits_out_of_scope_keywords("") is None

    def test_no_match_returns_none(self):
        assert hits_out_of_scope_keywords("台積電今天有什麼新聞？") is None

    def test_hits_valuation(self):
        assert hits_out_of_scope_keywords("台積電估值多少") == "VALUATION"

    def test_hits_target_price(self):
        assert hits_out_of_scope_keywords("給我目標價") == "PRICE_TARGET"

    def test_hits_technical_case_sensitive_rsi(self):
        assert hits_out_of_scope_keywords("最近 RSI 過熱") == "TECHNICAL"

    def test_technical_lowercase_rsi_does_not_hit(self):
        # Case-sensitive per spec §6.2
        assert hits_out_of_scope_keywords("最近 rsi 過熱") is None

    def test_hits_forecast(self):
        assert hits_out_of_scope_keywords("會漲嗎？") == "FORECAST"

    def test_hits_multi_stock_vs(self):
        assert hits_out_of_scope_keywords("2330 vs 2454") == "MULTI_STOCK"

    def test_hits_multi_stock_chinese(self):
        assert hits_out_of_scope_keywords("兩家比較") == "MULTI_STOCK"


# ---------------------------------------------------------------------------
# §6.3 apply_keyword_tag_fallback
# ---------------------------------------------------------------------------


class TestApplyKeywordTagFallback:
    def test_empty_query_returns_empty_tuple(self):
        assert apply_keyword_tag_fallback("") == ()

    def test_no_match_returns_empty_tuple(self):
        assert apply_keyword_tag_fallback("完全不相關的查詢") == ()

    def test_single_hit(self):
        tags = apply_keyword_tag_fallback("今年的 AI 題材")
        assert TopicTag.AI in tags

    def test_multiple_hits_deduplicated(self):
        # 法說 and 指引 both map to GUIDANCE — should appear once.
        tags = apply_keyword_tag_fallback("法說和指引")
        assert tags.count(TopicTag.GUIDANCE) == 1


# ---------------------------------------------------------------------------
# high_trust_count / all_stale helpers
# ---------------------------------------------------------------------------


class TestHighTrustCount:
    def test_empty(self):
        assert high_trust_count([]) == 0

    def test_mixed_tiers(self):
        q = datetime(2026, 4, 20, tzinfo=TAIPEI)
        ev = [
            make_evidence(url="https://a/1", source_name="A", source_tier=SourceTier.HIGH, published_at=q),
            make_evidence(url="https://b/1", source_name="B", source_tier=SourceTier.MEDIUM, published_at=q),
            make_evidence(url="https://c/1", source_name="C", source_tier=SourceTier.HIGH, published_at=q),
            make_evidence(url="https://d/1", source_name="D", source_tier=SourceTier.LOW, published_at=q),
        ]
        assert high_trust_count(ev) == 2


class TestAllStale:
    def _q(self) -> datetime:
        return datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)

    def test_empty_returns_false(self):
        # empty is R5's job, not all_stale.
        assert all_stale([], self._q()) is False

    def test_all_inside_window_returns_false(self):
        q = self._q()
        ev = [
            make_evidence(url="https://a/1", source_name="A", published_at=q - timedelta(hours=24)),
            make_evidence(url="https://b/1", source_name="B", published_at=q - timedelta(hours=72)),
        ]
        assert all_stale(ev, q) is False

    def test_all_outside_window_returns_true(self):
        q = self._q()
        ev = [
            make_evidence(url="https://a/1", source_name="A", published_at=q - timedelta(days=8)),
            make_evidence(url="https://b/1", source_name="B", published_at=q - timedelta(days=10)),
        ]
        assert all_stale(ev, q) is True

    def test_mixed_returns_false(self):
        q = self._q()
        ev = [
            make_evidence(url="https://a/1", source_name="A", published_at=q - timedelta(days=1)),
            make_evidence(url="https://b/1", source_name="B", published_at=q - timedelta(days=9)),
        ]
        assert all_stale(ev, q) is False
