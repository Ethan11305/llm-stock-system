"""Tests for `digest.policy.refusal_policy` — §8〜§12 三階段 checkpoint。

覆蓋：
- Checkpoint 1 (§9) — R1 / R2 / R4 / R4b
- Checkpoint 2 (§10) — R5 / R6
- Checkpoint 3 (§11, §12) — R7 / R8 / D0 / D1 / D2 / D3 / NORMAL
- Dormant 規則 (R9 / D4 / D5) 在 v1.1 條件下不觸發
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from llm_stock_system.core.enums import (
    Intent,
    QueryProfile,
    SourceTier,
    Topic,
    TopicTag,
)
from llm_stock_system.core.models import Document, Evidence, GovernanceReport
from llm_stock_system.digest.policy.enums import (
    DORMANT_DEGRADED_REASONS,
    DORMANT_REFUSAL_REASONS,
    DegradedReason,
    Outcome,
    RefusalCategory,
    RefusalReason,
)
from llm_stock_system.digest.policy.models import (
    ClassifierResult,
    DigestQuery,
    PolicyDecision,
)
from llm_stock_system.digest.policy.refusal_policy import (
    early_refusal,
    governance_decision,
    retrieval_refusal,
)


TAIPEI = ZoneInfo("Asia/Taipei")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_query(
    *,
    user_query: str = "台積電這週有什麼新聞",
    ticker: str | None = "2330",
    topic: Topic = Topic.NEWS,
    time_range_label: str = "7d",
    time_range_explicit: bool = False,
) -> DigestQuery:
    return DigestQuery(
        user_query=user_query,
        ticker=ticker,
        topic=topic,
        time_range_label=time_range_label,
        time_range_explicit=time_range_explicit,
    )


def make_classifier(
    *,
    status: str = "ok",
    tag_coverage: str = "sufficient",
) -> ClassifierResult:
    return ClassifierResult(status=status, tag_coverage=tag_coverage)


def make_document(
    *,
    published_at: datetime,
    ticker: str = "2330",
    title: str = "t",
    content: str = "c",
    source_name: str = "S",
    source_type: str = "news_article",
    source_tier: SourceTier = SourceTier.MEDIUM,
    url: str = "https://example.com/1",
) -> Document:
    return Document(
        ticker=ticker,
        title=title,
        content=content,
        source_name=source_name,
        source_type=source_type,
        source_tier=source_tier,
        url=url,
        published_at=published_at,
    )


def make_evidence(
    *,
    published_at: datetime,
    url: str = "https://example.com/a",
    source_name: str = "ExampleNews",
    source_tier: SourceTier = SourceTier.MEDIUM,
    document_id: str | None = None,
    title: str = "title",
    excerpt: str = "excerpt",
    support_score: float = 0.8,
    corroboration_count: int = 1,
) -> Evidence:
    return Evidence(
        document_id=document_id or f"doc-{url}",
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=source_tier,
        url=url,
        published_at=published_at,
        support_score=support_score,
        corroboration_count=corroboration_count,
    )


QUERY_TIME = datetime(2026, 4, 21, 10, 0, tzinfo=TAIPEI)


# ===========================================================================
# Checkpoint 1 — early_refusal (§9)
# ===========================================================================


class TestCheckpoint1EarlyRefusal:
    # --- R1 UNRESOLVED_TICKER ------------------------------------------------

    def test_r1_unresolved_ticker(self):
        q = make_query(ticker=None)
        decision = early_refusal(q, make_classifier())
        assert decision is not None
        assert decision.outcome is Outcome.REFUSE
        assert decision.refusal_category is RefusalCategory.PARSE
        assert decision.refusal_reason is RefusalReason.UNRESOLVED_TICKER

    def test_r1_takes_priority_over_r2(self):
        # Unknown classifier AND missing ticker → R1 wins.
        q = make_query(ticker=None)
        decision = early_refusal(q, make_classifier(status="unknown"))
        assert decision is not None
        assert decision.refusal_reason is RefusalReason.UNRESOLVED_TICKER

    # --- R2 CLASSIFIER_UNKNOWN ----------------------------------------------

    def test_r2_classifier_unknown(self):
        q = make_query()
        decision = early_refusal(q, make_classifier(status="unknown"))
        assert decision is not None
        assert decision.refusal_category is RefusalCategory.CLASSIFICATION
        assert decision.refusal_reason is RefusalReason.CLASSIFIER_UNKNOWN

    # --- R4 EXPLICIT_OUT_OF_SCOPE_TIME_RANGE --------------------------------

    def test_r4_explicit_1m_is_refused(self):
        q = make_query(time_range_explicit=True, time_range_label="1m")
        decision = early_refusal(q, make_classifier())
        assert decision is not None
        assert decision.refusal_category is RefusalCategory.CLASSIFICATION
        assert (
            decision.refusal_reason is RefusalReason.EXPLICIT_OUT_OF_SCOPE_TIME_RANGE
        )

    def test_r4_explicit_7d_passes(self):
        q = make_query(time_range_explicit=True, time_range_label="7d")
        decision = early_refusal(q, make_classifier())
        assert decision is None

    def test_r4_default_7d_without_explicit_flag_passes(self):
        q = make_query(time_range_explicit=False, time_range_label="7d")
        decision = early_refusal(q, make_classifier())
        assert decision is None

    # --- R4b OUT_OF_SCOPE_QUERY_KIND ----------------------------------------

    def test_r4b_valuation_keyword(self):
        q = make_query(user_query="台積電估值怎麼看")
        decision = early_refusal(q, make_classifier())
        assert decision is not None
        assert decision.refusal_reason is RefusalReason.OUT_OF_SCOPE_QUERY_KIND

    def test_r4b_target_price_keyword(self):
        q = make_query(user_query="目標價多少")
        decision = early_refusal(q, make_classifier())
        assert decision is not None
        assert decision.refusal_reason is RefusalReason.OUT_OF_SCOPE_QUERY_KIND

    def test_r4b_multi_stock_vs(self):
        q = make_query(user_query="2330 vs 2454")
        decision = early_refusal(q, make_classifier())
        assert decision is not None
        assert decision.refusal_reason is RefusalReason.OUT_OF_SCOPE_QUERY_KIND

    def test_happy_path_returns_none(self):
        q = make_query(user_query="台積電這週有什麼新聞")
        assert early_refusal(q, make_classifier()) is None


# ===========================================================================
# Checkpoint 2 — retrieval_refusal (§10)
# ===========================================================================


class TestCheckpoint2RetrievalRefusal:
    # --- R5 NO_EVIDENCE -----------------------------------------------------

    def test_r5_empty_documents(self):
        q = make_query()
        decision = retrieval_refusal(q, [], QUERY_TIME)
        assert decision is not None
        assert decision.outcome is Outcome.REFUSE
        assert decision.refusal_category is RefusalCategory.EVIDENCE
        assert decision.refusal_reason is RefusalReason.NO_EVIDENCE

    # --- R6 STALE_EVIDENCE_ONLY ---------------------------------------------

    def test_r6_all_documents_outside_7d(self):
        q = make_query()
        docs = [
            make_document(published_at=QUERY_TIME - timedelta(days=8)),
            make_document(
                url="https://example.com/2",
                published_at=QUERY_TIME - timedelta(days=9),
            ),
        ]
        decision = retrieval_refusal(q, docs, QUERY_TIME)
        assert decision is not None
        assert decision.refusal_reason is RefusalReason.STALE_EVIDENCE_ONLY

    def test_r6_at_least_one_inside_7d_passes(self):
        q = make_query()
        docs = [
            make_document(published_at=QUERY_TIME - timedelta(days=8)),
            make_document(
                url="https://example.com/2",
                published_at=QUERY_TIME - timedelta(days=1),
            ),
        ]
        assert retrieval_refusal(q, docs, QUERY_TIME) is None

    def test_r5_wins_over_r6_when_empty(self):
        # Empty doc list hits R5 before R6 logic would evaluate.
        q = make_query()
        decision = retrieval_refusal(q, [], QUERY_TIME)
        assert decision is not None
        assert decision.refusal_reason is RefusalReason.NO_EVIDENCE


# ===========================================================================
# Checkpoint 3 — governance_decision (§11 + §12)
# ===========================================================================


class TestCheckpoint3Refusals:
    # --- R7 LOW_TRUST_SINGLE_SOURCE -----------------------------------------

    def test_r7_single_low_trust_evidence(self):
        q = make_query()
        ev = [
            make_evidence(
                published_at=QUERY_TIME - timedelta(hours=24),
                source_tier=SourceTier.LOW,
            )
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is Outcome.REFUSE
        assert decision.refusal_category is RefusalCategory.GOVERNANCE
        assert decision.refusal_reason is RefusalReason.LOW_TRUST_SINGLE_SOURCE

    def test_r7_single_medium_tier_also_refused(self):
        q = make_query()
        ev = [
            make_evidence(
                published_at=QUERY_TIME - timedelta(hours=12),
                source_tier=SourceTier.MEDIUM,
            )
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.refusal_reason is RefusalReason.LOW_TRUST_SINGLE_SOURCE

    # --- R8 TOPIC_MISMATCH --------------------------------------------------

    def test_r8_topic_mismatch_with_provided_source_types(self):
        q = make_query(topic=Topic.NEWS)
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(
            q, report, make_classifier(), QUERY_TIME,
            evidence_source_types=["announcement", "regulatory_filing"],
        )
        assert decision.outcome is Outcome.REFUSE
        assert decision.refusal_reason is RefusalReason.TOPIC_MISMATCH

    def test_r8_not_triggered_without_source_types_in_v1_1(self):
        # v1.1 fallback: missing source_types → don't refuse.
        q = make_query(topic=Topic.NEWS)
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is not Outcome.REFUSE or (
            decision.refusal_reason is not RefusalReason.TOPIC_MISMATCH
        )


class TestCheckpoint3Degraded:
    # --- D0 SINGLE_HIGH_TRUST_SOURCE ----------------------------------------

    def test_d0_single_high_trust_source_early_return(self):
        q = make_query()
        ev = [
            make_evidence(
                published_at=QUERY_TIME - timedelta(hours=6),
                source_tier=SourceTier.HIGH,
            )
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is Outcome.DEGRADED
        # D0 early return: only SINGLE_HIGH_TRUST_SOURCE, no D1/D2/D3.
        assert decision.degraded_reasons == (DegradedReason.SINGLE_HIGH_TRUST_SOURCE,)

    def test_d0_not_masked_by_weak_cross_validation(self):
        # Regression: prior design let D1 dominate for len(evidence)==1.
        q = make_query()
        ev = [
            make_evidence(
                published_at=QUERY_TIME - timedelta(hours=6),
                source_tier=SourceTier.HIGH,
            )
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert DegradedReason.WEAK_CROSS_VALIDATION not in decision.degraded_reasons
        assert DegradedReason.NO_HIGH_TRUST_SOURCE not in decision.degraded_reasons

    # --- D1 WEAK_CROSS_VALIDATION -------------------------------------------

    def test_d1_weak_cross_validation_same_source(self):
        q = make_query()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=6),
            ),
            make_evidence(
                url="https://a/2",
                source_name="A",  # same source name → not cross-validated
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is Outcome.DEGRADED
        assert DegradedReason.WEAK_CROSS_VALIDATION in decision.degraded_reasons

    # --- D2 NO_HIGH_TRUST_SOURCE --------------------------------------------

    def test_d2_no_high_trust_source_with_multiple_evidence(self):
        q = make_query()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.MEDIUM,
                published_at=QUERY_TIME - timedelta(hours=6),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.LOW,
                published_at=QUERY_TIME - timedelta(hours=10),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is Outcome.DEGRADED
        assert DegradedReason.NO_HIGH_TRUST_SOURCE in decision.degraded_reasons

    # --- D3 BORDERLINE_FRESHNESS --------------------------------------------

    def test_d3_borderline_freshness_news_beyond_48h(self):
        q = make_query(topic=Topic.NEWS)
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=96),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=120),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is Outcome.DEGRADED
        assert DegradedReason.BORDERLINE_FRESHNESS in decision.degraded_reasons

    def test_d3_borderline_freshness_announcement_without_high_within_72h(self):
        q = make_query(topic=Topic.ANNOUNCEMENT)
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.MEDIUM,  # MEDIUM within 72h doesn't satisfy
                published_at=QUERY_TIME - timedelta(hours=24),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=96),  # HIGH but >72h
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert DegradedReason.BORDERLINE_FRESHNESS in decision.degraded_reasons

    # --- Multi-reason collection --------------------------------------------

    def test_multiple_degraded_reasons_can_coexist(self):
        # len(evidence)>=2, no cross-validation, no high-trust, not fresh → D1+D2+D3.
        q = make_query(topic=Topic.NEWS)
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.MEDIUM,
                published_at=QUERY_TIME - timedelta(hours=96),
            ),
            make_evidence(
                url="https://a/2",
                source_name="A",  # same source
                source_tier=SourceTier.LOW,
                published_at=QUERY_TIME - timedelta(hours=120),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is Outcome.DEGRADED
        assert DegradedReason.WEAK_CROSS_VALIDATION in decision.degraded_reasons
        assert DegradedReason.NO_HIGH_TRUST_SOURCE in decision.degraded_reasons
        assert DegradedReason.BORDERLINE_FRESHNESS in decision.degraded_reasons


class TestCheckpoint3Normal:
    def test_normal_path_high_trust_cross_validated_fresh(self):
        q = make_query(topic=Topic.NEWS)
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="Reuters",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=6),
            ),
            make_evidence(
                url="https://b/1",
                source_name="Bloomberg",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.outcome is Outcome.NORMAL
        assert decision.degraded_reasons == ()
        assert decision.refusal_category is None

    def test_normal_composite_topic(self):
        q = make_query(topic=Topic.COMPOSITE)
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="Reuters",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=6),
            ),
            make_evidence(
                url="https://b/1",
                source_name="TWSE",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(
            q, report, make_classifier(), QUERY_TIME,
            evidence_source_types=["news_article", "official_disclosure"],
        )
        assert decision.outcome is Outcome.NORMAL


# ===========================================================================
# Dormant rules — v1.1 must never trigger
# ===========================================================================


class TestDormantRulesNeverFire:
    """§18.5 — v1.1 Dormant rules 在 v1.1 條件下永不應觸發。"""

    def test_r9_insufficient_tag_coverage_is_dormant(self):
        assert RefusalReason.INSUFFICIENT_TAG_COVERAGE in DORMANT_REFUSAL_REASONS

    def test_d4_evidence_conflict_is_dormant(self):
        assert DegradedReason.EVIDENCE_CONFLICT in DORMANT_DEGRADED_REASONS

    def test_d5_partial_tag_coverage_is_dormant(self):
        assert DegradedReason.PARTIAL_TAG_COVERAGE in DORMANT_DEGRADED_REASONS

    def test_r9_not_triggered_because_tag_coverage_forced_sufficient(self):
        # classifier_tag_coverage is always "sufficient" on DigestQuery default,
        # and under v1.1 MINIMUM_TAG_SET_BY_TOPIC is empty → coverage always sufficient.
        q = make_query()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=6),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert decision.refusal_reason is not RefusalReason.INSUFFICIENT_TAG_COVERAGE

    def test_d4_evidence_conflict_never_appears_in_degraded_reasons(self):
        q = make_query()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.MEDIUM,
                published_at=QUERY_TIME - timedelta(hours=96),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.LOW,
                published_at=QUERY_TIME - timedelta(hours=120),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert DegradedReason.EVIDENCE_CONFLICT not in decision.degraded_reasons

    def test_d5_partial_tag_coverage_never_appears(self):
        q = make_query()
        ev = [
            make_evidence(
                url="https://a/1",
                source_name="A",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=6),
            ),
            make_evidence(
                url="https://b/1",
                source_name="B",
                source_tier=SourceTier.HIGH,
                published_at=QUERY_TIME - timedelta(hours=12),
            ),
        ]
        report = GovernanceReport(evidence=ev)
        decision = governance_decision(q, report, make_classifier(), QUERY_TIME)
        assert DegradedReason.PARTIAL_TAG_COVERAGE not in decision.degraded_reasons


# ===========================================================================
# PolicyDecision invariants (§7)
# ===========================================================================


class TestPolicyDecisionInvariants:
    def test_refuse_has_no_confidence(self):
        d = PolicyDecision.refuse(
            RefusalCategory.PARSE, RefusalReason.UNRESOLVED_TICKER
        )
        assert d.confidence_light is None
        assert d.confidence_score is None
        assert d.degraded_reasons == ()

    def test_degraded_requires_at_least_one_reason(self):
        with pytest.raises(ValueError):
            PolicyDecision.degraded([])

    def test_degraded_with_reason(self):
        d = PolicyDecision.degraded([DegradedReason.SINGLE_HIGH_TRUST_SOURCE])
        assert d.outcome is Outcome.DEGRADED
        assert d.refusal_category is None
        assert d.refusal_reason is None

    def test_normal_has_no_refusal_or_degraded(self):
        d = PolicyDecision.normal()
        assert d.outcome is Outcome.NORMAL
        assert d.refusal_category is None
        assert d.refusal_reason is None
        assert d.degraded_reasons == ()


# ===========================================================================
# DigestQuery validator guards (§6.1)
# ===========================================================================


class TestDigestQueryValidators:
    def test_rejects_non_digest_intent(self):
        with pytest.raises(ValueError):
            DigestQuery(
                user_query="x",
                ticker="2330",
                topic=Topic.NEWS,
                intent=Intent.EARNINGS_REVIEW,
            )

    def test_rejects_earnings_topic(self):
        with pytest.raises(ValueError):
            DigestQuery(
                user_query="x",
                ticker="2330",
                topic=Topic.EARNINGS,
            )

    def test_rejects_legacy_query_profile(self):
        with pytest.raises(ValueError):
            DigestQuery(
                user_query="x",
                ticker="2330",
                topic=Topic.NEWS,
                query_profile=QueryProfile.LEGACY,
            )

    def test_accepts_valid_digest_query(self):
        q = DigestQuery(user_query="x", ticker="2330", topic=Topic.NEWS)
        assert q.intent is Intent.NEWS_DIGEST
        assert q.query_profile is QueryProfile.SINGLE_STOCK_DIGEST
