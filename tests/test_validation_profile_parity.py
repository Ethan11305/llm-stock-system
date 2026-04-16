"""Parity tests: _evaluate_profile() vs legacy _apply_*_rules() methods.

Design
------
* ``run_parity`` is a module-level helper (not a base class) — lighter,
  no shared state.
* **Primary assert: score equality.**  If both paths produce the same
  float, the downstream GREEN/YELLOW/RED bucket is guaranteed identical.
* **Warning parity is opt-in** (``check_warnings=True``).  Some Group A
  profiles deliberately split a single compound legacy warning into
  per-condition warnings.  The divergence table is documented per test.

Warning-parity table
--------------------
Profile                              check_warnings  Reason
------------------------------------  -------------  ------------------------
A1  ex_dividend_performance           False          1 combined → 2 per-signal
A2  technical_indicator_review        True           identical single warning
A3  monthly_revenue_yoy_review        True           identical two-check texts
A4  pe_valuation_review               True           identical single warning
A5  profitability_stability_review    False          OR→split + different text
A6  margin_turnaround_review          False          compound OR → 4 separate
A7  fundamental_pe_review             True           custom_validator mirrors
A8  investment_support                True           same custom_validator
B1  season_line_margin_review         True           exact three-tier match
B2  fcf_dividend_sustainability       True           exact three-tier match
B3  debt_dividend_safety_review       True           exact three-tier match
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import AnswerDraft, Evidence, GovernanceReport, SourceCitation, StructuredQuery
from llm_stock_system.core.validation_profiles import ValidationProfile, get_profile
from llm_stock_system.layers.validation_layer import ValidationLayer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_TS = datetime(2026, 4, 16, tzinfo=_UTC)


def _ev(source_name: str, title: str = "", excerpt: str = "") -> Evidence:
    return Evidence(
        document_id=f"{source_name}|{title}",
        title=title,
        excerpt=excerpt,
        source_name=source_name,
        source_tier=SourceTier.HIGH,
        url="https://example.com",
        published_at=_TS,
        support_score=1.0,
        corroboration_count=1,
    )


def _gov(*evidence: Evidence, **kwargs) -> GovernanceReport:
    return GovernanceReport(
        evidence=list(evidence),
        sufficiency=SufficiencyStatus.SUFFICIENT,
        consistency=ConsistencyStatus.CONSISTENT,
        freshness=FreshnessStatus.RECENT,
        high_trust_ratio=1.0,
        **kwargs,
    )


def _draft(summary: str = "Grounded answer.", evidence: list[Evidence] | None = None) -> AnswerDraft:
    evs = evidence or []
    return AnswerDraft(
        summary=summary,
        highlights=[],
        facts=[],
        impacts=[],
        risks=["r1", "r2", "r3"],
        sources=[
            SourceCitation(
                title=e.title,
                source_name=e.source_name,
                source_tier=e.source_tier,
                url=e.url,
                published_at=e.published_at,
                excerpt=e.excerpt,
                support_score=e.support_score,
                corroboration_count=e.corroboration_count,
            )
            for e in evs
        ],
    )


def _query(question_type: str, **kwargs) -> StructuredQuery:
    defaults: dict = {
        "user_query": f"parity-test {question_type}",
        "ticker": "2330",
        "company_name": "台積電",
        "question_type": question_type,
        "time_range_label": "1y",
        "time_range_days": 365,
    }
    defaults.update(kwargs)
    return StructuredQuery(**defaults)


# ---------------------------------------------------------------------------
# Parity harness
# ---------------------------------------------------------------------------

@dataclass
class ParityResult:
    legacy_score: float
    profile_score: float
    legacy_warnings: list[str]
    profile_warnings: list[str]


def run_parity(
    legacy_fn: Callable[[list[str]], float],
    profile_fn: Callable[[list[str]], float],
) -> ParityResult:
    """Run both paths with independent warning lists and return results."""
    lw: list[str] = []
    pw: list[str] = []
    return ParityResult(legacy_fn(lw), profile_fn(pw), lw, pw)


def assert_parity(
    tc: unittest.TestCase,
    result: ParityResult,
    *,
    check_warnings: bool = False,
    msg: str = "",
) -> None:
    prefix = f"[{msg}] " if msg else ""
    tc.assertAlmostEqual(
        result.legacy_score,
        result.profile_score,
        places=9,
        msg=f"{prefix}Score mismatch: legacy={result.legacy_score} profile={result.profile_score}",
    )
    if check_warnings:
        tc.assertEqual(
            result.legacy_warnings,
            result.profile_warnings,
            msg=f"{prefix}Warning mismatch:\n  legacy={result.legacy_warnings}\n  profile={result.profile_warnings}",
        )


# ---------------------------------------------------------------------------
# Base helpers used by all test cases
# ---------------------------------------------------------------------------

class _ParityBase(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = ValidationLayer(min_green_confidence=0.8, min_yellow_confidence=0.55)

    def _profile(self, question_type: str) -> ValidationProfile:
        p = get_profile(question_type)
        self.assertIsNotNone(p, f"No profile registered for '{question_type}'")
        return p  # type: ignore[return-value]

    def _profile_fn(
        self,
        question_type: str,
        query: StructuredQuery,
        gov: GovernanceReport,
        draft: AnswerDraft,
        initial: float,
    ) -> Callable[[list[str]], float]:
        profile = self._profile(question_type)
        layer = self.layer
        return lambda w: layer._evaluate_profile(profile, query, gov, draft, initial, w)

    def _snames(self, gov: GovernanceReport) -> set[str]:
        return self.layer._source_names(gov)

    def _ctext(self, gov: GovernanceReport) -> str:
        return self.layer._combined_text(gov)


# ---------------------------------------------------------------------------
# Group A — warning-only profiles
# ---------------------------------------------------------------------------

class GroupAParityTestCase(_ParityBase):
    """Group A: score never changes; parity is on score + optionally warnings."""

    # -- A1: ex_dividend_performance ----------------------------------------

    def test_a1_ex_dividend_pass(self) -> None:
        ev = [_ev("FinMind Dividend price data"), _ev("FinMind TaiwanStockPrice")]
        gov = _gov(*ev)
        query = _query("ex_dividend_performance")
        draft = _draft(evidence=ev)
        sn = self._snames(gov)

        result = run_parity(
            lambda w: self.layer._apply_ex_dividend_rules(sn, 1.0, w),
            self._profile_fn("ex_dividend_performance", query, gov, draft, 1.0),
        )
        assert_parity(self, result, check_warnings=True, msg="A1 pass")

    def test_a1_ex_dividend_both_missing(self) -> None:
        ev = [_ev("Generic News Source")]
        gov = _gov(*ev)
        query = _query("ex_dividend_performance")
        sn = self._snames(gov)

        result = run_parity(
            lambda w: self.layer._apply_ex_dividend_rules(sn, 1.0, w),
            self._profile_fn("ex_dividend_performance", query, gov, _draft(), 1.0),
        )
        # score parity only — profile emits 2 warnings, legacy emits 1 combined
        assert_parity(self, result, check_warnings=False, msg="A1 both-missing")
        self.assertEqual(len(result.legacy_warnings), 1)
        self.assertEqual(len(result.profile_warnings), 2)

    # -- A2: technical_indicator_review -------------------------------------

    def test_a2_technical_indicator_parity(self) -> None:
        for has_price, label in ((True, "pass"), (False, "fail")):
            with self.subTest(has_price=has_price):
                evs = [_ev("FinMind TaiwanStockPrice" if has_price else "News Source")]
                gov = _gov(*evs)
                query = _query("technical_indicator_review")
                sn = self._snames(gov)

                result = run_parity(
                    lambda w, s=sn: self.layer._apply_technical_indicator_rules(s, 1.0, w),
                    self._profile_fn("technical_indicator_review", query, gov, _draft(), 1.0),
                )
                assert_parity(self, result, check_warnings=True, msg=f"A2 {label}")

    # -- A3: monthly_revenue_yoy_review -------------------------------------

    def test_a3_monthly_revenue_parity(self) -> None:
        partial_summary = "部分月份資料尚未公告，數據僅供參考。"
        scenarios = [
            (True, False, "pass"),
            (False, False, "fail-twse"),
            (True, True, "fail-partial"),
            (False, True, "fail-both"),
        ]
        for has_twse, partial, label in scenarios:
            with self.subTest(label=label):
                evs = [_ev("TWSE monthly revenue" if has_twse else "Generic Source")]
                gov = _gov(*evs)
                query = _query("monthly_revenue_yoy_review")
                summary = partial_summary if partial else "全年月營收年增 18%，創歷史新高。"
                draft = _draft(summary=summary, evidence=evs)
                sn = self._snames(gov)

                result = run_parity(
                    lambda w, s=sn, d=draft: self.layer._apply_monthly_revenue_rules(s, d, 1.0, w),
                    self._profile_fn("monthly_revenue_yoy_review", query, gov, draft, 1.0),
                )
                assert_parity(self, result, check_warnings=True, msg=f"A3 {label}")

    # -- A4: pe_valuation_review --------------------------------------------

    def test_a4_pe_valuation_parity(self) -> None:
        for has_twse, label in ((True, "pass"), (False, "fail")):
            with self.subTest(label=label):
                evs = [_ev("TWSE valuation data" if has_twse else "Broker note")]
                gov = _gov(*evs)
                query = _query("pe_valuation_review")
                sn = self._snames(gov)

                result = run_parity(
                    lambda w, s=sn: self.layer._apply_pe_valuation_rules(s, 1.0, w),
                    self._profile_fn("pe_valuation_review", query, gov, _draft(), 1.0),
                )
                assert_parity(self, result, check_warnings=True, msg=f"A4 {label}")

    # -- A5: profitability_stability_review ---------------------------------

    def test_a5_profitability_stability_score_parity(self) -> None:
        """Score parity across four scenarios; warnings intentionally differ."""
        fin_source = "FinMind financial_statements data"
        scenarios = [
            # (source_name, evidence_count, has_inference_text, expected_score)
            (fin_source, 4, False, 1.0),            # pass — all ok
            ("Generic News", 4, False, 1.0),         # missing financial source (warning only)
            (fin_source, 2, False, 1.0),             # count < 3 (warning only)
            (fin_source, 4, True, 0.75),             # compound cap fires
        ]
        for src, count, has_infer, expected in scenarios:
            with self.subTest(src=src[:15], count=count, has_infer=has_infer):
                excerpt = "若只看財報結構推估，獲利可能持續。" if has_infer else "EPS 持續成長。"
                evs = [_ev(src, excerpt=excerpt)] * count
                gov = _gov(*evs)
                query = _query("profitability_stability_review")
                sn = self._snames(gov)
                ct = self._ctext(gov)

                result = run_parity(
                    lambda w, g=gov, s=sn, c=ct: self.layer._apply_profitability_stability_rules(g, s, c, 1.0, w),
                    self._profile_fn("profitability_stability_review", query, gov, _draft(), 1.0),
                )
                assert_parity(self, result, check_warnings=False, msg=f"A5 src={src[:8]} n={count}")
                self.assertAlmostEqual(result.legacy_score, expected, places=9, msg="A5 expected score")

    # -- A6: margin_turnaround_review ---------------------------------------

    def test_a6_margin_turnaround_score_parity(self) -> None:
        """Score always 1.0 (warning-only); main check is that both paths agree."""
        fin_src = "FinMind financial_statements data"
        rich_excerpt = "毛利率 回升，營業利益 明顯改善。"
        poor_excerpt = "市場情緒偏謹慎。"

        scenarios = [
            # (source_name, excerpt, count, label)
            (fin_src, rich_excerpt, 3, "pass"),
            ("Generic", rich_excerpt, 3, "no-fin-source"),
            (fin_src, poor_excerpt, 3, "no-margin-kw"),
            (fin_src, rich_excerpt, 2, "low-count"),
        ]
        for src, excerpt, count, label in scenarios:
            with self.subTest(label=label):
                evs = [_ev(src, excerpt=excerpt)] * count
                gov = _gov(*evs)
                query = _query("margin_turnaround_review")
                sn = self._snames(gov)
                ct = self._ctext(gov)

                result = run_parity(
                    lambda w, g=gov, s=sn, c=ct: self.layer._apply_margin_turnaround_rules(g, s, c, 1.0, w),
                    self._profile_fn("margin_turnaround_review", query, gov, _draft(), 1.0),
                )
                # profile fires per-condition warnings; legacy fires one compound warning
                assert_parity(self, result, check_warnings=False, msg=f"A6 {label}")
                self.assertAlmostEqual(result.legacy_score, 1.0, places=9)

    # -- A7 / A8: fundamental_pe_review & investment_support ---------------

    def test_a7_a8_fundamental_valuation_parity(self) -> None:
        """Both share _validate_fundamental_valuation; warning texts match exactly."""
        # Keyword sets (from fundamental_valuation.py):
        #   fundamental: eps / 營收 / 獲利 / 財報 / 基本面 / 體質 / 毛利 / 毛利率 / 現金流
        #   valuation:   本益比 / pe ratio / p/e / 估值 / 歷史分位 / valuation
        scenarios = [
            # (excerpt, question_type, expected_warning_count)
            ("EPS 成長穩健，本益比 18 倍。", "fundamental_pe_review", 0),     # both present
            ("EPS 穩健成長，毛利率提升。", "fundamental_pe_review", 1),        # valuation missing
            ("本益比 18 倍，目前處歷史分位。", "fundamental_pe_review", 1),    # fundamental missing
            ("市場情緒偏中性。", "fundamental_pe_review", 2),                  # both missing
            ("EPS 成長穩健，本益比 18 倍。", "investment_support", 0),
            ("市場情緒偏中性。", "investment_support", 2),
        ]
        for excerpt, qt, expected_warnings in scenarios:
            with self.subTest(qt=qt, excerpt=excerpt[:20]):
                evs = [_ev("Source", excerpt=excerpt)]
                gov = _gov(*evs)
                query = _query(qt)

                result = run_parity(
                    lambda w, g=gov: self.layer._apply_fundamental_valuation_rules(g, 1.0, w),
                    self._profile_fn(qt, query, gov, _draft(), 1.0),
                )
                assert_parity(self, result, check_warnings=True, msg=f"A7/8 {qt}")
                self.assertEqual(len(result.legacy_warnings), expected_warnings)


# ---------------------------------------------------------------------------
# Group B — three-tier cap profiles (DUAL_SIGNAL_MISSING)
# ---------------------------------------------------------------------------

class GroupBParityTestCase(_ParityBase):
    """Group B: score caps at 0.25 / 0.50 / unchanged; warnings match exactly."""

    def _run_dual_signal_parity(
        self,
        question_type: str,
        gov_both: GovernanceReport,
        gov_a_only: GovernanceReport,
        gov_b_only: GovernanceReport,
        gov_neither: GovernanceReport,
        legacy_fn: Callable[[set[str], float, list[str]], float],
    ) -> None:
        """Shared three-tier parity assertion for all Group B profiles."""
        tier_cases = [
            (gov_both,    1.00, "both-present"),
            (gov_a_only,  0.50, "signal-A-only"),
            (gov_b_only,  0.50, "signal-B-only"),
            (gov_neither, 0.25, "neither"),
        ]
        query = _query(question_type)
        for gov, expected_score, label in tier_cases:
            with self.subTest(label=label):
                sn = self._snames(gov)
                result = run_parity(
                    lambda w, s=sn: legacy_fn(s, 1.0, w),
                    self._profile_fn(question_type, query, gov, _draft(), 1.0),
                )
                assert_parity(self, result, check_warnings=True, msg=f"{question_type} {label}")
                self.assertAlmostEqual(
                    result.legacy_score, expected_score, places=9,
                    msg=f"{question_type} {label}: expected {expected_score}, got {result.legacy_score}",
                )

    # -- B1: season_line_margin_review --------------------------------------

    def test_b1_season_line_margin_parity(self) -> None:
        price_ev = _ev("FinMind TaiwanStockPrice", excerpt="price data")
        margin_ev = _ev("FinMind margin_purchase data", excerpt="margin flow")
        other_ev = _ev("Generic News Source", excerpt="market commentary")

        self._run_dual_signal_parity(
            question_type="season_line_margin_review",
            gov_both=_gov(price_ev, margin_ev),
            gov_a_only=_gov(price_ev, other_ev),   # price only
            gov_b_only=_gov(other_ev, margin_ev),  # margin only
            gov_neither=_gov(other_ev),
            legacy_fn=self.layer._apply_season_line_margin_rules,
        )

    # -- B2: fcf_dividend_sustainability_review -----------------------------

    def test_b2_fcf_dividend_parity(self) -> None:
        cash_ev = _ev("FinMind cashflows statement", excerpt="FCF data")
        div_ev = _ev("FinMind Dividend policy data", excerpt="dividend policy")
        other_ev = _ev("Generic News Source", excerpt="market commentary")

        self._run_dual_signal_parity(
            question_type="fcf_dividend_sustainability_review",
            gov_both=_gov(cash_ev, div_ev),
            gov_a_only=_gov(cash_ev, other_ev),
            gov_b_only=_gov(other_ev, div_ev),
            gov_neither=_gov(other_ev),
            legacy_fn=self.layer._apply_fcf_dividend_rules,
        )

    # -- B3: debt_dividend_safety_review ------------------------------------

    def test_b3_debt_dividend_parity(self) -> None:
        bs_ev = _ev("FinMind balance_sheet data", excerpt="balance sheet")
        div_ev = _ev("FinMind Dividend policy data", excerpt="dividend policy")
        other_ev = _ev("Generic News Source", excerpt="market commentary")

        self._run_dual_signal_parity(
            question_type="debt_dividend_safety_review",
            gov_both=_gov(bs_ev, div_ev),
            gov_a_only=_gov(bs_ev, other_ev),
            gov_b_only=_gov(other_ev, div_ev),
            gov_neither=_gov(other_ev),
            legacy_fn=self.layer._apply_debt_dividend_rules,
        )


if __name__ == "__main__":
    unittest.main()
