"""test_policy_registry.py

測試 P2：PolicyRegistry 骨架（Wave 4 Stage 6a：tag-based routing）。

驗證項目：
1. 每個 Intent 都有最少一支 policy（27 支原始 policy 全數保留，改以 intent+tag 主鍵）
2. resolve_by_tags() 以 intent + controlled_tags 精確匹配正確回傳
3. resolve_by_tags() fallback：無 tag 時回傳 intent 的通用 policy，match_type="generic"
4. resolve_by_tags() 最終 fallback（未知 intent → news_generic match_type="fallback"）
5. 所有 policy 的 facet 不為空
6. cove_eligible 只有財報/估值/財務健康類才是 True（NEWS_DIGEST 全為 False）
7. register() 以 (intent, frozenset(topic_tags)) 為主鍵並可覆蓋
8. get_policy_registry() 回傳單例
"""
from __future__ import annotations

import unittest

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.query_policy import (
    PolicyRegistry,
    QueryPolicy,
    get_policy_registry,
)


class PolicyRegistryBasicTest(unittest.TestCase):

    def setUp(self) -> None:
        self.registry = PolicyRegistry()

    # ── 完整性檢查 ──────────────────────────────────────────────────────────

    def test_all_expected_policies_registered(self):
        """7 大 Intent 都應該至少有一支 policy 登記，而且總數維持 27 支。"""
        all_policies = self.registry.get_all()
        self.assertEqual(
            len(all_policies), 27,
            f"預期 27 支 policy，實際 {len(all_policies)}",
        )
        intents_with_policy = {p.intent for p in all_policies}
        expected_intents = {
            Intent.NEWS_DIGEST,
            Intent.EARNINGS_REVIEW,
            Intent.VALUATION_CHECK,
            Intent.DIVIDEND_ANALYSIS,
            Intent.FINANCIAL_HEALTH,
            Intent.TECHNICAL_VIEW,
            Intent.INVESTMENT_ASSESSMENT,
        }
        self.assertEqual(
            intents_with_policy, expected_intents,
            f"缺少 intent：{expected_intents - intents_with_policy}",
        )

    def test_all_policies_have_non_empty_facets(self):
        """每個 policy 的 required_facets 或 preferred_facets 至少有一個不為空。"""
        for policy in self.registry.get_all():
            has_facets = bool(policy.required_facets or policy.preferred_facets)
            self.assertTrue(
                has_facets,
                f"{policy.retrieval_profile_key} 的 required + preferred facets 都是空的",
            )

    def test_all_policies_have_retrieval_profile_key(self):
        """每個 policy 都必須有非空的 retrieval_profile_key。"""
        for policy in self.registry.get_all():
            self.assertTrue(
                policy.retrieval_profile_key,
                f"intent={policy.intent} 的 policy 缺少 retrieval_profile_key",
            )

    # ── resolve_by_tags() 精確匹配 ──────────────────────────────────────────

    def test_resolve_shipping_exact_match(self):
        """NEWS_DIGEST + {航運, SCFI} 應命中 news_shipping。"""
        policy = self.registry.resolve_by_tags(
            Intent.NEWS_DIGEST, ("航運", "SCFI"),
        )
        self.assertEqual(policy.intent, Intent.NEWS_DIGEST)
        self.assertEqual(policy.retrieval_profile_key, "news_shipping")
        self.assertEqual(policy.match_type, "exact")

    def test_resolve_earnings_fundamental(self):
        """EARNINGS_REVIEW + {財報} 應命中 earnings_fundamental 且 cove_eligible。"""
        policy = self.registry.resolve_by_tags(
            Intent.EARNINGS_REVIEW, ("財報",),
        )
        self.assertEqual(policy.retrieval_profile_key, "earnings_fundamental")
        self.assertTrue(policy.cove_eligible)
        self.assertEqual(policy.match_type, "exact")

    def test_resolve_investment_support_min_evidence(self):
        """investment_support 需要至少 3 篇文件。"""
        policy = self.registry.resolve_by_tags(
            Intent.INVESTMENT_ASSESSMENT, ("投資評估", "基本面", "本益比"),
        )
        self.assertGreaterEqual(policy.min_evidence_count, 3)

    def test_resolve_dividend_fcf_requires_cashflow(self):
        """dividend_fcf 必須包含 CASH_FLOW facet。"""
        policy = self.registry.resolve_by_tags(
            Intent.DIVIDEND_ANALYSIS, ("股利", "現金流"),
        )
        self.assertIn(DataFacet.CASH_FLOW, policy.required_facets)

    # ── resolve_by_tags() fallback 行為 ──────────────────────────────────────

    def test_resolve_empty_tags_returns_generic(self):
        """空 controlled_tags 應 fallback 到 intent 的通用 policy，match_type='generic'。"""
        policy = self.registry.resolve_by_tags(
            Intent.EARNINGS_REVIEW, (),
        )
        self.assertEqual(policy.intent, Intent.EARNINGS_REVIEW)
        self.assertEqual(policy.match_type, "generic")

    def test_resolve_unknown_tags_falls_back_to_generic(self):
        """沒有任何已知 tag 命中時，應 fallback 到同 intent 的通用 policy。"""
        policy = self.registry.resolve_by_tags(
            Intent.EARNINGS_REVIEW, ("這個標籤完全沒人認識",),
        )
        self.assertEqual(policy.intent, Intent.EARNINGS_REVIEW)
        self.assertEqual(policy.match_type, "generic")

    # ── cove_eligible 邏輯 ──────────────────────────────────────────────────

    def test_cove_eligible_false_for_news_policies(self):
        """NEWS_DIGEST 下所有 policy 都不應 cove_eligible。"""
        news_policies = [
            p for p in self.registry.get_all()
            if p.intent == Intent.NEWS_DIGEST
        ]
        for p in news_policies:
            self.assertFalse(
                p.cove_eligible,
                f"NEWS_DIGEST 的 {p.retrieval_profile_key} 不應是 cove_eligible",
            )

    def test_cove_eligible_true_for_key_financial_policies(self):
        """財報主線 policies 應 cove_eligible=True。"""
        cases = [
            (Intent.EARNINGS_REVIEW,   ("財報",),            "earnings_fundamental"),
            (Intent.EARNINGS_REVIEW,   ("EPS", "股利"),       "earnings_eps_dividend"),
            (Intent.FINANCIAL_HEALTH,  ("獲利", "穩定性"),     "health_profitability"),
        ]
        for intent, tags, expected_profile in cases:
            policy = self.registry.resolve_by_tags(intent, tags)
            self.assertTrue(
                policy.cove_eligible,
                f"{expected_profile} 應是 cove_eligible=True",
            )
            self.assertEqual(policy.retrieval_profile_key, expected_profile)

    # ── 手動 register ────────────────────────────────────────────────────────

    def test_manual_register_overrides_existing(self):
        """以同 (intent, topic_tags) 呼叫 register() 應覆蓋舊 policy。"""
        original = self.registry.resolve_by_tags(
            Intent.NEWS_DIGEST, ("航運", "SCFI"),
        )
        self.assertEqual(original.retrieval_profile_key, "news_shipping")

        custom = QueryPolicy(
            intent=Intent.NEWS_DIGEST,
            required_facets=frozenset({DataFacet.NEWS}),
            preferred_facets=frozenset(),
            retrieval_profile_key="custom_test",
            topic_tags=("航運", "SCFI"),
            confidence_cap=0.9,
        )
        self.registry.register(custom)
        resolved = self.registry.resolve_by_tags(
            Intent.NEWS_DIGEST, ("航運", "SCFI"),
        )
        self.assertEqual(resolved.retrieval_profile_key, "custom_test")
        self.assertEqual(resolved.confidence_cap, 0.9)


class PolicyRegistrySingletonTest(unittest.TestCase):

    def test_get_policy_registry_returns_singleton(self):
        """get_policy_registry() 多次呼叫應回傳同一個物件。"""
        registry_a = get_policy_registry()
        registry_b = get_policy_registry()
        self.assertIs(registry_a, registry_b)

    def test_singleton_has_27_policies(self):
        """單例中應包含 27 支 policy。"""
        registry = get_policy_registry()
        self.assertEqual(len(registry.get_all()), 27)


if __name__ == "__main__":
    unittest.main()
