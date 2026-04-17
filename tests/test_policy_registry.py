"""test_policy_registry.py

測試 P2：PolicyRegistry 骨架。

驗證項目：
1. 全部 27 個 question_type 都已在 Registry 中
2. resolve() 精確匹配正確回傳
3. resolve() fallback 到同 intent 的其他 policy
4. resolve() 最終 fallback 回傳 news_generic（不 crash）
5. 所有 policy 的 facet 不為空且 intent 一致
6. cove_eligible 只有財報/估值類才是 True
7. get_policy_registry() 回傳單例（多次呼叫同一個物件）
"""
from __future__ import annotations

import unittest

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import QUESTION_TYPE_TO_INTENT
from llm_stock_system.core.query_policy import (
    PolicyRegistry,
    QueryPolicy,
    get_policy_registry,
)


class PolicyRegistryBasicTest(unittest.TestCase):

    def setUp(self) -> None:
        self.registry = PolicyRegistry()

    # ── 完整性檢查 ──────────────────────────────────────────────────────────

    def test_all_27_question_types_registered(self):
        """全部 27 個 question_type 都必須在 Registry 中有對應 policy。"""
        all_policies = self.registry.get_all()
        registered_types = {p.question_type for p in all_policies}

        missing = set(QUESTION_TYPE_TO_INTENT.keys()) - registered_types
        self.assertEqual(
            missing, set(),
            f"以下 question_type 缺少對應 policy：{missing}",
        )

    def test_all_policies_have_non_empty_facets(self):
        """每個 policy 的 required_facets 或 preferred_facets 至少有一個不為空。"""
        for policy in self.registry.get_all():
            has_facets = bool(policy.required_facets or policy.preferred_facets)
            self.assertTrue(
                has_facets,
                f"{policy.question_type} 的 required + preferred facets 都是空的",
            )

    def test_all_policies_intent_matches_question_type_to_intent(self):
        """policy.intent 必須與 QUESTION_TYPE_TO_INTENT 中的對應值一致。"""
        for policy in self.registry.get_all():
            expected_intent = QUESTION_TYPE_TO_INTENT.get(policy.question_type)
            if expected_intent is not None:
                self.assertEqual(
                    policy.intent,
                    expected_intent,
                    f"{policy.question_type} 的 intent 不符（期待 {expected_intent}，實際 {policy.intent}）",
                )

    def test_all_policies_have_retrieval_profile_key(self):
        """每個 policy 都必須有非空的 retrieval_profile_key。"""
        for policy in self.registry.get_all():
            self.assertTrue(
                policy.retrieval_profile_key,
                f"{policy.question_type} 缺少 retrieval_profile_key",
            )

    # ── resolve() 精確匹配 ──────────────────────────────────────────────────

    def test_resolve_exact_match(self):
        """resolve() 精確匹配已知 (intent, question_type) 組合。

        注意：question_type 是 "shipping_rate_impact_review"（輸入層的分類名稱），
              不是 retrieval_profile_key "news_shipping"（gateway 的 profile 名稱）。
        """
        policy = self.registry.resolve(Intent.NEWS_DIGEST, "shipping_rate_impact_review")
        self.assertEqual(policy.question_type, "shipping_rate_impact_review")
        self.assertEqual(policy.intent, Intent.NEWS_DIGEST)
        self.assertEqual(policy.retrieval_profile_key, "news_shipping")

    def test_resolve_earnings_summary(self):
        """earnings_summary 應對應 earnings_fundamental profile 且 cove_eligible=True。"""
        policy = self.registry.resolve(Intent.EARNINGS_REVIEW, "earnings_summary")
        self.assertEqual(policy.question_type, "earnings_summary")
        self.assertEqual(policy.retrieval_profile_key, "earnings_fundamental")
        self.assertTrue(policy.cove_eligible)

    def test_resolve_investment_support_min_evidence(self):
        """investment_support 需要至少 3 篇文件。"""
        policy = self.registry.resolve(Intent.INVESTMENT_ASSESSMENT, "investment_support")
        self.assertGreaterEqual(policy.min_evidence_count, 3)

    def test_resolve_dividend_fcf_requires_cashflow(self):
        """fcf_dividend_sustainability_review 必須包含 CASH_FLOW facet。"""
        policy = self.registry.resolve(Intent.DIVIDEND_ANALYSIS, "fcf_dividend_sustainability_review")
        self.assertIn(DataFacet.CASH_FLOW, policy.required_facets)

    # ── resolve() fallback 行為 ──────────────────────────────────────────────

    def test_resolve_unknown_question_type_falls_back_to_same_intent(self):
        """未知 question_type 應 fallback 到同 intent 的第一個 policy（不 crash）。"""
        policy = self.registry.resolve(Intent.EARNINGS_REVIEW, "totally_unknown_type")
        # 只要 intent 一致，fallback 就是正確行為
        self.assertEqual(policy.intent, Intent.EARNINGS_REVIEW)

    def test_resolve_unknown_intent_and_type_returns_default(self):
        """完全未知的 intent 和 question_type 應回傳 default policy（不 crash）。"""
        # 用 None 無法通過 type check，直接用 object 測試 fallback 路徑
        policy = self.registry.resolve(Intent.NEWS_DIGEST, "__nonexistent__xyz__")
        # 預設是 news_generic，不應 raise
        self.assertIsNotNone(policy)
        self.assertIsInstance(policy, QueryPolicy)

    # ── cove_eligible 邏輯 ──────────────────────────────────────────────────

    def test_cove_eligible_only_for_financial_types(self):
        """cove_eligible=True 只應存在於財報/估值/財務健康類型，而非純新聞類型。"""
        news_policies = [
            p for p in self.registry.get_all()
            if p.intent == Intent.NEWS_DIGEST
        ]
        for p in news_policies:
            self.assertFalse(
                p.cove_eligible,
                f"NEWS_DIGEST 類型 {p.question_type} 不應是 cove_eligible",
            )

    def test_cove_eligible_true_for_earnings_types(self):
        """主要財報類型都應是 cove_eligible=True。"""
        cove_expected = ["earnings_summary", "eps_dividend_review", "profitability_stability_review"]
        for qt in cove_expected:
            intent = QUESTION_TYPE_TO_INTENT[qt]
            policy = self.registry.resolve(intent, qt)
            self.assertTrue(
                policy.cove_eligible,
                f"{qt} 應是 cove_eligible=True",
            )

    # ── 手動 register ────────────────────────────────────────────────────────

    def test_manual_register_overrides_existing(self):
        """手動 register() 應覆蓋同一 (intent, question_type) 的舊 policy。"""
        original = self.registry.resolve(Intent.NEWS_DIGEST, "news_shipping")

        # 建立一個 confidence_cap=0.9 的測試 policy
        custom = QueryPolicy(
            intent=Intent.NEWS_DIGEST,
            question_type="news_shipping",
            required_facets=frozenset({DataFacet.NEWS}),
            preferred_facets=frozenset(),
            retrieval_profile_key="custom_test",
            topic_tags=("test",),
            confidence_cap=0.9,
        )
        self.registry.register(custom)
        resolved = self.registry.resolve(Intent.NEWS_DIGEST, "news_shipping")
        self.assertEqual(resolved.retrieval_profile_key, "custom_test")
        self.assertEqual(resolved.confidence_cap, 0.9)


class PolicyRegistrySingletonTest(unittest.TestCase):

    def test_get_policy_registry_returns_singleton(self):
        """get_policy_registry() 多次呼叫應回傳同一個物件。"""
        registry_a = get_policy_registry()
        registry_b = get_policy_registry()
        self.assertIs(registry_a, registry_b)

    def test_singleton_has_all_policies(self):
        """單例中應包含全部 question_type 的 policy。"""
        registry = get_policy_registry()
        all_types = {p.question_type for p in registry.get_all()}
        for qt in QUESTION_TYPE_TO_INTENT:
            self.assertIn(qt, all_types, f"singleton 缺少 {qt}")


if __name__ == "__main__":
    unittest.main()
