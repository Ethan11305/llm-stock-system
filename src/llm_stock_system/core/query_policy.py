"""query_policy.py  —  P2 Routing / Policy Registry

目標：將目前散落在三處的 routing 配置集中到一個「唯一事實來源」：
  1. InputLayer._detect_question_type()      → 決定 question_type（保持不動）
  2. FinMindPostgresGateway._resolve_retrieval_profile() → retrieval_profile_key（未來可刪）
  3. INTENT_FACET_SPECS（models.py）         → facet 需求（未來可刪）

遷移策略（漸進式）：
  Phase 1（本版）：
    - 建立 QueryPolicy 資料類別 + PolicyRegistry 骨架
    - 註冊全部 27 個 question_type 的 policy（直接對應現有邏輯）
    - Pipeline / InputLayer 完全不變；現有邏輯保持原樣
    - PolicyRegistry 可被外部讀取，但尚未接入主流程
  Phase 2（已完成）：
    - 新增 resolve_by_tags(intent, topic_tags)：以 intent + topic_tags 路由，不依賴 question_type
    - _resolve_retrieval_profile 優先查 Registry（tag-based），找不到才 fallback 到原有 if-elif
    - ValidationLayer 改用 resolve_by_tags，移除 question_type 依賴
  Phase 3（最終）：
    - InputLayer 只做 ticker 辨識 + intent 分類
    - _resolve_retrieval_profile if-elif 邏輯刪除（Registry 完全取代）

設計原則：
  - frozen dataclass：policy 是不可變配置，便於快取和測試
  - 全部 key 都是字串（不綁定具體 gateway 實作）
  - 失敗 fallback：resolve() / resolve_by_tags() 永遠回傳一個 policy，不會 raise
  - routing 主軸：intent + topic_tags（不依賴 question_type）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import INTENT_FACET_SPECS, QUESTION_TYPE_FALLBACK_TOPIC_TAGS, QUESTION_TYPE_TO_INTENT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QueryPolicy：單一查詢策略的完整描述
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryPolicy:
    """單一查詢策略：描述某種查詢意圖需要哪些資源與驗證規則。

    Attributes:
        intent:                 查詢意圖（NewsDigest / EarningsReview 等）
        question_type:          細粒度查詢類型（如 "news_shipping"）
        required_facets:        必要的資料 facet（缺少時標記 facet_miss）
        preferred_facets:       可選的資料 facet（缺少時降低信心分）
        retrieval_profile_key:  對應 RETRIEVAL_PROFILES 的 key（用於 gateway）
        topic_tags:             路由信號標籤（傳給 generation / validation）
        confidence_cap:         信心分上限（None = 不限）
        min_evidence_count:     最低文件數要求（低於此數降為 YELLOW）
        cove_eligible:          是否符合 CoVe 驗證條件（P3 用）
    """

    intent: Intent
    question_type: str
    required_facets: frozenset[DataFacet]
    preferred_facets: frozenset[DataFacet]
    retrieval_profile_key: str
    topic_tags: tuple[str, ...]
    confidence_cap: float | None = None
    min_evidence_count: int = 1
    cove_eligible: bool = False   # P3 CoVe 用：財報類 + 數字型回答才啟用


# ─────────────────────────────────────────────────────────────────────────────
# 工廠函式：從現有常數快速建立 policy
# ─────────────────────────────────────────────────────────────────────────────

def _make_policy(
    question_type: str,
    retrieval_profile_key: str,
    *,
    extra_required: frozenset[DataFacet] | None = None,
    extra_preferred: frozenset[DataFacet] | None = None,
    routing_tags: tuple[str, ...] | None = None,
    confidence_cap: float | None = None,
    min_evidence_count: int = 1,
    cove_eligible: bool = False,
) -> QueryPolicy:
    """從現有常數建立 QueryPolicy（避免重複定義 facet）。

    facet 來源優先順序：
      1. INTENT_FACET_SPECS[intent]（主要來源）
      2. extra_required / extra_preferred（細粒度覆蓋）

    topic_tags 來源優先順序：
      1. routing_tags（明確指定）← 用於 resolve_by_tags() 路由，應包含所有路由信號標籤
      2. QUESTION_TYPE_FALLBACK_TOPIC_TAGS（最小集 fallback）← 僅為 StructuredQuery 注入用

    注意：QUESTION_TYPE_FALLBACK_TOPIC_TAGS 是「注入到 StructuredQuery 的最小集」，
    而路由需要的標籤集往往更寬（如 theme_impact_review 在 if-elif 也匹配 "AI", "電動車" 等）。
    需要使用 routing_tags 來指定完整的路由標籤集。
    """
    intent = QUESTION_TYPE_TO_INTENT.get(question_type, Intent.NEWS_DIGEST)
    base_spec = INTENT_FACET_SPECS.get(intent)

    required = (base_spec.required if base_spec else frozenset()) | (extra_required or frozenset())
    preferred = (base_spec.preferred if base_spec else frozenset()) | (extra_preferred or frozenset())
    topic_tags = routing_tags if routing_tags is not None else QUESTION_TYPE_FALLBACK_TOPIC_TAGS.get(question_type, ())

    return QueryPolicy(
        intent=intent,
        question_type=question_type,
        required_facets=required,
        preferred_facets=preferred,
        retrieval_profile_key=retrieval_profile_key,
        topic_tags=topic_tags,
        confidence_cap=confidence_cap,
        min_evidence_count=min_evidence_count,
        cove_eligible=cove_eligible,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PolicyRegistry
# ─────────────────────────────────────────────────────────────────────────────

class PolicyRegistry:
    """集中管理所有 QueryPolicy 的 Registry。

    取代目前分散在三處的配置：
      - InputLayer._detect_question_type()                  → 匹配邏輯（未來）
      - FinMindPostgresGateway._resolve_retrieval_profile() → 資料需求（未來）
      - models.INTENT_FACET_SPECS                           → facet 規格（未來）

    routing 主軸（Phase 2+）：
      - 使用 resolve_by_tags(intent, topic_tags) 作為主要路由，不依賴 question_type
      - resolve(intent, question_type) 保留以供需要 question_type 的遺留路徑使用

    使用方式：
        registry = PolicyRegistry()
        # ✅ 新方式（Phase 2+）：intent + tags routing
        policy = registry.resolve_by_tags(intent=Intent.NEWS_DIGEST, topic_tags={"航運", "SCFI"})
        # → QueryPolicy(retrieval_profile_key="news_shipping", ...)

        # 遺留方式（question_type based）：
        policy = registry.resolve(intent=Intent.NEWS_DIGEST, question_type="shipping_rate_impact_review")
    """

    def __init__(self) -> None:
        # (intent, question_type) → QueryPolicy
        self._policies: dict[tuple[Intent, str], QueryPolicy] = {}
        self._register_defaults()

    # ──────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────

    def resolve_by_tags(
        self,
        intent: Intent,
        topic_tags: frozenset[str] | set[str] | tuple[str, ...] | list[str],
    ) -> QueryPolicy:
        """根據 intent + topic_tags 查找最匹配的 policy。

        這是 Phase 2+ 的目標路由方式，完全不依賴 question_type。

        Matching 策略：
          1. 過濾出 policy.intent == intent 的所有 policy
          2. 計算 query topic_tags 與每個 policy.topic_tags 的交集大小（tag overlap score）
          3. 回傳 overlap score 最高的 policy（至少要有 1 個 tag 匹配）
          4. 若無任何交集（例如空 tags 或通用查詢），fallback 到該 intent 的通用 policy
             （topic_tags 最少的 policy，通常是 "market_summary" / generic 類型）
          5. 若連 intent 也找不到，fallback 到 news_generic

        設計原則：
          - 不 raise：永遠回傳有效的 policy
          - 無狀態：tag matching 為純函式，可測試
          - 與現有 _resolve_retrieval_profile() if-elif 邏輯語意對齊
        """
        tag_set = frozenset(topic_tags)

        # 找出同 intent 的所有 policy
        candidates = [
            policy
            for (pol_intent, _), policy in self._policies.items()
            if pol_intent == intent
        ]

        if not candidates:
            logger.warning(
                "PolicyRegistry.resolve_by_tags: 找不到 intent=%s 的 policy，使用 news_generic。",
                intent.value,
            )
            return self._default_policy()

        # 計算每個 candidate 的 tag overlap score
        best_policy: QueryPolicy | None = None
        best_score = 0

        for policy in candidates:
            if not policy.topic_tags:
                # 沒有 topic_tags 的 policy（如 market_summary）是通用 fallback，
                # 不參與 tag matching 競爭，留到最後處理
                continue
            score = len(frozenset(policy.topic_tags) & tag_set)
            if score > best_score:
                best_score = score
                best_policy = policy

        # 有明確 tag 匹配：回傳最佳匹配
        if best_policy is not None and best_score > 0:
            return best_policy

        # 無 tag 匹配（空 tags 或通用查詢）：回傳 intent 的通用 policy
        # 選 topic_tags 最少的，通常是 generic / market_summary 類型
        generic = min(candidates, key=lambda p: len(p.topic_tags))
        return generic

    def resolve(self, intent: Intent, question_type: str) -> QueryPolicy:
        """根據 intent + question_type 查找對應的 policy。

        遺留路由方式，供需要 question_type 精確匹配的路徑使用。
        新路徑請改用 resolve_by_tags()。

        Fallback 順序：
        1. 精確匹配 (intent, question_type)
        2. 只用 intent 匹配（取第一個符合的）
        3. 預設 policy（news_generic）
        """
        key = (intent, question_type)
        if key in self._policies:
            return self._policies[key]

        # Fallback: 只用 intent 匹配
        for (pol_intent, _), policy in self._policies.items():
            if pol_intent == intent:
                return policy

        logger.warning(
            "PolicyRegistry: 找不到 policy（intent=%s, question_type=%s），使用 news_generic。",
            intent.value, question_type,
        )
        return self._default_policy()

    def get_all(self) -> list[QueryPolicy]:
        """回傳所有已註冊的 policy（便於偵錯和測試）。"""
        return list(self._policies.values())

    def register(self, policy: QueryPolicy) -> None:
        """手動註冊一個 policy（便於測試或動態擴充）。"""
        key = (policy.intent, policy.question_type)
        self._policies[key] = policy

    # ──────────────────────────────────────
    # 預設 policy 註冊
    # ──────────────────────────────────────

    def _register_defaults(self) -> None:
        """將全部 27 個 question_type 的 policy 預先登記。

        遷移來源：
          - INTENT_FACET_SPECS（models.py）
          - RETRIEVAL_PROFILES（postgres_market_data.py）
          - _resolve_retrieval_profile() 的 if/elif 邏輯
        """
        policies = [
            # ── NEWS_DIGEST 類 ──────────────────────────────────────────────
            _make_policy("market_summary",                "news_generic"),
            _make_policy(
                "shipping_rate_impact_review", "news_shipping",
                # routing_tags = if-elif 中 news_shipping 的完整判斷標籤集
                routing_tags=("航運", "SCFI"),
            ),
            _make_policy(
                "electricity_cost_impact_review", "news_electricity",
                routing_tags=("電價", "成本"),
            ),
            _make_policy(
                "macro_yield_sentiment_review", "news_macro",
                # if-elif 檢查 {"總經", "CPI", "殖利率"}；fallback_tags 只有 ("CPI", "殖利率")
                routing_tags=("總經", "CPI", "殖利率"),
            ),
            _make_policy(
                "theme_impact_review", "news_theme",
                # if-elif 檢查 {"題材", "產業", "AI", "電動車", "半導體設備"}；
                # fallback_tags 只有 ("題材", "產業")，補齊完整路由標籤集
                routing_tags=("題材", "產業", "AI", "電動車", "半導體設備"),
            ),
            _make_policy(
                "guidance_reaction_review", "news_guidance",
                routing_tags=("法說", "指引"),
            ),
            _make_policy(
                "listing_revenue_review", "news_listing",
                routing_tags=("上市", "營收"),
            ),

            # ── EARNINGS_REVIEW 類 ─────────────────────────────────────────
            _make_policy(
                "earnings_summary",
                "earnings_fundamental",
                cove_eligible=True,          # 財報類 + 數字多，適合 CoVe
                min_evidence_count=2,
            ),
            _make_policy(
                "eps_dividend_review",
                "earnings_eps_dividend",
                cove_eligible=True,
            ),
            _make_policy(
                "monthly_revenue_yoy_review",
                "earnings_monthly_revenue",
                cove_eligible=True,
            ),
            _make_policy(
                "margin_turnaround_review",
                "earnings_margin_turnaround",
                cove_eligible=True,
            ),

            # ── VALUATION_CHECK 類 ─────────────────────────────────────────
            _make_policy(
                "pe_valuation_review",
                "valuation_pe_only",
                cove_eligible=True,
            ),
            _make_policy(
                "fundamental_pe_review",
                "valuation_fundamental",
                cove_eligible=True,
            ),
            _make_policy("price_range",   "valuation_price_range"),
            _make_policy("price_outlook", "valuation_price_outlook"),

            # ── DIVIDEND_ANALYSIS 類 ──────────────────────────────────────
            _make_policy("dividend_yield_review",            "dividend_yield"),
            _make_policy("ex_dividend_performance",          "dividend_ex"),
            _make_policy(
                "fcf_dividend_sustainability_review",
                "dividend_fcf",
                cove_eligible=True,
                extra_required=frozenset({DataFacet.CASH_FLOW}),
            ),
            _make_policy(
                "debt_dividend_safety_review",
                "dividend_debt",
                cove_eligible=True,
                extra_required=frozenset({DataFacet.BALANCE_SHEET}),
            ),

            # ── FINANCIAL_HEALTH 類 ───────────────────────────────────────
            _make_policy(
                "profitability_stability_review",
                "health_profitability",
                cove_eligible=True,
                min_evidence_count=2,
            ),
            _make_policy(
                "gross_margin_comparison_review",
                "health_gross_margin_cmp",
                cove_eligible=True,
            ),
            _make_policy(
                "revenue_growth_review",
                "health_revenue_growth",
                cove_eligible=True,
            ),

            # ── TECHNICAL_VIEW 類 ─────────────────────────────────────────
            _make_policy("technical_indicator_review", "technical_indicators"),
            _make_policy("season_line_margin_review",  "technical_margin_flow"),

            # ── INVESTMENT_ASSESSMENT 類 ──────────────────────────────────
            _make_policy(
                "investment_support",
                "investment_support",
                cove_eligible=True,
                min_evidence_count=3,
            ),
            _make_policy("risk_review",          "investment_risk"),
            _make_policy("announcement_summary", "investment_announcement"),
        ]

        for policy in policies:
            self.register(policy)

        logger.debug("PolicyRegistry 初始化：已載入 %d 個 policy。", len(self._policies))

    def _default_policy(self) -> QueryPolicy:
        """最終 fallback policy（news_generic）。"""
        return _make_policy("market_summary", "news_generic")


# ─────────────────────────────────────────────────────────────────────────────
# 模組層級單例（便於重用，避免每次都重新初始化）
# ─────────────────────────────────────────────────────────────────────────────

_registry: PolicyRegistry | None = None


def get_policy_registry() -> PolicyRegistry:
    """取得全域 PolicyRegistry 單例。

    使用延遲初始化確保 import 時不會有副作用。
    """
    global _registry
    if _registry is None:
        _registry = PolicyRegistry()
    return _registry
