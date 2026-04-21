"""query_policy.py  —  Routing / Policy Registry

目標：將 routing 配置集中到單一事實來源。主要 API 是
:py:meth:`PolicyRegistry.resolve_by_tags`，由 ``intent + controlled_tags``
決定 retrieval profile、facet 需求、驗證規則與預設時間範圍。

Wave 4 Stage 6a（本版）：
    - QueryPolicy 不再持有 ``question_type`` 欄位
    - PolicyRegistry 只剩一條查找路徑：``resolve_by_tags``
    - ``resolve(intent, question_type)`` 與 ``_qt_to_key`` 索引已刪除
    - ``_make_policy(intent, retrieval_profile_key, routing_tags=...)``
      改為以 Intent 為必要參數，完全擺脫 question_type 字串

設計原則：
  - frozen dataclass：policy 是不可變配置，便於快取與測試
  - 全部 routing key 都以 frozenset[str] 表示（不綁定具體 TopicTag enum 來源）
  - 失敗 fallback：resolve_by_tags() 永遠回傳一個 policy，不會 raise
  - routing 主軸：intent + controlled_tags（不依賴 question_type，不混入 free_keywords）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from llm_stock_system.core.enums import DataFacet, Intent
from llm_stock_system.core.models import INTENT_FACET_SPECS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QueryPolicy：單一查詢策略的完整描述
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryPolicy:
    """單一查詢策略：描述某種查詢意圖需要哪些資源與驗證規則。

    Attributes:
        intent:                 查詢意圖（NewsDigest / EarningsReview 等）
        required_facets:        必要的資料 facet（缺少時標記 facet_miss）
        preferred_facets:       可選的資料 facet（缺少時降低信心分）
        retrieval_profile_key:  對應 RETRIEVAL_PROFILES 的 key（用於 gateway）
        topic_tags:             路由信號標籤（主鍵之一，參與 resolve_by_tags 匹配）
        confidence_cap:         信心分上限（None = 不限）
        min_evidence_count:     最低文件數要求（低於此數降為 YELLOW）
        cove_eligible:          是否符合 CoVe 驗證條件（P3 用）
    """

    intent: Intent
    required_facets: frozenset[DataFacet]
    preferred_facets: frozenset[DataFacet]
    retrieval_profile_key: str
    topic_tags: tuple[str, ...]
    confidence_cap: float | None = None
    min_evidence_count: int = 1
    cove_eligible: bool = False   # P3 CoVe 用：財報類 + 數字型回答才啟用
    match_type: str = "generic"           # 由 resolve_by_tags() 解析時設定，勿手動指定
                                          # "exact"   → controlled_tags 全部命中
                                          # "partial" → controlled_tags 部分命中
                                          # "generic" → 無 tag 匹配，退回 intent 通用 policy
                                          # "fallback"→ intent 也找不到，退回全域預設
    default_time_range_days: int | None = None   # 規劃項目 C：此查詢類型的預設時間窗口
    default_time_range_label: str | None = None  # 對應的 label（如 "1y", "latest_quarter"）


# ─────────────────────────────────────────────────────────────────────────────
# 工廠函式：以 intent + routing_tags 建立 policy
# ─────────────────────────────────────────────────────────────────────────────

def _make_policy(
    intent: Intent,
    retrieval_profile_key: str,
    *,
    routing_tags: tuple[str, ...] = (),
    extra_required: frozenset[DataFacet] | None = None,
    extra_preferred: frozenset[DataFacet] | None = None,
    confidence_cap: float | None = None,
    min_evidence_count: int = 1,
    cove_eligible: bool = False,
    default_time_range_days: int | None = None,
    default_time_range_label: str | None = None,
) -> QueryPolicy:
    """建立 QueryPolicy。

    facet 來源：
      1. INTENT_FACET_SPECS[intent]（主要來源）
      2. extra_required / extra_preferred（細粒度覆蓋）

    ``routing_tags`` 是 resolve_by_tags() 匹配的主鍵之一，空 tuple 表示
    「同 intent 下的通用 policy」。
    """
    base_spec = INTENT_FACET_SPECS.get(intent)

    required = (base_spec.required if base_spec else frozenset()) | (extra_required or frozenset())
    preferred = (base_spec.preferred if base_spec else frozenset()) | (extra_preferred or frozenset())

    return QueryPolicy(
        intent=intent,
        required_facets=required,
        preferred_facets=preferred,
        retrieval_profile_key=retrieval_profile_key,
        topic_tags=tuple(routing_tags),
        confidence_cap=confidence_cap,
        min_evidence_count=min_evidence_count,
        cove_eligible=cove_eligible,
        default_time_range_days=default_time_range_days,
        default_time_range_label=default_time_range_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PolicyRegistry
# ─────────────────────────────────────────────────────────────────────────────

class PolicyRegistry:
    """集中管理所有 QueryPolicy 的 Registry。

    routing 主軸（Wave 4 Stage 6a）：
      - 儲存主鍵：``(Intent, frozenset[str])``（str 為 TopicTag.value）
      - 唯一公開查找 API：``resolve_by_tags(intent, controlled_tags)``
      - 無 question_type 反查索引；routing 完全以 intent + tag set 決定

    使用方式：
        registry = PolicyRegistry()
        policy = registry.resolve_by_tags(
            intent=Intent.NEWS_DIGEST,
            controlled_tags={TopicTag.SHIPPING},
        )
        # → QueryPolicy(retrieval_profile_key="news_shipping", match_type="exact", ...)
    """

    # 每個 intent 的「通用 policy」tag set：當 resolve_by_tags() 沒有任何 tag
    # 命中時，以此 tag set 對主鍵做精確查找取得泛型 policy。這張表必須與
    # _register_defaults 裡每個 intent 「最通用」的那支 policy 的 routing tag
    # 完全一致（見 _INTENT_GENERIC_TAG_EXPECTATIONS 註解）。
    _INTENT_GENERIC_TAGS: dict[Intent, frozenset[str]] = {
        Intent.NEWS_DIGEST:          frozenset(),
        Intent.EARNINGS_REVIEW:      frozenset({"財報"}),
        Intent.VALUATION_CHECK:      frozenset({"本益比"}),
        Intent.FINANCIAL_HEALTH:     frozenset({"獲利", "穩定性"}),
        Intent.DIVIDEND_ANALYSIS:    frozenset({"股利", "殖利率"}),
        Intent.TECHNICAL_VIEW:       frozenset({"技術面"}),
        Intent.INVESTMENT_ASSESSMENT: frozenset({"投資評估", "基本面", "本益比"}),
    }

    def __init__(self) -> None:
        # 主鍵：(intent, frozenset[str] of routing tags) → QueryPolicy
        self._policies: dict[tuple[Intent, frozenset[str]], QueryPolicy] = {}
        self._register_defaults()

    # ──────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────

    def resolve_by_tags(
        self,
        intent: Intent,
        controlled_tags: frozenset | set | tuple | list,
    ) -> QueryPolicy:
        """根據 intent + controlled_tags 查找最匹配的 policy。

        只吃 controlled_tags（TopicTag enum list），不接受混合了
        free_keywords 的 topic_tags。free_keywords 僅供 RetrievalLayer
        展開搜尋用，不參與 routing。

        Matching 策略：
          1. 過濾出 policy.intent == intent 的所有 policy
          2. 將 controlled_tags 轉為 str frozenset（支援 TopicTag enum 或純字串）
          3. 計算 query tag_set 與每個 policy.topic_tags 的交集大小（overlap score）
          4. 回傳 overlap score 最高的 policy，並設定 match_type：
             - "exact"    → score >= len(tag_set) > 0（全部 tag 都被某個 policy 涵蓋）
             - "partial"  → 0 < score < len(tag_set)（部分命中）
             - "generic"  → score == 0 或 tag_set 為空（退回 intent 的通用 policy）
             - "fallback" → 連 intent 都找不到（退回全域預設）
          5. 若連 intent 也找不到，fallback 到 news_generic

        設計原則：
          - 不 raise：永遠回傳有效的 policy
          - 無狀態：tag matching 為純函式，可測試
          - match_type 透過 dataclasses.replace() 設定在回傳的 policy 副本上
        """
        # 將 TopicTag enum 或純字串統一轉為 str frozenset
        tag_set = frozenset(
            t.value if hasattr(t, "value") else str(t)
            for t in controlled_tags
        )

        # 找出同 intent 的所有 policy（以新鍵結構直接分流）
        candidates = [
            (key_tags, policy)
            for (key_intent, key_tags), policy in self._policies.items()
            if key_intent == intent
        ]

        if not candidates:
            logger.warning(
                "PolicyRegistry.resolve_by_tags: 找不到 intent=%s 的 policy，使用 news_generic。",
                intent.value,
            )
            return replace(self._default_policy(), match_type="fallback")

        # 計算每個 candidate 的 tag overlap score（以主鍵中的 tag set 為準）
        best_policy: QueryPolicy | None = None
        best_score = 0

        for key_tags, policy in candidates:
            if not key_tags:
                # 空 tag set 的 policy（如 NEWS_DIGEST 的 market_summary）是通用
                # fallback，不參與 tag matching 競爭，留到最後的 generic 分支處理
                continue
            score = len(key_tags & tag_set)
            if score > best_score:
                best_score = score
                best_policy = policy

        # 有明確 tag 匹配：回傳最佳匹配，並標記 match_type
        if best_policy is not None and best_score > 0:
            if best_score >= len(tag_set):
                computed_match_type = "exact"
            else:
                computed_match_type = "partial"
            return replace(best_policy, match_type=computed_match_type)

        # 無 tag 匹配（空 tags 或通用查詢）：回傳 intent 的通用 policy
        # 以 _INTENT_GENERIC_TAGS 提供的 tag set 對主鍵做精確查找，
        # 避免 min(topic_tags) 因 tag 數量不一而選到錯誤的具體 policy。
        generic_tags = self._INTENT_GENERIC_TAGS.get(intent)
        if generic_tags is not None:
            generic_key = (intent, generic_tags)
            if generic_key in self._policies:
                return replace(self._policies[generic_key], match_type="generic")
        # 最終兜底：選 routing tag 最少的（理論上不應走到這裡）
        _, generic = min(candidates, key=lambda entry: len(entry[0]))
        return replace(generic, match_type="generic")

    def get_all(self) -> list[QueryPolicy]:
        """回傳所有已註冊的 policy（便於偵錯和測試）。"""
        return list(self._policies.values())

    def register(self, policy: QueryPolicy) -> None:
        """手動註冊或覆蓋一個 policy（便於測試或動態擴充）。

        以 ``(intent, frozenset(topic_tags))`` 為主鍵；同主鍵再 register
        會覆蓋舊 policy。
        """
        primary_key = (policy.intent, frozenset(policy.topic_tags))
        self._policies[primary_key] = policy

    # ──────────────────────────────────────
    # 預設 policy 註冊
    # ──────────────────────────────────────

    def _register_defaults(self) -> None:
        """將全部預設 policy 預先登記。

        遷移來源：
          - INTENT_FACET_SPECS（models.py）
          - RETRIEVAL_PROFILES（postgres_market_data.py）
          - _resolve_retrieval_profile() 的 if/elif 邏輯
        """
        policies = [
            # ── NEWS_DIGEST 類 ──────────────────────────────────────────────
            _make_policy(
                Intent.NEWS_DIGEST, "news_generic",
                default_time_range_days=7, default_time_range_label="7d",
            ),
            _make_policy(
                Intent.NEWS_DIGEST, "news_shipping",
                routing_tags=("航運", "SCFI"),
                default_time_range_days=30, default_time_range_label="30d",
            ),
            _make_policy(
                Intent.NEWS_DIGEST, "news_electricity",
                routing_tags=("電價", "成本"),
                default_time_range_days=30, default_time_range_label="30d",
            ),
            _make_policy(
                Intent.NEWS_DIGEST, "news_macro",
                routing_tags=("總經", "CPI", "殖利率"),
                default_time_range_days=30, default_time_range_label="30d",
            ),
            _make_policy(
                Intent.NEWS_DIGEST, "news_theme",
                routing_tags=("題材", "產業", "AI", "電動車", "半導體設備"),
                default_time_range_days=30, default_time_range_label="30d",
            ),
            _make_policy(
                Intent.NEWS_DIGEST, "news_guidance",
                routing_tags=("法說", "指引"),
                default_time_range_days=30, default_time_range_label="30d",
            ),
            _make_policy(
                Intent.NEWS_DIGEST, "news_listing",
                routing_tags=("上市", "營收"),
                default_time_range_days=30, default_time_range_label="30d",
            ),

            # ── EARNINGS_REVIEW 類 ─────────────────────────────────────────
            _make_policy(
                Intent.EARNINGS_REVIEW, "earnings_fundamental",
                routing_tags=("財報",),
                cove_eligible=True, min_evidence_count=2,
                default_time_range_days=90, default_time_range_label="latest_quarter",
            ),
            _make_policy(
                Intent.EARNINGS_REVIEW, "earnings_eps_dividend",
                routing_tags=("EPS", "股利"),
                cove_eligible=True,
                default_time_range_days=365, default_time_range_label="1y",
            ),
            _make_policy(
                Intent.EARNINGS_REVIEW, "earnings_monthly_revenue",
                routing_tags=("月營收",),
                cove_eligible=True,
                default_time_range_days=365, default_time_range_label="1y",
            ),
            _make_policy(
                Intent.EARNINGS_REVIEW, "earnings_margin_turnaround",
                routing_tags=("毛利率", "轉正"),
                cove_eligible=True,
                default_time_range_days=90, default_time_range_label="latest_quarter",
            ),

            # ── VALUATION_CHECK 類 ─────────────────────────────────────────
            _make_policy(
                Intent.VALUATION_CHECK, "valuation_pe_only",
                routing_tags=("本益比",),
                cove_eligible=True,
                default_time_range_days=365, default_time_range_label="1y",
            ),
            _make_policy(
                Intent.VALUATION_CHECK, "valuation_fundamental",
                routing_tags=("基本面", "本益比"),
                cove_eligible=True,
                default_time_range_days=365, default_time_range_label="1y",
            ),
            _make_policy(
                Intent.VALUATION_CHECK, "valuation_price_range",
                routing_tags=("股價區間",),
                default_time_range_days=7, default_time_range_label="7d",
            ),
            _make_policy(
                Intent.VALUATION_CHECK, "valuation_price_outlook",
                routing_tags=("股價", "展望"),
                default_time_range_days=30, default_time_range_label="30d",
            ),

            # ── DIVIDEND_ANALYSIS 類 ──────────────────────────────────────
            _make_policy(
                Intent.DIVIDEND_ANALYSIS, "dividend_yield",
                routing_tags=("股利", "殖利率"),
                default_time_range_days=365, default_time_range_label="1y",
            ),
            _make_policy(
                Intent.DIVIDEND_ANALYSIS, "dividend_ex",
                routing_tags=("除息", "填息"),
                default_time_range_days=365, default_time_range_label="1y",
            ),
            _make_policy(
                Intent.DIVIDEND_ANALYSIS, "dividend_fcf",
                routing_tags=("股利", "現金流"),
                cove_eligible=True,
                extra_required=frozenset({DataFacet.CASH_FLOW}),
                default_time_range_days=1095, default_time_range_label="3y",
            ),
            _make_policy(
                Intent.DIVIDEND_ANALYSIS, "dividend_debt",
                routing_tags=("股利", "負債"),
                cove_eligible=True,
                extra_required=frozenset({DataFacet.BALANCE_SHEET}),
                default_time_range_days=1095, default_time_range_label="3y",
            ),

            # ── FINANCIAL_HEALTH 類 ───────────────────────────────────────
            _make_policy(
                Intent.FINANCIAL_HEALTH, "health_profitability",
                routing_tags=("獲利", "穩定性"),
                cove_eligible=True, min_evidence_count=2,
                default_time_range_days=1825, default_time_range_label="5y",
            ),
            _make_policy(
                Intent.FINANCIAL_HEALTH, "health_gross_margin_cmp",
                routing_tags=("毛利率", "比較"),
                cove_eligible=True,
                default_time_range_days=90, default_time_range_label="latest_quarter",
            ),
            _make_policy(
                Intent.FINANCIAL_HEALTH, "health_revenue_growth",
                routing_tags=("營收", "成長"),
                cove_eligible=True,
                default_time_range_days=90, default_time_range_label="latest_quarter",
            ),

            # ── TECHNICAL_VIEW 類 ─────────────────────────────────────────
            _make_policy(
                Intent.TECHNICAL_VIEW, "technical_indicators",
                routing_tags=("技術面",),
                default_time_range_days=30, default_time_range_label="30d",
            ),
            _make_policy(
                Intent.TECHNICAL_VIEW, "technical_margin_flow",
                routing_tags=("季線", "籌碼"),
                default_time_range_days=90, default_time_range_label="90d",
            ),

            # ── INVESTMENT_ASSESSMENT 類 ──────────────────────────────────
            _make_policy(
                Intent.INVESTMENT_ASSESSMENT, "investment_support",
                routing_tags=("投資評估", "基本面", "本益比"),
                cove_eligible=True, min_evidence_count=3,
                default_time_range_days=7, default_time_range_label="7d",
            ),
            _make_policy(
                Intent.INVESTMENT_ASSESSMENT, "investment_risk",
                routing_tags=("風險",),
                default_time_range_days=7, default_time_range_label="7d",
            ),
            _make_policy(
                Intent.INVESTMENT_ASSESSMENT, "investment_announcement",
                routing_tags=("公告",),
                default_time_range_days=7, default_time_range_label="7d",
            ),
        ]

        for policy in policies:
            self.register(policy)

        logger.debug("PolicyRegistry 初始化：已載入 %d 個 policy。", len(self._policies))

    def _default_policy(self) -> QueryPolicy:
        """最終 fallback policy（news_generic）。"""
        return _make_policy(Intent.NEWS_DIGEST, "news_generic")


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
