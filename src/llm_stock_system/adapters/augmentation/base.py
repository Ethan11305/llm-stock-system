"""base.py — AugmentationStrategy 抽象基礎與共用工具函式

所有 intent-specific strategy 都繼承 AugmentationStrategy，
並實作 build(query, governance_report) → AugmentedContext。

共用工具：
  - split_evidence()：依 source_type 拆分結構化資料與新聞
  - format_data_gaps()：列出缺失的結構化資料類型
  - label_for()：取公司顯示名稱
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from llm_stock_system.core.models import AugmentedContext, Evidence, GovernanceReport, StructuredQuery

# source_type 對應到「可被結構化展示」的類型
STRUCTURED_SOURCE_TYPES: frozenset[str] = frozenset({
    "pe_current",
    "pe_history",
    "pe_assessment",
    "financial_statement",
    "financial_statement_breakdown",
    "financial_statement_latest",
    "dividend_policy",
    "dividend_analysis",
    "market_data",
    "market_data_summary",
    "technical_indicator",
    "margin_data",
    "monthly_revenue",
    "monthly_revenue_yoy",
    "monthly_revenue_mom",
    "revenue_growth",
    "profitability_stability",
    "gross_margin_comparison",
    "fcf_dividend",
    "debt_dividend",
    "ex_dividend",
    "dividend_yield",
    "balance_sheet",
    "cash_flow",
    "eps",
})

NARRATIVE_SOURCE_TYPES: frozenset[str] = frozenset({
    "news_article",
    "news",
    "announcement",
    "guidance_reaction",
    "shipping_rate_impact",
    "electricity_cost_impact",
    "macro_yield_sentiment",
    "theme_impact",
    "listing_revenue",
    "market_summary",
})


def split_evidence(
    evidence: list[Evidence],
) -> tuple[list[Evidence], list[Evidence]]:
    """將 evidence 拆成 (structured, narrative) 兩組。

    source_type 在 STRUCTURED_SOURCE_TYPES 的視為結構化資料；
    其餘（含 NARRATIVE_SOURCE_TYPES 與未知類型）視為敘述性文字。
    """
    structured: list[Evidence] = []
    narrative: list[Evidence] = []
    for e in evidence:
        if e.source_type in STRUCTURED_SOURCE_TYPES:
            structured.append(e)
        else:
            narrative.append(e)
    return structured, narrative


def label_for(query: StructuredQuery) -> str:
    return query.company_name or query.ticker or "此標的"


def format_data_gaps(
    expected_types: list[str],
    evidence: list[Evidence],
) -> list[str]:
    """列出 expected_types 中缺少的 source_type，轉換成人類可讀的缺口說明。"""
    present_types = {e.source_type for e in evidence}
    _TYPE_LABELS: dict[str, str] = {
        "pe_current": "本益比（當前）",
        "pe_history": "本益比歷史區間",
        "financial_statement": "財務報表",
        "financial_statement_latest": "最新財報",
        "dividend_policy": "股利政策",
        "dividend_analysis": "股利分析",
        "market_data": "股價資料",
        "market_data_summary": "股價摘要",
        "technical_indicator": "技術指標",
        "margin_data": "融資融券",
        "monthly_revenue": "月營收",
        "profitability_stability": "獲利穩定性",
        "gross_margin_comparison": "毛利率比較",
        "fcf_dividend": "自由現金流股利覆蓋",
        "debt_dividend": "負債股利安全度",
        "eps": "EPS",
    }
    gaps = []
    for t in expected_types:
        if t not in present_types:
            label = _TYPE_LABELS.get(t, t)
            gaps.append(f"{label}資料未取得，相關數字無法確認")
    return gaps


class AugmentationStrategy(ABC):
    """所有 augmentation strategy 的抽象基類。"""

    # 子類宣告此 intent 期望的結構化 source_type（用於 data_gaps 計算）
    EXPECTED_STRUCTURED_TYPES: ClassVar[list[str]] = []

    @abstractmethod
    def build(
        self,
        query: StructuredQuery,
        report: GovernanceReport,
    ) -> AugmentedContext:
        ...

    def _data_gaps(self, evidence: list[Evidence]) -> list[str]:
        return format_data_gaps(self.EXPECTED_STRUCTURED_TYPES, evidence)

    def _narrative_texts(self, narrative: list[Evidence]) -> list[str]:
        """從 narrative evidence 取 full_content（或 excerpt fallback）。"""
        texts = []
        for e in narrative:
            text = e.full_content if e.full_content else e.excerpt
            if text:
                texts.append(f"[{e.published_at:%Y-%m-%d}｜{e.source_name}] {e.title}\n{text}")
        return texts
