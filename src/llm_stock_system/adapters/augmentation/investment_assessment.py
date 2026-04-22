"""investment_assessment.py — INVESTMENT_ASSESSMENT intent 的 augmentation strategy

投資評估是最複雜的 intent，需要跨多個面向整合：
  估值（PE）+ 獲利（財報/EPS）+ 股利 + 近期事件（新聞）

策略：把所有結構化資料整理成一份「投資評估框架」，
分 4 個面向呈現，讓 LLM 可以有條理地推理，而不是從 8 篇雜亂的 excerpt 拼湊。
"""
from __future__ import annotations

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery
from llm_stock_system.adapters.augmentation.base import (
    AugmentationStrategy,
    label_for,
    split_evidence,
)

_PE_TYPES = {"pe_current", "pe_history", "pe_assessment"}
_EARNINGS_TYPES = {
    "financial_statement", "financial_statement_breakdown",
    "financial_statement_latest", "eps", "revenue_growth",
    "monthly_revenue", "monthly_revenue_yoy",
}
_DIVIDEND_TYPES = {
    "dividend_policy", "dividend_analysis", "dividend_yield",
    "ex_dividend", "fcf_dividend", "debt_dividend",
}
_PRICE_TYPES = {"market_data", "market_data_summary"}


class InvestmentAssessmentAugmentation(AugmentationStrategy):
    EXPECTED_STRUCTURED_TYPES = [
        "pe_current",
        "financial_statement_latest",
        "dividend_policy",
    ]

    def build(self, query: StructuredQuery, report: GovernanceReport) -> AugmentedContext:
        label = label_for(query)
        structured, narrative = split_evidence(report.evidence)

        pe_docs = [e for e in structured if e.source_type in _PE_TYPES]
        earnings_docs = [e for e in structured if e.source_type in _EARNINGS_TYPES]
        dividend_docs = [e for e in structured if e.source_type in _DIVIDEND_TYPES]
        price_docs = [e for e in structured if e.source_type in _PRICE_TYPES]

        lines: list[str] = [f"【{label} 投資評估框架】\n"]

        # 1. 估值面
        if pe_docs:
            lines.append("── 1. 估值 ──")
            order = {"pe_current": 0, "pe_history": 1, "pe_assessment": 2}
            for e in sorted(pe_docs, key=lambda e: order.get(e.source_type, 9)):
                lines.append(f"  {e.title}：{e.full_content or e.excerpt}")
        if price_docs:
            for e in price_docs:
                lines.append(f"  {e.title}：{e.full_content or e.excerpt}")

        # 2. 獲利面
        if earnings_docs:
            lines.append("\n── 2. 獲利品質 ──")
            e_order = {
                "financial_statement_latest": 0,
                "financial_statement": 1,
                "eps": 2,
                "monthly_revenue_yoy": 3,
                "monthly_revenue": 4,
                "revenue_growth": 5,
            }
            for e in sorted(earnings_docs, key=lambda e: (e_order.get(e.source_type, 9), -e.published_at.timestamp())):
                lines.append(f"\n  ▌{e.title}（{e.published_at:%Y-%m}）")
                lines.append(f"  {e.full_content or e.excerpt}")

        # 3. 股利面
        if dividend_docs:
            lines.append("\n── 3. 股利 ──")
            for e in sorted(dividend_docs, key=lambda e: -e.published_at.timestamp()):
                lines.append(f"  {e.title}：{e.full_content or e.excerpt}")

        # 4. 近期事件（由新聞填充，在 narrative_texts 中呈現）
        if narrative:
            lines.append(f"\n── 4. 近期重要事件（{len(narrative)} 則）──")
            lines.append("  詳見下方新聞與公告清單。")

        structured_block = "\n".join(lines)

        # 列出缺失的面向
        gaps = self._data_gaps(report.evidence)
        if not pe_docs:
            gaps.insert(0, "本益比資料未取得，估值面向無法評估")
        if not earnings_docs:
            gaps.insert(0, "財報 / EPS 資料未取得，獲利品質無法評估")

        intent_frame = (
            f"請根據以下投資評估框架，從估值、獲利品質、股利與近期事件四個面向，"
            f"對 {label} 給出均衡的投資分析。"
            "每個面向要有具體數字支撐，缺乏資料的面向請明確說明「資料不足」。"
            "不要給出明確的買 / 賣建議，而是呈現各面向的客觀事實讓使用者自行判斷。"
        )

        return AugmentedContext(
            intent_frame=intent_frame,
            structured_block=structured_block,
            narrative_texts=self._narrative_texts(narrative),
            data_gaps=gaps,
        )
