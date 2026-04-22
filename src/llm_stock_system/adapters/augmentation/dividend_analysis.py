"""dividend_analysis.py — DIVIDEND_ANALYSIS intent 的 augmentation strategy

股利類查詢的核心是：
  1. 歷史股利金額與殖利率
  2. 現金流 / 負債是否支撐得起配息（FCF 覆蓋率 / 債務安全度）
  3. 除息 / 填息紀錄

策略：把股利政策、殖利率、FCF、債務這幾類資料整合成「股利安全度快照」。
"""
from __future__ import annotations

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery
from llm_stock_system.adapters.augmentation.base import (
    AugmentationStrategy,
    label_for,
    split_evidence,
)

_DIVIDEND_TYPES = {
    "dividend_policy", "dividend_analysis",
    "dividend_yield", "ex_dividend",
    "fcf_dividend", "debt_dividend",
    "cash_flow", "balance_sheet",
}


class DividendAnalysisAugmentation(AugmentationStrategy):
    EXPECTED_STRUCTURED_TYPES = ["dividend_policy", "dividend_yield"]

    def build(self, query: StructuredQuery, report: GovernanceReport) -> AugmentedContext:
        label = label_for(query)
        structured, narrative = split_evidence(report.evidence)

        div_docs = [e for e in structured if e.source_type in _DIVIDEND_TYPES]
        other_docs = [e for e in structured if e.source_type not in _DIVIDEND_TYPES]

        lines: list[str] = [f"【{label} 股利安全度快照】\n"]

        priority = {
            "dividend_yield": 0,
            "dividend_policy": 1,
            "dividend_analysis": 2,
            "ex_dividend": 3,
            "fcf_dividend": 4,
            "debt_dividend": 5,
            "cash_flow": 6,
            "balance_sheet": 7,
        }
        for e in sorted(div_docs, key=lambda e: (priority.get(e.source_type, 9), -e.published_at.timestamp())):
            lines.append(f"▌{e.title}（{e.published_at:%Y-%m}）")
            lines.append(e.full_content or e.excerpt)
            lines.append("")

        if other_docs:
            lines.append("【背景資料】")
            for e in other_docs:
                lines.append(f"• {e.title}：{e.full_content or e.excerpt}")

        structured_block = "\n".join(lines) if len(lines) > 1 else ""

        intent_frame = (
            f"請根據以下股利資料，分析 {label} 的殖利率水準、股利配發趨勢與可持續性。"
            "若有自由現金流或負債資料，請評估配息是否有財務支撐。"
            "明確說明近幾年現金股利金額、殖利率，以及任何配息風險。"
        )

        return AugmentedContext(
            intent_frame=intent_frame,
            structured_block=structured_block,
            narrative_texts=self._narrative_texts(narrative),
            data_gaps=self._data_gaps(report.evidence),
        )
