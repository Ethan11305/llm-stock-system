"""earnings_review.py — EARNINGS_REVIEW intent 的 augmentation strategy

財報類查詢的核心是「數字趨勢」，需要把分散在多篇 evidence 的
EPS、毛利率、月營收整合成一個可閱讀的表格，讓 LLM 不用自己拼湊。
"""
from __future__ import annotations

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery
from llm_stock_system.adapters.augmentation.base import (
    AugmentationStrategy,
    label_for,
    split_evidence,
)

# 財報類的核心結構化資料
_EARNINGS_STRUCTURED = {
    "financial_statement",
    "financial_statement_breakdown",
    "financial_statement_latest",
    "monthly_revenue",
    "monthly_revenue_yoy",
    "monthly_revenue_mom",
    "revenue_growth",
    "eps",
}


class EarningsReviewAugmentation(AugmentationStrategy):
    EXPECTED_STRUCTURED_TYPES = [
        "financial_statement_latest",
        "monthly_revenue",
        "eps",
    ]

    def build(self, query: StructuredQuery, report: GovernanceReport) -> AugmentedContext:
        label = label_for(query)
        structured, narrative = split_evidence(report.evidence)

        # 財報類結構化資料按優先順序排：財報 > 月營收 > EPS
        priority = {
            "financial_statement_latest": 0,
            "financial_statement": 1,
            "financial_statement_breakdown": 1,
            "eps": 2,
            "monthly_revenue_yoy": 3,
            "monthly_revenue": 3,
            "monthly_revenue_mom": 4,
            "revenue_growth": 5,
        }
        earnings_docs = [e for e in structured if e.source_type in _EARNINGS_STRUCTURED]
        earnings_docs.sort(key=lambda e: (priority.get(e.source_type, 99), -e.published_at.timestamp()))

        lines: list[str] = [f"【{label} 財報與營收資料】"]
        for e in earnings_docs:
            content = e.full_content or e.excerpt
            lines.append(f"\n▌{e.title}（{e.published_at:%Y-%m}）")
            lines.append(content)

        # 其他背景結構化資料（例如股價）
        other_docs = [e for e in structured if e.source_type not in _EARNINGS_STRUCTURED]
        if other_docs:
            lines.append("\n【背景資料】")
            for e in other_docs:
                lines.append(f"• {e.title}：{e.full_content or e.excerpt}")

        structured_block = "\n".join(lines) if len(lines) > 1 else ""

        intent_frame = (
            f"請根據以下財報與營收資料，分析 {label} 的獲利趨勢、EPS 水準與月營收動能。"
            "優先使用結構化數字進行推理，新聞僅作輔助驗證。"
            "若數字顯示趨勢，請明確說明是改善、持平還是惡化。"
        )

        return AugmentedContext(
            intent_frame=intent_frame,
            structured_block=structured_block,
            narrative_texts=self._narrative_texts(narrative),
            data_gaps=self._data_gaps(report.evidence),
        )
