"""financial_health.py — FINANCIAL_HEALTH intent 的 augmentation strategy

財務健康度查詢聚焦在多期趨勢：
  - 毛利率 / 營業利益率是否穩定或改善
  - 營收成長動能
  - 若有比較 ticker，需要呈現雙方的可比數字

策略：把多期財報、毛利率比較、獲利穩定性資料整合成時序趨勢表。
"""
from __future__ import annotations

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery
from llm_stock_system.adapters.augmentation.base import (
    AugmentationStrategy,
    label_for,
    split_evidence,
)

_HEALTH_TYPES = {
    "financial_statement",
    "financial_statement_breakdown",
    "financial_statement_latest",
    "profitability_stability",
    "gross_margin_comparison",
    "revenue_growth",
    "eps",
    "monthly_revenue",
    "monthly_revenue_yoy",
}


class FinancialHealthAugmentation(AugmentationStrategy):
    EXPECTED_STRUCTURED_TYPES = [
        "financial_statement_latest",
        "profitability_stability",
    ]

    def build(self, query: StructuredQuery, report: GovernanceReport) -> AugmentedContext:
        label = label_for(query)
        comparison_label = query.comparison_company_name or query.comparison_ticker
        structured, narrative = split_evidence(report.evidence)

        health_docs = [e for e in structured if e.source_type in _HEALTH_TYPES]
        other_docs = [e for e in structured if e.source_type not in _HEALTH_TYPES]

        lines: list[str] = [f"【{label} 財務健康度資料】\n"]

        if health_docs:
            priority = {
                "profitability_stability": 0,
                "gross_margin_comparison": 1,
                "financial_statement_latest": 2,
                "financial_statement": 3,
                "financial_statement_breakdown": 3,
                "revenue_growth": 4,
                "eps": 5,
                "monthly_revenue_yoy": 6,
                "monthly_revenue": 7,
            }
            for e in sorted(health_docs, key=lambda e: (priority.get(e.source_type, 9), -e.published_at.timestamp())):
                lines.append(f"▌{e.title}（{e.published_at:%Y-%m}）")
                lines.append(e.full_content or e.excerpt)
                lines.append("")

        if other_docs:
            lines.append("【背景資料】")
            for e in other_docs:
                lines.append(f"• {e.title}：{e.full_content or e.excerpt}")

        structured_block = "\n".join(lines) if len(lines) > 1 else ""

        if comparison_label:
            intent_frame = (
                f"請根據以下財務資料，比較 {label} 與 {comparison_label} 的獲利品質、"
                "毛利率趨勢與營收成長。以最新可比期間為主，說明誰的財務健康度較佳及原因。"
            )
        else:
            intent_frame = (
                f"請根據以下多期財務資料，評估 {label} 的獲利品質與財務健康度。"
                "重點分析：毛利率趨勢（是否改善）、EPS 穩定性、營收成長動能。"
                "若有虧損年度，請指出並說明可能原因。"
            )

        return AugmentedContext(
            intent_frame=intent_frame,
            structured_block=structured_block,
            narrative_texts=self._narrative_texts(narrative),
            data_gaps=self._data_gaps(report.evidence),
        )
