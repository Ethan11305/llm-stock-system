"""valuation_check.py — VALUATION_CHECK intent 的 augmentation strategy

估值類查詢的核心是：
  1. 當前 PE 在歷史分位的位置（偏高 / 合理 / 偏低）
  2. 與同業比較
  3. 當前股價的背景（區間、均價）

把三種 PE 文件（pe_current / pe_history / pe_assessment）整合成
一張清晰的估值快照表，讓 LLM 不需要跨文件推導。
"""
from __future__ import annotations

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery
from llm_stock_system.adapters.augmentation.base import (
    AugmentationStrategy,
    label_for,
    split_evidence,
)

_PE_TYPES = {"pe_current", "pe_history", "pe_assessment"}
_PRICE_TYPES = {"market_data", "market_data_summary"}


class ValuationCheckAugmentation(AugmentationStrategy):
    EXPECTED_STRUCTURED_TYPES = ["pe_current", "pe_history"]

    def build(self, query: StructuredQuery, report: GovernanceReport) -> AugmentedContext:
        label = label_for(query)
        structured, narrative = split_evidence(report.evidence)

        pe_docs = [e for e in structured if e.source_type in _PE_TYPES]
        price_docs = [e for e in structured if e.source_type in _PRICE_TYPES]
        other_docs = [e for e in structured if e.source_type not in _PE_TYPES | _PRICE_TYPES]

        lines: list[str] = []

        # PE 估值快照
        if pe_docs:
            lines.append(f"【{label} 估值快照】")
            # 排序：pe_current → pe_history → pe_assessment
            order = {"pe_current": 0, "pe_history": 1, "pe_assessment": 2}
            pe_docs.sort(key=lambda e: order.get(e.source_type, 9))
            for e in pe_docs:
                lines.append(f"\n▌{e.title}")
                lines.append(e.full_content or e.excerpt)

        # 股價背景
        if price_docs:
            lines.append(f"\n【近期股價】")
            for e in price_docs:
                lines.append(f"• {e.title}：{e.full_content or e.excerpt}")

        # 其他背景（如財報）
        if other_docs:
            lines.append(f"\n【背景參考】")
            for e in other_docs:
                lines.append(f"• {e.title}：{e.full_content or e.excerpt}")

        structured_block = "\n".join(lines) if lines else ""

        intent_frame = (
            f"請根據以下估值資料，評估 {label} 目前本益比相對歷史區間的位置（偏高 / 合理 / 偏低），"
            "並說明與同業的差距。若有股價資料，請結合說明估值是否具吸引力。"
            "不要推測未來股價，聚焦在現有數字的客觀比較。"
        )

        return AugmentedContext(
            intent_frame=intent_frame,
            structured_block=structured_block,
            narrative_texts=self._narrative_texts(narrative),
            data_gaps=self._data_gaps(report.evidence),
        )
