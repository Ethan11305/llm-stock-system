"""technical_view.py — TECHNICAL_VIEW intent 的 augmentation strategy

技術面查詢的核心是：
  1. 價格相對均線的位置（MA5 / MA20 / MA60）
  2. 動能指標（RSI、KD、MACD）
  3. 若有籌碼 tag：融資餘額 / 融券餘額趨勢

策略：把 technical_indicator、market_data、margin_data 整合成
一份「技術面快照」，依序呈現均線位置 → 指標 → 籌碼。
"""
from __future__ import annotations

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery
from llm_stock_system.adapters.augmentation.base import (
    AugmentationStrategy,
    label_for,
    split_evidence,
)

_TECHNICAL_TYPES = {
    "technical_indicator",
    "market_data",
    "market_data_summary",
    "margin_data",
}


class TechnicalViewAugmentation(AugmentationStrategy):
    EXPECTED_STRUCTURED_TYPES = ["technical_indicator", "market_data_summary"]

    def build(self, query: StructuredQuery, report: GovernanceReport) -> AugmentedContext:
        label = label_for(query)
        structured, narrative = split_evidence(report.evidence)

        tech_docs = [e for e in structured if e.source_type in _TECHNICAL_TYPES]
        other_docs = [e for e in structured if e.source_type not in _TECHNICAL_TYPES]

        lines: list[str] = [f"【{label} 技術面快照】\n"]

        priority = {
            "technical_indicator": 0,
            "market_data_summary": 1,
            "market_data": 2,
            "margin_data": 3,
        }
        for e in sorted(tech_docs, key=lambda e: (priority.get(e.source_type, 9), -e.published_at.timestamp())):
            lines.append(f"▌{e.title}（{e.published_at:%Y-%m-%d}）")
            lines.append(e.full_content or e.excerpt)
            lines.append("")

        if other_docs:
            lines.append("【背景資料】")
            for e in other_docs:
                lines.append(f"• {e.title}：{e.full_content or e.excerpt}")

        structured_block = "\n".join(lines) if len(lines) > 1 else ""

        tags = set(query.topic_tags)
        if "籌碼" in tags or "季線" in tags:
            intent_frame = (
                f"請根據以下技術指標與籌碼資料，分析 {label} 目前的均線位置、"
                "動能指標狀態（RSI、KD、MACD）以及融資融券餘額趨勢。"
                "說明多空動能是否有明確訊號，並指出支撐與壓力區域（若資料足夠）。"
            )
        else:
            intent_frame = (
                f"請根據以下技術資料，分析 {label} 目前的股價趨勢與技術指標狀態。"
                "說明股價相對主要均線（MA5 / MA20 / MA60）的位置，"
                "以及 RSI / KD / MACD 是否有超買、超賣或交叉訊號。"
            )

        return AugmentedContext(
            intent_frame=intent_frame,
            structured_block=structured_block,
            narrative_texts=self._narrative_texts(narrative),
            data_gaps=self._data_gaps(report.evidence),
        )
