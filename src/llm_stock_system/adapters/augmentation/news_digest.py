"""news_digest.py — NEWS_DIGEST intent 的 augmentation strategy

新聞摘要類查詢以敘述性內容為主，結構化資料（若有）僅作背景補充。
主要工作是：
  1. 把 narrative evidence 按日期排序，整理成清晰的時序列表
  2. 把少量的背景結構化資料（股價、PE）附在後面
  3. 產生清晰的 intent_frame 讓 LLM 知道重點是「事件摘要」而非數字分析
"""
from __future__ import annotations

from llm_stock_system.core.models import AugmentedContext, GovernanceReport, StructuredQuery
from llm_stock_system.adapters.augmentation.base import (
    AugmentationStrategy,
    label_for,
    split_evidence,
)


class NewsDigestAugmentation(AugmentationStrategy):
    EXPECTED_STRUCTURED_TYPES = []  # 新聞類不強制要求結構化資料

    def build(self, query: StructuredQuery, report: GovernanceReport) -> AugmentedContext:
        label = label_for(query)
        structured, narrative = split_evidence(report.evidence)

        # 新聞按日期由新到舊排列
        narrative_sorted = sorted(narrative, key=lambda e: e.published_at, reverse=True)
        narrative_texts = self._narrative_texts(narrative_sorted)

        # 背景資料（股價 / PE）整合成簡短一段
        bg_lines: list[str] = []
        for e in structured:
            content = e.full_content or e.excerpt
            if content:
                bg_lines.append(f"• {e.title}：{content}")
        structured_block = "\n".join(bg_lines) if bg_lines else ""

        intent_frame = (
            f"請根據以下近期新聞與公告，摘要 {label} 目前市場關注的主要事件與情緒。"
            "重點在於「發生了什麼」與「市場如何解讀」，不需要推導財務結論。"
        )

        return AugmentedContext(
            intent_frame=intent_frame,
            structured_block=structured_block,
            narrative_texts=narrative_texts,
            data_gaps=self._data_gaps(report.evidence),
        )
