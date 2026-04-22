from __future__ import annotations

from llm_stock_system.core.enums import Intent
from llm_stock_system.core.interfaces import LLMClient
from llm_stock_system.core.models import AnswerDraft, AugmentedContext, GovernanceReport, SourceCitation, StructuredQuery

from .synthesis.registry import get_strategy

# AugmentedContext 的 structured_block / narrative_texts 欄位若有內容，
# 從中萃取 highlights / facts / summary 的最大行數上限。
_MAX_STRUCTURED_FACTS = 4
_MAX_NARRATIVE_HIGHLIGHTS = 3


class RuleBasedSynthesisClient(LLMClient):
    """Rule-based synthesizer. Routes by intent → IntentStrategy, then topic_tags.

    當 augmented_context 存在時，優先從 structured_block 萃取 facts、
    從 narrative_texts 萃取 highlights，並以 intent_frame 作為 summary 的前置語境，
    修補 A→G fallback 路徑中 AugmentedContext 被忽略的問題。
    """

    def synthesize(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        system_prompt: str,
        augmented_context: AugmentedContext | None = None,
    ) -> AnswerDraft:
        _ = system_prompt

        if not governance_report.evidence:
            return AnswerDraft(
                summary="資料不足，無法確認。",
                highlights=["現有證據不足或一致性不足，系統已降級回答。"],
                facts=["尚未取得足夠的官方公告、新聞或財報資料。"],
                impacts=["資料不足時，不應將單一訊息直接解讀為趨勢。"],
                risks=[
                    "資料不足時，容易誤把單一訊息當成趨勢。",
                    "若只依賴未驗證資訊，可能放大判斷偏誤。",
                    "建議等待更多公告、財報或主流來源更新。",
                ],
                sources=[],
            )

        sources = [
            SourceCitation(
                title=item.title,
                source_name=item.source_name,
                source_tier=item.source_tier,
                url=item.url,
                published_at=item.published_at,
                excerpt=item.excerpt,
                support_score=item.support_score,
                corroboration_count=item.corroboration_count,
            )
            for item in governance_report.evidence
        ]

        strategy = get_strategy(query.intent)

        # ── 嘗試從 AugmentedContext 萃取更豐富的 highlights / facts ────────
        highlights, facts = self._extract_from_context(augmented_context, governance_report)

        # ── Summary：優先讓 strategy 依 intent / topic_tags 判斷 ───────────
        # strategy.build_summary 本身已能用 helpers 掃 full_content；
        # 若 intent_frame 存在，附加在前方提供分析框架語境。
        base_summary = strategy.build_summary(query, governance_report)
        summary = self._prefix_intent_frame(base_summary, augmented_context)

        risks = strategy.build_risks(query, governance_report)
        if governance_report.high_trust_ratio < 0.5:
            risks.append("目前高可信來源占比不高，建議回看原文再做判斷。")

        return AnswerDraft(
            summary=summary,
            highlights=highlights,
            facts=facts,
            impacts=strategy.build_impacts(query),
            risks=risks[:4],
            sources=sources,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _extract_from_context(
        self,
        ctx: AugmentedContext | None,
        report: GovernanceReport,
    ) -> tuple[list[str], list[str]]:
        """從 AugmentedContext 萃取 highlights（新聞語意）與 facts（結構化數字）。

        若 ctx 不存在或內容為空，退回原本的 excerpt-based 邏輯。
        """
        facts: list[str] = []
        highlights: list[str] = []

        if ctx is not None:
            # facts ← structured_block 的非空行（跳過標題行 【…】）
            if ctx.structured_block:
                for line in ctx.structured_block.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("【") and not stripped.startswith("#"):
                        facts.append(stripped)
                        if len(facts) >= _MAX_STRUCTURED_FACTS:
                            break

            # highlights ← narrative_texts 各篇的首句（取前 N 篇）
            if ctx.narrative_texts:
                for text in ctx.narrative_texts[:_MAX_NARRATIVE_HIGHLIGHTS]:
                    # 取第一個句點或換行之前的文字作為 highlight
                    first_line = text.splitlines()[0].strip() if text.strip() else ""
                    if first_line:
                        highlights.append(first_line[:160])

        # 若 augmented_context 沒提供足夠內容，補足至 3 條
        if len(highlights) < 3:
            for item in report.evidence[:3 - len(highlights)]:
                highlights.append(
                    item.excerpt[:120] if item.excerpt and len(item.excerpt) > 20
                    else f"{item.title}（{item.source_name}）"
                )

        if len(facts) < 3:
            for item in report.evidence[:3 - len(facts)]:
                facts.append(
                    f"{item.source_name} 於 {item.published_at:%Y-%m-%d} 提供資料：{item.excerpt}"
                )

        return highlights, facts

    @staticmethod
    def _prefix_intent_frame(summary: str, ctx: AugmentedContext | None) -> str:
        """若 intent_frame 存在且不與 summary 重複，附加在 summary 前方作為分析語境。"""
        if ctx is None or not ctx.intent_frame:
            return summary
        frame = ctx.intent_frame.strip()
        # 避免重複：若 summary 已包含 intent_frame 的核心片語，不再重複附加
        if frame[:20] in summary:
            return summary
        return f"{frame}\n{summary}"
