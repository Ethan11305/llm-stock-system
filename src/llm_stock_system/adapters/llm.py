from llm_stock_system.core.enums import Intent
from llm_stock_system.core.interfaces import LLMClient
from llm_stock_system.core.models import AnswerDraft, GovernanceReport, SourceCitation, StructuredQuery

from .synthesis.registry import get_strategy


class RuleBasedSynthesisClient(LLMClient):
    """Rule-based synthesizer. Routes by intent → IntentStrategy, then topic_tags."""

    def synthesize(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        system_prompt: str,
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

        highlights = [
            item.excerpt[:120] if item.excerpt and len(item.excerpt) > 20
            else f"{item.title}（{item.source_name}）"
            for item in governance_report.evidence[:3]
        ]
        facts = [
            f"{item.source_name} 於 {item.published_at:%Y-%m-%d} 提供資料：{item.excerpt}"
            for item in governance_report.evidence[:3]
        ]

        if query.intent in {Intent.EARNINGS_REVIEW, Intent.FINANCIAL_HEALTH, Intent.TECHNICAL_VIEW}:
            highlights = [
                item.excerpt[:120] if item.excerpt else f"{item.title}（{item.source_name}）"
                for item in governance_report.evidence[:3]
            ]

        strategy = get_strategy(query.intent)
        risks = strategy.build_risks(query, governance_report)
        if governance_report.high_trust_ratio < 0.5:
            risks.append("目前高可信來源占比不高，建議回看原文再做判斷。")

        return AnswerDraft(
            summary=strategy.build_summary(query, governance_report),
            highlights=highlights,
            facts=facts,
            impacts=strategy.build_impacts(query),
            risks=risks[:4],
            sources=sources,
        )
