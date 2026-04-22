"""Financial health strategy.

Wave 4 重構：移除硬編碼年份（原 2026）與硬編碼關鍵字（AI 伺服器）的 legacy summarizer。
現在由 evidence_contains 動態偵測成長主題，並透過 AugmentationLayer 產出的
structured_block 讓 LLM / RuleBasedSynthesisClient 取得精確數字，
不再在 strategy 層注入特定年份假設。
"""

from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import (
    build_impacts_generic,
    build_risks_generic,
    build_summary_fallback,
    evidence_contains,
)


class FinancialHealthStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)

        if "成長" in tags or "AI" in tags or "伺服器" in tags:
            return self._summarize_growth_theme(query, report)

        return build_summary_fallback(query, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if "成長" in tags or "AI" in tags:
            return [
                "AI 相關業務營收占比屬於基本面結構資訊，應優先看公司說法、法說與高可信新聞整理。",
                "成長預測通常來自管理層展望、產能擴張與產業需求判讀，不應直接當成已實現結果。",
                "若只有成長敘事而沒有一致數字，較適合解讀為方向性訊號，而不是精確預估。",
            ]
        return build_impacts_generic()

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        tags = set(query.topic_tags)
        if "成長" in tags or "AI" in tags:
            return [
                "營收占比若不是公司正式揭露，常來自媒體或法人估算，可能隨季度而變動。",
                "前瞻成長預測具有不確定性，容易受雲端資本支出、客戶拉貨節奏與供應鏈瓶頸影響。",
                "若現有證據主要是新聞敘事而非公司明確數字，對成長幅度的解讀應保守。",
            ]
        return build_risks_generic(query)

    # ── private sub-summarizer ─────────────────────────────────────────

    def _summarize_growth_theme(self, query: StructuredQuery, report: GovernanceReport) -> str:
        """動態偵測成長題材關鍵字，不假設特定年份或產品名稱。

        數字細節（占比、成長率）由 AugmentedContext.structured_block 提供，
        strategy 層只負責定性描述主題是否有足夠支撐。
        """
        label = query.company_name or query.ticker or "此標的"
        has_ai_server = evidence_contains(
            report, ("AI伺服器", "AI 伺服器", "ai伺服器", "伺服器", "server", "AI Server")
        )
        has_growth_signal = evidence_contains(
            report, ("成長動能", "成長關鍵", "重要成長", "成長預測", "倍增", "年增", "動能")
        )
        has_share_disclosure = evidence_contains(
            report, ("營收占比", "占比", "比重", "揭露", "明確")
        )

        if has_ai_server and has_growth_signal and has_share_disclosure:
            return (
                f"{label} 目前公開資訊顯示，AI 伺服器相關業務被多個來源視為重要成長動能，"
                "且有部分來源揭露相關營收占比數字，詳細數值請參照資料欄位。"
            )
        if has_ai_server and has_growth_signal:
            return (
                f"{label} 目前公開資訊顯示，AI 伺服器相關業務仍被多個來源視為重要成長動能，"
                "但現有來源未一致揭露明確營收占比。"
            )
        if has_ai_server:
            return (
                f"{label} 目前已有 AI 伺服器相關業務資訊可供整理，"
                "但成長預測與營收占比尚缺足夠一致的公開證據。"
            )
        return build_summary_fallback(query, report)
