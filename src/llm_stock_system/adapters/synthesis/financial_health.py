"""Financial health strategy.

Wave 3 sunset：兩條 deep-dive 已下架：
  * 毛利率比較（_summarize_gross_margin_comparison）
  * 獲利穩定性（_summarize_profitability_stability）
只保留 AI 伺服器 / 成長敘事這條仍然活著的簡單摘要，其餘走 fallback summary。
"""

from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import (
    build_impacts_generic,
    build_risks_generic,
    build_summary_fallback,
    evidence_contains,
    extract_number,
)


class FinancialHealthStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if "成長" in tags or "AI" in tags or "伺服器" in tags:
            return self._summarize_revenue_growth(label, report)

        return build_summary_fallback(query, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if "成長" in tags or "AI" in tags:
            return [
                "AI 伺服器營收占比屬於基本面結構資訊，應優先看公司說法、法說與高可信新聞整理。",
                "2026 年成長預測通常來自管理層展望、產能擴張與產業需求判讀，不應直接當成已實現結果。",
                "若只有成長敘事而沒有一致數字，較適合解讀為方向性訊號，而不是精確預估。",
            ]
        return build_impacts_generic()

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        tags = set(query.topic_tags)
        if "成長" in tags or "AI" in tags:
            return [
                "營收占比若不是公司正式揭露，常來自媒體或法人估算，可能隨季度而變動。",
                "2026 年成長預測具有前瞻不確定性，容易受雲端資本支出、客戶拉貨節奏與供應鏈瓶頸影響。",
                "若現有證據主要是新聞敘事而非公司明確數字，對成長幅度的解讀應保守。",
            ]
        return build_risks_generic(query)

    # ── private sub-summarizer ─────────────────────────────────────────

    def _summarize_revenue_growth(self, label: str, report: GovernanceReport) -> str:
        share_value = extract_number(report, r"(\d+(?:\.\d+)?)%")
        has_ai_server = evidence_contains(report, ("ai伺服器", "ai 伺服器", "伺服器", "server"))
        has_growth_2026 = evidence_contains(report, ("2026", "倍增", "成長", "動能"))
        if has_ai_server and has_growth_2026 and share_value:
            return (
                f"{label} 目前公開資訊顯示，AI 伺服器相關營收比重約 {share_value}%，"
                "且 2026 年成長仍被多個來源視為重要動能。"
            )
        if has_ai_server and has_growth_2026:
            return (
                f"{label} 目前公開資訊顯示，AI 伺服器仍被視為 2026 年的重要成長動能，"
                "但現有來源未一致揭露明確營收占比。"
            )
        if has_ai_server:
            return f"{label} 目前已有 AI 伺服器相關資訊可供整理，但對 2026 年成長預測與營收占比仍缺乏足夠一致的公開證據。"
        return f"資料不足，無法確認{label}AI 伺服器營收占比與 2026 年成長預測。"
