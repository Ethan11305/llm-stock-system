import re

from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import (
    build_impacts_generic,
    build_risks_generic,
    build_summary_fallback,
    evidence_contains,
    extract_number,
    extract_text,
    extract_company_margin,
)


class FinancialHealthStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if ("GROSS_MARGIN" in tags or "毛利率" in tags) and (
            query.comparison_ticker or query.comparison_company_name
        ):
            return self._summarize_gross_margin_comparison(label, query, report)
        if "獲利" in tags or "穩定性" in tags:
            return self._summarize_profitability_stability(label, report)
        if "成長" in tags or "AI" in tags or "伺服器" in tags:
            return self._summarize_revenue_growth(label, report)

        return build_summary_fallback(query, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if query.comparison_ticker or query.comparison_company_name:
            return [
                "毛利率較高的一方，通常代表定價能力、貨源結構或成本控管相對較占優勢。",
                "若兩家公司的毛利率差距能在近幾季持續，較容易反映為結構性差異，不只是單季波動。",
                "不過「經營效率」不能只看毛利率，仍要搭配營益率、費用率和資產週轉一起觀察。",
            ]
        if "獲利" in tags or "穩定性" in tags:
            return [
                "拿近五年的年度獲利一起看，比單看最新一年更能判斷這檔股票是否適合當成長期現金流型部位觀察。",
                "若近年出現轉虧，就不適合直接視為「穩定獲利」公司，後續要再看是週期性還是結構性問題。",
                "若處於轉虧年，能再拆出是本業轉弱還是業外拖累，會比只看 EPS 更有參考價值。",
            ]
        if "成長" in tags or "AI" in tags:
            return [
                "AI 伺服器營收占比屬於基本面結構資訊，應優先看公司說法、法說與高可信新聞整理。",
                "2026 年成長預測通常來自管理層展望、產能擴張與產業需求判讀，不應直接當成已實現結果。",
                "若只有成長敘事而沒有一致數字，較適合解讀為方向性訊號，而不是精確預估。",
            ]
        return build_impacts_generic()

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        tags = set(query.topic_tags)
        if query.comparison_ticker or query.comparison_company_name:
            return [
                "毛利率較高不一定代表整體經營效率就更好，仍要看營益率、費用控管與資產週轉。",
                "航運股毛利率容易受運價週期、燃油成本與航線結構影響，單一季數字可能波動很大。",
                "若比較時點不完全一致，或財報口徑剛好受一次性因素影響，解讀上要保守。",
            ]
        if "獲利" in tags or "穩定性" in tags:
            return [
                "就算近幾年都有獲利，若獲利幅度起伏很大，對「退休存股」的穩定性仍要保守看待。",
                "轉虧年的原因若只能從財報結構推估，仍應與公司法說或年報說明互相對照。",
                "傳統產業獲利容易受景氣、原料報價和利差影響，過去五年不一定能直接代表未來五年。",
            ]
        if "成長" in tags or "AI" in tags:
            return [
                "營收占比若不是公司正式揭露，常來自媒體或法人估算，可能隨季度而變動。",
                "2026 年成長預測具有前瞻不確定性，容易受雲端資本支出、客戶拉貨節奏與供應鏈瓶頸影響。",
                "若現有證據主要是新聞敘事而非公司明確數字，對成長幅度的解讀應保守。",
            ]
        return build_risks_generic(query)

    # ── private sub-summarizers ─────────────────────────────────────────

    def _summarize_gross_margin_comparison(
        self, label: str, query: StructuredQuery, report: GovernanceReport
    ) -> str:
        primary_label = query.company_name or query.ticker or "第一家公司"
        comparison_label = query.comparison_company_name or query.comparison_ticker or "第二家公司"
        primary_margin = extract_company_margin(report, primary_label)
        comparison_margin = extract_company_margin(report, comparison_label)
        higher_company = extract_text(
            report,
            rf"由\s*({re.escape(primary_label)}|{re.escape(comparison_label)})\s*較高",
        )
        margin_gap = extract_number(report, r"高出 (\d+(?:\.\d+)?) 個百分點")
        if primary_margin and comparison_margin:
            resolved_higher = higher_company or (
                primary_label if float(primary_margin) >= float(comparison_margin) else comparison_label
            )
            gap_segment = f"，高出 {margin_gap} 個百分點" if margin_gap is not None else ""
            return (
                f"若以最新可比財報口徑比較，"
                f"{primary_label} 毛利率約 {primary_margin}%，"
                f"{comparison_label} 毛利率約 {comparison_margin}%，"
                f"由 {resolved_higher} 較高{gap_segment}。"
                f"單看毛利率，{resolved_higher} 在定價能力或成本結構上相對較占優勢，"
                "但不能單憑毛利率就等同整體經營效率更好。"
            )
        return (
            f"資料不足，無法確認{primary_label}與{comparison_label}"
            "最新可比財報口徑下的毛利率差異與經營效率評估。"
        )

    def _summarize_profitability_stability(self, label: str, report: GovernanceReport) -> str:
        stability_sentence = extract_text(
            report,
            r"(近五年[^。]*(?:穩定獲利|波動[^。]*|並非每年都有穩定獲利)[^。]*。)",
        )
        loss_year = extract_number(report, r"在 (\d{4}) 年歸屬母公司淨損約")
        loss_amount = extract_number(report, r"淨損約 (-?\d+(?:\.\d+)?) 億元")
        loss_reason = extract_text(report, r"(若只看財報結構推估[^。]*。)")
        if stability_sentence and loss_year and loss_amount:
            tail = f"{loss_reason}" if loss_reason else ""
            normalized = str(abs(float(loss_amount)))
            return (
                f"{stability_sentence}{loss_year} 年是較明顯的虧損年度，"
                f"歸屬母公司淨損約 {normalized} 億元。{tail}"
            )
        if stability_sentence:
            return stability_sentence
        return f"資料不足，無法確認{label}過去五年的獲利穩定性與轉虧年度原因。"

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
