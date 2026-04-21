"""Dividend analysis strategy.

Wave 3 sunset：除息填息 / 自由現金流永續性 / 負債支應股利 三條 deep-dive
summarizer 已全部下架。現在只保留最基本的「現金股利 + 收盤價 → 殖利率」整理。
更深入的股利觀察已完整整併到 digest 路徑，不再於 synthesis layer 重做一次。
"""

from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import extract_number


class DividendAnalysisStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        label = query.company_name or query.ticker or "此標的"
        return self._summarize_dividend_yield(label, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        return [
            "股利政策可反映公司對現金回饋的安排。",
            "現金殖利率可協助估算以目前股價換算的現金回饋水準。",
            "殖利率仍會隨股價變動，不是固定值。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        return [
            "股利最終內容仍需以董事會、股東會或公司正式公告為準。",
            "現金殖利率會隨股價變動，查詢當下與實際買入時點可能不同。",
            "若公司後續更新股利政策，現有換算結果需要重新檢查。",
        ]

    # ── private sub-summarizer ─────────────────────────────────────────

    def _summarize_dividend_yield(self, label: str, report: GovernanceReport) -> str:
        cash_dividend = extract_number(report, r"現金股利(?:合計)?約 (\d+(?:\.\d+)?) 元")
        close_price = extract_number(report, r"最新收盤價約 (\d+(?:\.\d+)?) 元")
        yield_pct = extract_number(report, r"現金殖利率約 (\d+(?:\.\d+)?)%")
        if cash_dividend and close_price and yield_pct:
            return (
                f"{label} 最新可取得的現金股利約 {cash_dividend} 元；"
                f"若以最新收盤價約 {close_price} 元換算，現金殖利率約 {yield_pct}%。"
            )
        if cash_dividend:
            return f"{label} 最新可取得的現金股利約 {cash_dividend} 元，但目前不足以穩定換算殖利率。"
        return f"{label} 目前已有部分股利資料，但不足以完整確認最新配息政策與現金殖利率。"
