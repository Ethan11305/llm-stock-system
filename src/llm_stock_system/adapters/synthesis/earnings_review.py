"""Earnings review strategy.

Wave 3 sunset：毛利率轉正 deep-dive（_summarize_margin_turnaround）已下架。
保留的兩條路徑：
  * 月營收年增（_summarize_monthly_revenue_yoy）— TWSE 月營收資料是唯一的
    真實來源，這條仍是 synthesis layer 的核心 fallback。
  * EPS / 股利（_summarize_earnings_eps）— 當財報 tag 明確時仍會觸發。
其餘 fallback 交由 build_summary 最後幾行處理。
"""

from llm_stock_system.core.enums import TopicTag
from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import extract_number, extract_text


class EarningsReviewStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if TopicTag.REVENUE.value in tags or "月增率" in tags or "年增率" in tags:
            return self._summarize_monthly_revenue_yoy(label, report)
        if "EPS" in tags or "財報" in tags:
            return self._summarize_earnings_eps(label, report)

        annual_eps = extract_number(report, r"全年 EPS 約 (\d+(?:\.\d+)?) 元")
        latest_quarter_eps = extract_number(report, r"最新一季 EPS 約 (\d+(?:\.\d+)?) 元")
        if annual_eps:
            return f"{label} 目前可整理到的財報重點顯示，去年全年 EPS 約 {annual_eps} 元。"
        if latest_quarter_eps:
            return f"{label} 目前可整理到的最新一季財報重點顯示， EPS 約 {latest_quarter_eps} 元。"
        return f"{label} 目前有財報相關資料可供整理，但細節仍應回看原始揭露內容。"

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.REVENUE.value in tags or "月增率" in tags or "年增率" in tags:
            return [
                "月營收累計年增率可以快速觀察公司當年開局的營運動能。",
                "拿今年前幾個月與去年同期比較，能避免單看單月數字受基期或出貨時點影響。",
                "營收年增可以當作基本面熱度的先行指標，但仍需搭配毛利率與 EPS 才能完整解讀。",
            ]
        return [
            "EPS 表現可反映公司獲利能力與資本支出承受度。",
            "股利資訊常影響市場對現金回饋與殖利率的預期。",
            "若財報與股利方向一致，通常更容易形成穩定敘事。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.REVENUE.value in tags or "月增率" in tags:
            return [
                "累計營收年增只反映營收端的變化，不直接代表毛利率或獲利同步改善。",
                "單看前幾個月的累計營收，仍可能受出貨時點、匯率或季節性因素影響。",
                "若沒有同時搭配財報與法說，容易過度把營收成長解讀為獲利確定性。",
            ]
        return [
            "股利最終內容仍需以董事會、股東會或公司正式公告為準。",
            "歷史 EPS 不能直接保證未來獲利延續。",
            "若市場預期主要來自單一新聞來源，解讀上要保留彈性。",
        ]

    # ── private sub-summarizers ─────────────────────────────────────────

    def _summarize_monthly_revenue_yoy(self, label: str, report: GovernanceReport) -> str:
        availability_sentence = extract_text(
            report,
            r"(截至 \d{4}-\d{2}-\d{2}，官方月營收資料最新僅到 \d{4}-\d{2}，尚未公布 \d{4}-\d{2} 月營收。[^。]*。)",
        )
        if availability_sentence:
            return availability_sentence

        month_revenue = extract_number(report, r"單月營收約 (\d+(?:\.\d+)?) 億元")
        mom_pct = extract_number(report, r"月增率約 (-?\d+(?:\.\d+)?)%")
        yoy_pct = extract_number(report, r"年增率約 (-?\d+(?:\.\d+)?)%")
        mom_status = extract_text(report, r"(月增率已超過 20%|月增率未達 20%)")
        high_status = extract_text(
            report,
            r"(創下近一年新高|尚未創下近一年新高|近一年月營收歷史不足[^。]*。)",
        )
        market_view = extract_text(report, r"(市場解讀：[^。]*。)")

        if month_revenue is not None and mom_pct is not None:
            parts = [f"{label} 最新已取得的單月營收約 {month_revenue} 億元，月增率約 {mom_pct}%。"]
            if mom_status:
                parts.append(mom_status + "。")
            if yoy_pct is not None:
                parts.append(f"年增率約 {yoy_pct}%。")
            if high_status:
                parts.append(high_status if high_status.endswith("。") else high_status + "。")
            if market_view:
                parts.append(market_view)
            return "".join(parts)

        cumulative_month_count = extract_number(report, r"\d{4} 年前 (\d+) 個月累計營收約")
        current_total = extract_number(
            report, r"\d{4} 年前 \d+ 個月累計營收約 (\d+(?:\.\d+)?) 億元"
        )
        previous_total = extract_number(report, r"\d{4} 年同期約 (\d+(?:\.\d+)?) 億元")
        cumulative_yoy_pct = extract_number(report, r"年增率約 (-?\d+(?:\.\d+)?)%")
        cumulative_availability = extract_text(
            report,
            r"(截至 \d{4}-\d{2}，官方月營收資料僅更新到今年前 \d+ 個月。)",
        )
        if current_total is not None and previous_total is not None and cumulative_yoy_pct is not None:
            leading = cumulative_availability or ""
            month_label = int(cumulative_month_count) if cumulative_month_count is not None else 3
            return (
                f"{leading}{label} 今年前 {month_label} 個月累計營收約 {current_total} 億元，"
                f"去年同期約 {previous_total} 億元，年增率約 {cumulative_yoy_pct}%。"
            )
        return f"資料不足，無法確認{label}的月營收變化、單月月增率或與去年同期相比的成長幅度。"

    def _summarize_earnings_eps(self, label: str, report: GovernanceReport) -> str:
        annual_eps = extract_number(report, r"全年 EPS 約 (\d+(?:\.\d+)?) 元")
        latest_quarter_eps = extract_number(report, r"最新一季 EPS 約 (\d+(?:\.\d+)?) 元")
        cash_dividend = extract_number(report, r"現金股利(?:合計)?約 (\d+(?:\.\d+)?) 元")
        if annual_eps and cash_dividend:
            return (
                f"{label} 去年全年 EPS 約 {annual_eps} 元；"
                f"目前已取得的股利資料顯示，現金股利約 {cash_dividend} 元，"
                "市場預期仍需配合最新公告與新聞觀察。"
            )
        if latest_quarter_eps and cash_dividend:
            return (
                f"{label} 最新一季 EPS 約 {latest_quarter_eps} 元；"
                f"目前可取得的股利資料顯示，現金股利約 {cash_dividend} 元。"
            )
        if annual_eps:
            return f"{label} 去年全年 EPS 約 {annual_eps} 元，但股利預期仍需等待更多公告或新聞佐證。"
        return f"{label} 目前已有部分財報或股利資料，但仍不足以完整確認 EPS 與市場股利預期。"
