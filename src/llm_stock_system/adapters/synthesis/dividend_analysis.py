from llm_stock_system.core.enums import TopicTag
from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import extract_number, extract_text, extract_year_metric_pairs


class DividendAnalysisStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if "除息" in tags or "填息" in tags:
            return self._summarize_ex_dividend_performance(label, report)
        if TopicTag.CASH_FLOW.value in tags or "現金流" in tags:
            return self._summarize_fcf_dividend_sustainability(label, report)
        if TopicTag.DEBT.value in tags or "負債" in tags:
            return self._summarize_debt_dividend_safety(label, report)

        return self._summarize_dividend_yield(label, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.CASH_FLOW.value in tags or "現金流" in tags:
            return [
                "自由現金流可以幫助觀察公司在維持資本支出之後，還剩下多少現金能支應股利或其他資金用途。",
                "把現金股利發放總額與 FCF 一起看，比單看殖利率更能判斷股利政策是否有現金流支撐。",
                "若連續多年 FCF 高於現金股利支出，通常代表現階段股利政策較具穩定基礎。",
            ]
        if TopicTag.DEBT.value in tags or "負債" in tags:
            return [
                "負債比率可以協助觀察公司近幾期的槓桿水位是否突然墊高。",
                "把最新負債比率和前一季、去年同期一起比較，較能分辨是短期波動還是財務結構轉弱。",
                "現金及約當現金若明顯高於現金股利總額，通常代表眼前的股利支付緩衝仍在。",
            ]
        if "除息" in tags or "填息" in tags:
            return [
                "除權息當天的填息率可以反映市場對股利題材的接受度。",
                "盤中最高填息率與收盤填息率可協助區分是短線衡動還是買盤延續。",
                "交易量與當日價格反應一起看，比單看漲跌更能描述市場態度。",
            ]
        return [
            "股利政策可反映公司對現金回饋的安排。",
            "現金殖利率可協助估算以目前股價換算的現金回饋水準。",
            "殖利率仍會隨股價變動，不是固定值。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.CASH_FLOW.value in tags or "現金流" in tags:
            return [
                "自由現金流雖能說明現金支應能力，仍不代表未來股利一定維持不變。",
                "若後續資本支出提升、現金流轉弱或監理政策改變，股利政策仍可能調整。",
                "現金股利發放總額為推估值，仍應以公司正式公告與實際發放安排為準。",
            ]
        if TopicTag.DEBT.value in tags or "負債" in tags:
            return [
                "負債比率短期變動不一定代表財務體質立即惡化，仍需拆解是應付帳款、借款或營運週轉造成。",
                "帳上現金高於股利總額，只能說目前支付緩衝存在，仍需搭配未來現金流與資本支出一起看。",
                "若後續公司調整股利政策、擴大投資或出現一次性資金需求，現有支應能力判斷也可能改變。",
            ]
        if "除息" in tags or "填息" in tags:
            return [
                "填息表現只反映當天市場行為，不代表後續走勢一定延續。",
                "若單日波動受到大盤或外部消息影響，可能放大或扭曲填息觀察。",
                "若缺少完整交易資料或公開報導，對市場反應的描述應保守解讀。",
            ]
        return [
            "股利最終內容仍需以董事會、股東會或公司正式公告為準。",
            "現金殖利率會隨股價變動，查詢當下與實際買入時點可能不同。",
            "若公司後續更新股利政策，現有換算結果需要重新檢查。",
        ]

    # ── private sub-summarizers ─────────────────────────────────────────

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

    def _summarize_ex_dividend_performance(self, label: str, report: GovernanceReport) -> str:
        cash_dividend = extract_number(report, r"現金股利約 (\d+(?:\.\d+)?) 元")
        ex_date = extract_text(report, r"除息交易日為 (\d{4}-\d{2}-\d{2})")
        intraday_fill_ratio = extract_number(report, r"盤中最高填息率約 (\d+(?:\.\d+)?)%")
        close_fill_ratio = extract_number(report, r"收盤填息率約 (\d+(?:\.\d+)?)%")
        reaction = extract_text(report, r"市場反應偏([中性正向強勁弱勢]+)")
        same_day_fill = extract_text(report, r"(當天(?:盤中)?(?:完成|未完成)填息)")
        if ex_date and intraday_fill_ratio and close_fill_ratio:
            fill_sentence = (
                f"除息日 {ex_date} 盤中最高填息率約 {intraday_fill_ratio}%，"
                f"收盤填息率約 {close_fill_ratio}%。"
            )
            if same_day_fill:
                fill_sentence += same_day_fill + "。"
            if reaction:
                fill_sentence += f"市場反應偏{reaction}。"
            return f"{label} 最新一次除息事件顯示，{fill_sentence}"
        if cash_dividend and ex_date:
            return (
                f"{label} 目前可確認最新現金股利約 {cash_dividend} 元，除息日為 {ex_date}，"
                "但仍缺少足夠價格證據來確認當天的填息表現。"
            )
        return f"資料不足，無法確認{label}除權息當天的填息表現與市場反應。"

    def _summarize_fcf_dividend_sustainability(self, label: str, report: GovernanceReport) -> str:
        annual_fcf = extract_year_metric_pairs(
            report,
            r"(\d{4}) 年營業活動淨現金流入約 [\d.]+ 億元，資本支出約 [\d.]+ 億元，推估自由現金流約 ([\d.]+) 億元",
        )
        annual_payout = extract_year_metric_pairs(
            report,
            r"(\d{4}) 年現金股利每股約 [\d.]+ 元。依參與分派總股數約 [\d.]+ 股估算，現金股利發放總額約 ([\d.]+) 億元",
        )
        sustainability = extract_text(
            report,
            r"(近三年自由現金流均高於現金股利支出，顯示目前股利政策具一定永續性|近三年大致能以自由現金流支應現金股利，整體永續性偏穩健|自由現金流對股利支應能力接近打平，後續仍需留意資本支出與獲利變化|自由現金流對現金股利支應能力偏弱，股利政策永續性需保守看待)",
        )
        if annual_fcf and annual_payout:
            fcf_segment = "、".join(f"{year} 年約 {value} 億元" for year, value in annual_fcf[:3])
            payout_segment = "、".join(f"{year} 年約 {value} 億元" for year, value in annual_payout[:3])
            conclusion = sustainability or "目前仍需持續觀察資本支出與獲利變化"
            return (
                f"{label} 近三個已揭露股利年度的自由現金流約為 {fcf_segment}；"
                f"現金股利發放總額約為 {payout_segment}。"
                f"就目前公開資料看，{conclusion}。"
            )
        if annual_fcf:
            fcf_segment = "、".join(f"{year} 年約 {value} 億元" for year, value in annual_fcf[:3])
            return (
                f"{label} 近三年自由現金流約為 {fcf_segment}，"
                "但現金股利發放總額資料仍不足以完整評估股利政策永續性。"
            )
        return f"資料不足，無法確認{label}過去三年的自由現金流、現金股利發放總額與股利政策永續性。"

    def _summarize_debt_dividend_safety(self, label: str, report: GovernanceReport) -> str:
        debt_ratio = extract_number(report, r"負債比率約 (\d+(?:\.\d+)?)%")
        previous_ratio = extract_number(report, r"前一季約 (\d+(?:\.\d+)?)%")
        year_ago_ratio = extract_number(report, r"去年同期約 (\d+(?:\.\d+)?)%")
        debt_status = extract_text(
            report,
            r"負債比率(未見突然升高，反而較前幾期回落|未見突然升高，大致持平|近期沒有明顯異常升高|有溫和升高|有明顯升高)",
        )
        cash_balance = extract_number(report, r"現金及約當現金約 (\d+(?:\.\d+)?) 億元")
        dividend_total = extract_number(report, r"現金股利發放總額約 (\d+(?:\.\d+)?) 億元")
        coverage_ratio = extract_number(report, r"約可覆蓋 (\d+(?:\.\d+)?) 倍")
        payout_view = extract_text(
            report,
            r"(若只看帳上現金，現金部位看起來足以支應現金股利|若只看帳上現金，現金部位大致可支應現金股利，但仍要留意後續營運與資本支出|若只看帳上現金，現金部位勉強可支應現金股利，但緩衝不算特別厚|若只看帳上現金，支應現金股利的緩衝偏薄)",
        )
        if debt_ratio and debt_status and cash_balance and dividend_total and coverage_ratio:
            previous_segment = ""
            if previous_ratio and year_ago_ratio:
                previous_segment = f"；前一季約 {previous_ratio}%，去年同期約 {year_ago_ratio}%"
            payout_segment = (
                f"；最新現金及約當現金約 {cash_balance} 億元，"
                f"對最近一次現金股利總額約 {dividend_total} 億元，約可覆蓋 {coverage_ratio} 倍"
            )
            payout_tail = f"，{payout_view}" if payout_view else ""
            return (
                f"{label} 最新負債比率約 {debt_ratio}%{previous_segment}，{debt_status}"
                f"{payout_segment}{payout_tail}。"
            )
        return f"資料不足，無法確認{label}負債比率是否突然升高，以及現金部位是否足以支應股利。"
