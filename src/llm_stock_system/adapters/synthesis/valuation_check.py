from llm_stock_system.core.models import GovernanceReport, StructuredQuery
from llm_stock_system.core.fundamental_valuation import (
    build_fundamental_valuation_summary,
    is_fundamental_valuation_question,
)
from llm_stock_system.core.target_price import (
    build_forward_price_summary,
    is_forward_price_question,
)

from .helpers import build_summary_fallback, extract_number, extract_price_range, extract_text


class ValuationCheckStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if is_fundamental_valuation_question(query):
            return build_fundamental_valuation_summary(query, report)
        if is_forward_price_question(query):
            return build_forward_price_summary(query, report)
        if "股價區間" in tags:
            return self._summarize_price_range(label, query, report)
        if "VALUATION" in tags or "本益比" in tags:
            return self._summarize_pe_valuation(label, report)

        return build_summary_fallback(query, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if is_fundamental_valuation_question(query):
            return [
                "基本面可以幫助判斷獲利與營運動能是否具備延續性。",
                "本益比則可以拿來定位市場目前給這檔公司的估值水位。",
                "若要做進場判斷，最好是把獲利趨勢和估值位置一起看。",
            ]
        if "股價區間" in tags:
            return [
                "區間高低點可用來觀察短線波動與市場交易熱度。",
                "若價格靠近區間上緣或下緣，需搭配後續消息確認是否延續。",
                "單看區間無法判斷基本面是否同步改善或轉弱。",
            ]
        return [
            "本益比能幫助估計市場目前願意用多高的評價倍數反映公司獲利能力。",
            "把目前本益比放回近 13 個月歷史區間，比單看單一倍數更容易判斷估值是否偏高或偏低。",
            "對長期投資來說，本益比只是進場價格的一個觀察面，仍需搭配成長性與現金流一起看。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        tags = set(query.topic_tags)
        if is_fundamental_valuation_question(query):
            return [
                "估值高低會隨股價與獲利預期變化，今天的本益比位置不代表之後不會再調整。",
                "基本面如果只有單一季或單月證據，不一定能代表中長期獲利趨勢。",
                "單看本益比或單看營收都可能失真，仍要搭配現金流、資本支出和產業景氣一起解讀。",
            ]
        if is_forward_price_question(query):
            return [
                "漲跌預測本身具有高度不確定性，不能視為確定方向。",
                "短期價格可能受到外部消息、資金輪動與整體市場風險影響。",
                "若近期有公告或財報更新，市場反應可能放大波動。",
            ]
        if "股價區間" in tags:
            return [
                "價格區間只反映歷史交易結果，不能直接推論未來方向。",
                "若期間內剛好出現重大事件，區間可能失真放大短期波動。",
                "未結合基本面與公告時，容易忽略趨勢反轉風險。",
            ]
        return [
            "本益比會隨股價與獲利預期變動，今天的估值位置不代表之後不會再調整。",
            "歷史偏高不一定代表馬上貴到不能買，仍需搭配未來 EPS 成長與產業景氣判斷。",
            "若只用本益比決定是否進場，可能忽略現金流、股利與資本支出等更適合長投的變數。",
        ]

    # ── private sub-summarizers ─────────────────────────────────────────

    def _summarize_price_range(
        self, label: str, query: StructuredQuery, report: GovernanceReport
    ) -> str:
        high_price, low_price = extract_price_range(report)
        if high_price and low_price:
            return (
                f"{label} 近 {query.time_range_days} 天資料顯示，"
                f"最高價為 {high_price} 元，最低價為 {low_price} 元。"
            )
        return f"{label} 目前已有股價資料，但不足以穩定整理出完整區間。"

    def _summarize_pe_valuation(self, label: str, report: GovernanceReport) -> str:
        current_pe = extract_number(report, r"本益比約 (\d+(?:\.\d+)?) 倍")
        low_pe = extract_number(report, r"本益比區間約 (\d+(?:\.\d+)?) 至")
        high_pe = extract_number(report, r"至 (\d+(?:\.\d+)?) 倍")
        percentile = extract_number(report, r"歷史分位 (\d+(?:\.\d+)?)%")
        valuation_zone = extract_text(report, r"屬(歷史偏高區|歷史偏低區|歷史中段區)")
        entry_view = extract_text(report, r"對長期投資來說，(.+?)。")
        if current_pe and low_pe and high_pe and valuation_zone:
            tail = f"；若以近 13 個月歷史區間衡量，{entry_view}" if entry_view else ""
            return (
                f"{label} 目前本益比約 {current_pe} 倍；"
                f"若看近 13 個月區間約 {low_pe} 至 {high_pe} 倍，"
                f"目前屬{valuation_zone}{tail}。"
            )
        if current_pe and percentile:
            return (
                f"{label} 目前本益比約 {current_pe} 倍，"
                f"約落在近 13 個月歷史分位 {percentile}% 左右，"
                "但仍需搭配獲利成長與產業狀況一起判斷是否偏貴。"
            )
        return f"資料不足，無法確認{label}目前本益比在歷史區間所處的位置與是否偏貴。"
