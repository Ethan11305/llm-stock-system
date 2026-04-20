from llm_stock_system.core.enums import TopicTag
from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import extract_number, extract_text


class TechnicalViewStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if TopicTag.MARGIN_FLOW.value in tags or "籌碼" in tags or "季線" in tags:
            return self._summarize_season_line_margin(label, report)

        return self._summarize_technical_indicators(label, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.MARGIN_FLOW.value in tags or "籌碼" in tags or "季線" in tags:
            return [
                "季線是中短期趨勢的常見觀察線，能快速建立目前價格強弱的背景。",
                "融資餘額與融資使用率能協助觀察槓桿資金是否集中，對短線波動通常比較敏感。",
                "把季線位置與融資餘額一起看，比單看價格或單看籌碼更容易掌握風險。",
            ]
        return [
            "RSI 與 KD 能幫助觀察短期買盤熱度與強弱變化。",
            "MACD 適合用來看價格動能是否持續轉強或轉弱，布林通道則可以觀察股價相對區間位置。",
            "均線乖離可用來檢查短線漲勢是否過熱或過冷，搭配多個指標會比單看 RSI 或 KD 更完整。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        tags = set(query.topic_tags)
        if TopicTag.MARGIN_FLOW.value in tags or "籌碼" in tags or "季線" in tags:
            return [
                "跌破季線不一定代表中期趨勢確認轉空，仍需觀察後續幾個交易日是否繼續失守。",
                "融資餘額偏高不代表必然會出現修正，但對波動放大的風險確實較高。",
                "若公開資料缺少主流來源直接評論融資熱度，對「市場看法」的描述仍應以籌碼推估為主。",
            ]
        return [
            "超買不代表股價會立即回檔，仍可能因強勢趨勢持續上行。",
            "技術指標是根據歷史價格推算，遇到突發消息時可能很快失真。",
            "若 MACD 與 RSI、KD 訊號不一致，表示短線動能與價格位階可能正在拉鋸，不宜單看單一指標下結論。",
        ]

    # ── private sub-summarizers ─────────────────────────────────────────

    def _summarize_technical_indicators(self, label: str, report: GovernanceReport) -> str:
        rsi = extract_number(report, r"RSI14 約 (\d+(?:\.\d+)?)")
        k_value = extract_number(report, r"K 值約 (\d+(?:\.\d+)?)")
        d_value = extract_number(report, r"D 值約 (\d+(?:\.\d+)?)")
        latest_close = extract_number(report, r"最新收盤價約 (\d+(?:\.\d+)?) 元")
        macd_line = extract_number(report, r"MACD 線約 (-?\d+(?:\.\d+)?)")
        signal_line = extract_number(report, r"Signal 線約 (-?\d+(?:\.\d+)?)")
        histogram = extract_number(report, r"Histogram 約 (-?\d+(?:\.\d+)?)")
        bollinger_position = extract_text(
            report,
            r"(接近布林上軌|接近布林下軌|位於布林中軌偏上|位於布林中軌偏下|位於中軌附近)",
        )
        macd_trend = extract_text(report, r"MACD 動能([偏多偏空轉強轉弱]+)")
        ma5_bias = extract_number(report, r"MA5 乖離率約 (-?\d+(?:\.\d+)?)%")
        ma20_bias = extract_number(report, r"MA20 乖離率約 (-?\d+(?:\.\d+)?)%")
        overbought = extract_text(report, r"(尚未進入超買區|疑似進入超買區|已進入超買區)")
        if rsi and k_value and d_value and overbought and macd_trend and bollinger_position and ma5_bias and ma20_bias:
            return (
                f"{label} 最新技術指標顯示，RSI14 約 {rsi}，"
                f"K 值約 {k_value}，D 值約 {d_value}，{overbought}；"
                f"MACD 動能{macd_trend}，股價{bollinger_position}，"
                f"MA5 乖離率約 {ma5_bias}%，MA20 乖離率約 {ma20_bias}%。"
            )
        if latest_close and macd_line and signal_line and histogram:
            return (
                f"{label} 目前最新收盤價約 {latest_close} 元，"
                f"MACD 線約 {macd_line}，Signal 線約 {signal_line}，Histogram 約 {histogram}，"
                "但仍需結合 RSI、KD 與布林通道承接足夠證據後才能完整判讀。"
            )
        if latest_close:
            return (
                f"{label} 目前已有最新價格資料，"
                "但不足以完整確認 RSI、KD、MACD、布林通道與均線乖離的綜合判讀。"
            )
        return f"資料不足，無法確認{label}目前的 RSI、KD、MACD、布林通道與均線乖離狀態。"

    def _summarize_season_line_margin(self, label: str, report: GovernanceReport) -> str:
        latest_close = extract_number(report, r"最新收盤價約 (\d+(?:\.\d+)?) 元")
        season_line = extract_number(report, r"季線\(MA60\)約 (\d+(?:\.\d+)?) 元")
        season_line_status = extract_text(
            report, r"(近期跌破季線|仍在季線下方|重新站回季線|尚未跌破季線)"
        )
        margin_balance = extract_number(report, r"最新融資餘額約 (\d+(?:\.\d+)?) 張")
        utilization_pct = extract_number(report, r"融資使用率約 (\d+(?:\.\d+)?)%")
        average_delta_pct = extract_number(report, r"相較近 20 日平均變動約 (-?\d+(?:\.\d+)?)%")
        margin_status = extract_text(report, r"籌碼面屬(偏高|中性偏高|中性偏低|中性)")
        if (
            latest_close and season_line and season_line_status
            and margin_balance and utilization_pct and margin_status
        ):
            delta_segment = f"，相較近 20 日平均變動約 {average_delta_pct}%" if average_delta_pct is not None else ""
            return (
                f"{label} 最新收盤價約 {latest_close} 元，季線約 {season_line} 元，"
                f"目前{season_line_status}；最新融資餘額約 {margin_balance} 張，"
                f"融資使用率約 {utilization_pct}%{delta_segment}，若以籌碼面推估屬{margin_status}。"
            )
        if latest_close and season_line and season_line_status:
            return (
                f"{label} 最新收盤價約 {latest_close} 元，季線約 {season_line} 元，"
                f"目前{season_line_status}，但融資餘額證據仍不足以完整評估市場看法。"
            )
        return f"資料不足，無法確認{label}股價是否跌破季線及融資餘額的市場觀感。"
