"""Technical view strategy.

Wave 3 sunset：季線 + 融資餘額 deep-dive（_summarize_season_line_margin）已下架。
現在只保留 RSI / KD / MACD / 布林 / 均線乖離這套最基本的技術指標摘要。
"""

from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import extract_number, extract_text


class TechnicalViewStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        label = query.company_name or query.ticker or "此標的"
        return self._summarize_technical_indicators(label, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        return [
            "RSI 與 KD 能幫助觀察短期買盤熱度與強弱變化。",
            "MACD 適合用來看價格動能是否持續轉強或轉弱，布林通道則可以觀察股價相對區間位置。",
            "均線乖離可用來檢查短線漲勢是否過熱或過冷，搭配多個指標會比單看 RSI 或 KD 更完整。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        return [
            "超買不代表股價會立即回檔，仍可能因強勢趨勢持續上行。",
            "技術指標是根據歷史價格推算，遇到突發消息時可能很快失真。",
            "若 MACD 與 RSI、KD 訊號不一致，表示短線動能與價格位階可能正在拉鋸，不宜單看單一指標下結論。",
        ]

    # ── private sub-summarizer ─────────────────────────────────────────

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
