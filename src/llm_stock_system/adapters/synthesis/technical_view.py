"""Technical view strategy.

Wave 4 重構：移除 10-regex 全命中才輸出的 legacy summarizer。
新實作以 evidence_contains 動態偵測可用的技術指標類型，
描述哪些指標有資料、哪些訊號已出現，數字細節由 AugmentedContext.structured_block 提供。
這讓 rule-based 路徑在只有部分指標資料時也能輸出有意義的摘要。
"""

from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import evidence_contains, extract_text


class TechnicalViewStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        label = query.company_name or query.ticker or "此標的"
        return self._summarize_available_signals(label, report)

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

    def _summarize_available_signals(self, label: str, report: GovernanceReport) -> str:
        """依 evidence_contains 動態偵測哪些指標有資料，組合摘要描述。

        不要求所有 10 個 regex 全命中；只要有部分指標即可輸出有意義的定性描述。
        精確數值留給 AugmentedContext.structured_block 在 GenerationLayer 使用。
        """
        has_rsi = evidence_contains(report, ("RSI14", "RSI"))
        has_kd = evidence_contains(report, ("K 值", "D 值", "KD"))
        has_macd = evidence_contains(report, ("MACD",))
        has_bollinger = evidence_contains(report, ("布林", "Bollinger", "布林上軌", "布林下軌"))
        has_ma_bias = evidence_contains(report, ("MA5 乖離率", "MA20 乖離率", "乖離率"))
        has_overbought = evidence_contains(report, ("超買", "超賣"))

        macd_trend = extract_text(report, r"MACD 動能([偏多偏空轉強轉弱]+)")
        bollinger_pos = extract_text(
            report,
            r"(接近布林上軌|接近布林下軌|位於布林中軌偏上|位於布林中軌偏下|位於中軌附近)",
        )

        available_indicators = [
            name for name, present in [
                ("RSI", has_rsi), ("KD", has_kd), ("MACD", has_macd),
                ("布林通道", has_bollinger), ("均線乖離", has_ma_bias),
            ] if present
        ]

        if not available_indicators:
            return f"資料不足，無法確認{label}目前的技術指標狀態。"

        indicators_str = "、".join(available_indicators)
        signal_parts: list[str] = []

        if macd_trend:
            signal_parts.append(f"MACD 動能{macd_trend}")
        if bollinger_pos:
            signal_parts.append(f"股價{bollinger_pos}")
        if has_overbought:
            overbought_signal = extract_text(report, r"(尚未進入超買區|疑似進入超買區|已進入超買區)")
            if overbought_signal:
                signal_parts.append(overbought_signal)

        base = f"{label} 目前已有 {indicators_str} 等技術指標資料可供整理"
        if signal_parts:
            return base + "；" + "，".join(signal_parts) + "；詳細數值請參照下方資料欄位。"
        return base + "，詳細數值請參照下方資料欄位。"
