"""Forecast helper module for price_outlook queries.

Centralises scenario estimation, ForecastBlock construction, and
guardrail logic for forward-looking answers.

Modes:
- scenario_estimate: analyst target prices or multiple numeric anchors found.
- historical_proxy: only recent price volatility available as a proxy.
- unsupported: no usable forward or price evidence at all.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from llm_stock_system.core.enums import ForecastDirection, ForecastMode
from llm_stock_system.core.models import (
    ForecastBlock,
    ForecastWindow,
    GovernanceReport,
    ScenarioRange,
    StructuredQuery,
)
from llm_stock_system.core.target_price import (
    extract_target_price_values,
    has_target_price_context,
)


# ------------------------------------------------------------------ #
# Direction inference                                                   #
# ------------------------------------------------------------------ #

_BULLISH_TOKENS = ("看多", "看漲", "偏多", "上漲", "站上", "突破", "多方", "利多", "上修")
_BEARISH_TOKENS = ("看空", "看跌", "偏空", "下跌", "跌破", "失守", "空方", "利空", "下修")


def _infer_direction(query: StructuredQuery, governance_report: GovernanceReport) -> ForecastDirection:
    """Infer directional bias from evidence text.

    Counts bullish vs. bearish tokens; if neither dominates, returns
    RANGE_BOUND when evidence exists or UNDETERMINED otherwise.
    """
    combined = " ".join(f"{e.title} {e.excerpt}" for e in governance_report.evidence).lower()

    bull_count = sum(1 for token in _BULLISH_TOKENS if token in combined)
    bear_count = sum(1 for token in _BEARISH_TOKENS if token in combined)

    if bull_count > bear_count and bull_count >= 2:
        return ForecastDirection.BULLISH_BIAS
    if bear_count > bull_count and bear_count >= 2:
        return ForecastDirection.BEARISH_BIAS
    if governance_report.evidence:
        return ForecastDirection.RANGE_BOUND
    return ForecastDirection.UNDETERMINED


# ------------------------------------------------------------------ #
# Price range extraction from evidence                                  #
# ------------------------------------------------------------------ #

# Source names that contain genuine stock price data
_PRICE_SOURCE_FRAGMENTS = (
    "taiwanstockprice",
    "taiwan_stock_price",
    "price",
    "twse",
)

# Context keywords: a number is considered a stock price only if the
# surrounding text mentions one of these price-specific terms.
_PRICE_CONTEXT_KEYWORDS = (
    "收盤價", "收盤", "最高價", "最低價", "開盤價", "開盤",
    "股價", "成交價", "高點", "低點", "高低", "波動",
    "支撐", "壓力", "關卡", "前高", "前低",
)

# Contextual price pattern: (price-context keyword) ... (number 元)
_CONTEXTUAL_PRICE_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(kw) for kw in _PRICE_CONTEXT_KEYWORDS) + r")"
    r"[^0-9]{0,20}"
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:元|塊)?",
)

# Also match "(number) 元" when immediately preceded by price context
_PRICE_AFTER_NUMBER_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:元|塊)"
    r"[^0-9]{0,10}"
    r"(?:" + "|".join(re.escape(kw) for kw in (
        "收盤", "最高", "最低", "開盤", "支撐", "壓力", "關卡",
        "高點", "低點", "波動",
    )) + r")",
)


def _is_price_source(source_name: str) -> bool:
    """Check if the source is a recognised stock price data source."""
    lowered = source_name.lower().replace(" ", "")
    return any(fragment in lowered for fragment in _PRICE_SOURCE_FRAGMENTS)


def _extract_recent_price_range(governance_report: GovernanceReport) -> tuple[float | None, float | None]:
    """Extract lowest/highest price mentions from evidence as proxy range.

    Only considers:
    1. Evidence from recognised price data sources (e.g., TaiwanStockPrice).
    2. Numbers in a price-specific context (收盤價, 最高價, 支撐, 壓力, etc.).

    Excludes EPS, revenue, target prices, and other non-price numbers.
    """
    values: list[float] = []

    for e in governance_report.evidence:
        haystack = f"{e.title} {e.excerpt}"

        # Strategy 1: trust numbers with explicit price units from price data sources
        if _is_price_source(e.source_name):
            for match in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(?:元|塊)", haystack):
                _try_append_price(values, match.group(1))
            continue

        # Strategy 2: only accept numbers next to price-context keywords
        for pattern in (_CONTEXTUAL_PRICE_PATTERN, _PRICE_AFTER_NUMBER_PATTERN):
            for match in pattern.finditer(haystack):
                _try_append_price(values, match.group(1))

    if len(values) >= 2:
        low, high = min(values), max(values)
        # Sanity: reject absurd spreads (> 5x between low and high)
        if high <= low * 5:
            return low, high
    return None, None


def _try_append_price(values: list[float], raw: str) -> None:
    """Parse a raw price string and append to values if it looks like a stock price."""
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return
    # Stock prices are typically 1–99999; reject outliers and tiny values
    # that are more likely EPS or percentages.
    if 10 < value < 100000:
        values.append(value)


# ------------------------------------------------------------------ #
# ForecastBlock builder                                                 #
# ------------------------------------------------------------------ #

def build_forecast_block(
    query: StructuredQuery,
    governance_report: GovernanceReport,
    reference_date: date | None = None,
) -> ForecastBlock:
    """Build a ForecastBlock for a price_outlook query.

    Decision tree:
    1. If analyst target prices or multiple numeric anchors → scenario_estimate.
    2. If recent price data can serve as proxy range → historical_proxy.
    3. Otherwise → unsupported.
    """
    ref = reference_date or date.today()
    horizon_days = query.forecast_horizon_days or 7
    horizon_label = query.forecast_horizon_label or "未來一週"
    window = ForecastWindow(
        label=horizon_label,
        start_date=ref.isoformat(),
        end_date=(ref + timedelta(days=horizon_days)).isoformat(),
    )

    target_prices = extract_target_price_values(governance_report)
    has_analyst_context = has_target_price_context(governance_report)

    # --- Mode 1: scenario_estimate ---
    if target_prices and len(target_prices) >= 2:
        return ForecastBlock(
            mode=ForecastMode.SCENARIO_ESTIMATE,
            forecast_window=window,
            direction=_infer_direction(query, governance_report),
            scenario_range=ScenarioRange(
                low=target_prices[0],
                high=target_prices[-1],
                basis_type="analyst_target",
            ),
            forecast_basis=[
                f"分析師目標價區間 {target_prices[0]}–{target_prices[-1]} 元",
                "此為情境推估，不是確定預測",
            ],
        )

    if target_prices and len(target_prices) == 1 and has_analyst_context:
        return ForecastBlock(
            mode=ForecastMode.SCENARIO_ESTIMATE,
            forecast_window=window,
            direction=_infer_direction(query, governance_report),
            scenario_range=None,  # only one anchor; not enough for a range
            forecast_basis=[
                f"單一分析師目標價約 {target_prices[0]} 元",
                "來源有限，僅供參考",
                "此為情境推估，不是確定預測",
            ],
        )

    if has_analyst_context and not target_prices:
        # Direction evidence only, no numeric range
        return ForecastBlock(
            mode=ForecastMode.SCENARIO_ESTIMATE,
            forecast_window=window,
            direction=_infer_direction(query, governance_report),
            scenario_range=None,
            forecast_basis=[
                "分析師/法人有方向性觀點但未提供明確目標價",
                "此為情境推估，不是確定預測",
            ],
        )

    # --- Mode 2: historical_proxy ---
    low, high = _extract_recent_price_range(governance_report)
    if low is not None and high is not None and low < high:
        return ForecastBlock(
            mode=ForecastMode.HISTORICAL_PROXY,
            forecast_window=window,
            direction=_infer_direction(query, governance_report),
            scenario_range=ScenarioRange(
                low=low,
                high=high,
                basis_type="historical_proxy",
            ),
            forecast_basis=[
                f"近期價格波動區間約 {low}–{high} 元（歷史波動代理，不是明確目標價）",
                "僅以歷史波動代理，不代表未來實際走勢",
            ],
        )

    if governance_report.evidence:
        return ForecastBlock(
            mode=ForecastMode.HISTORICAL_PROXY,
            forecast_window=window,
            direction=_infer_direction(query, governance_report),
            scenario_range=None,
            forecast_basis=[
                "有近期資料但無法組成有效價格區間",
                "僅以歷史波動代理，不代表未來實際走勢",
            ],
        )

    # --- Mode 3: unsupported ---
    return ForecastBlock(
        mode=ForecastMode.UNSUPPORTED,
        forecast_window=window,
        direction=ForecastDirection.UNDETERMINED,
        scenario_range=None,
        forecast_basis=["無可用的前瞻或價格依據"],
    )


# ------------------------------------------------------------------ #
# Guardrail: rewrite overly confident forecast language                 #
# ------------------------------------------------------------------ #

_OVERCONFIDENT_PATTERNS = [
    (re.compile(r"(一定|肯定|必定|絕對)(會|將)(漲|跌|上漲|下跌)"), r"情境推估\3的可能性較高"),
    (re.compile(r"(預計|預期)(將)(漲到|跌到|上漲至|下跌至)(\d[\d,]*(?:\.\d+)?)"), r"情境推估可能\3\4"),
    (re.compile(r"(會漲到|會跌到|會上漲至|會下跌至)(\d[\d,]*(?:\.\d+)?)"), r"情境推估可能\1\2"),
]

_SCENARIO_DISCLAIMER_ZH = "（此回答為情境推估，非事實預測）"
_HISTORICAL_DISCLAIMER_ZH = "（僅以歷史波動代理，非明確目標價）"

# Pattern to find price-like numbers in LLM output (e.g. "950 元", "1,050元")
_LLM_PRICE_NUMBER_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:元|塊)"
)

# Price-prediction phrases that wrap a number (e.g. "預估區間 900~950 元")
_LLM_RANGE_PHRASE_PATTERN = re.compile(
    r"(?:預估區間|預期區間|推估區間|可能區間|波動區間|區間約?|區間(?:大約|落在)?)"
    r"\s*(?:(?:為|在|約|落在|大約)\s*)?"
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:元|塊)?\s*(?:~|～|–|-|至|到)\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:元|塊)?",
)


def _collect_allowed_numbers(forecast: ForecastBlock) -> set[float]:
    """Build a whitelist of numbers that appear in the ForecastBlock."""
    allowed: set[float] = set()
    if forecast.scenario_range is not None:
        allowed.add(forecast.scenario_range.low)
        allowed.add(forecast.scenario_range.high)
    # Also allow numbers explicitly mentioned in forecast_basis strings
    for basis in forecast.forecast_basis:
        for match in re.finditer(r"(\d[\d,]*(?:\.\d+)?)", basis):
            try:
                allowed.add(float(match.group(1).replace(",", "")))
            except ValueError:
                pass
    return allowed


def _is_number_allowed(value: float, allowed: set[float], tolerance: float = 0.05) -> bool:
    """Check if *value* is within *tolerance* (fraction) of any allowed number."""
    if not allowed:
        return False
    for ref in allowed:
        if ref == 0:
            continue
        if abs(value - ref) / ref <= tolerance:
            return True
    return False


def _sanitize_fabricated_numbers(text: str, forecast: ForecastBlock) -> str:
    """Replace price numbers in *text* that don't match the ForecastBlock.

    When the LLM fabricates a concrete price number that has no basis in
    the structured forecast data, we replace it with a safe phrase so the
    user never sees an ungrounded number presented as a prediction.
    """
    allowed = _collect_allowed_numbers(forecast)

    # If no allowed numbers at all (unsupported / no range), strip ALL
    # price-prediction range phrases and isolated price numbers that look
    # like predictions.
    has_range = forecast.scenario_range is not None

    # 1. Replace entire fabricated range phrases first
    def _replace_range_phrase(match: re.Match) -> str:
        low_raw = match.group(1).replace(",", "")
        high_raw = match.group(2).replace(",", "")
        try:
            low_val, high_val = float(low_raw), float(high_raw)
        except ValueError:
            return match.group(0)

        if has_range and _is_number_allowed(low_val, allowed) and _is_number_allowed(high_val, allowed):
            return match.group(0)  # legitimate, keep it
        return "（系統無法提供具體預估區間數值）"

    text = _LLM_RANGE_PHRASE_PATTERN.sub(_replace_range_phrase, text)

    # 2. Replace isolated fabricated "X 元" numbers that look like
    #    price predictions (preceded by forecast-ish verbs).
    _PREDICTION_PREFIX = re.compile(
        r"(預估|預期|推估|可能(?:到達|觸及|挑戰|跌至|漲至)?|目標|上看|下看|看到)"
        r"\s*(\d[\d,]*(?:\.\d+)?)\s*(?:元|塊)"
    )

    def _replace_prediction_number(match: re.Match) -> str:
        raw = match.group(2).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            return match.group(0)
        if _is_number_allowed(val, allowed):
            return match.group(0)
        return f"{match.group(1)}（無可靠依據的具體數值）"

    text = _PREDICTION_PREFIX.sub(_replace_prediction_number, text)

    return text


def apply_forecast_guardrail(summary: str, forecast: ForecastBlock) -> str:
    """Rewrite overly assertive forecast language into scenario tone.

    Three-pass guardrail:
    1. Rewrite overconfident verb phrases (一定會漲 → 情境推估).
    2. Cross-check numbers against ForecastBlock; replace fabricated ones.
    3. Append mode-specific disclaimer.
    """
    text = summary

    # Pass 1: overconfident language
    for pattern, replacement in _OVERCONFIDENT_PATTERNS:
        text = pattern.sub(replacement, text)

    # Pass 2: fabricated number cross-check
    text = _sanitize_fabricated_numbers(text, forecast)

    # Pass 3: disclaimer
    if forecast.mode == ForecastMode.SCENARIO_ESTIMATE:
        if _SCENARIO_DISCLAIMER_ZH not in text:
            text = text.rstrip("。") + _SCENARIO_DISCLAIMER_ZH
    elif forecast.mode == ForecastMode.HISTORICAL_PROXY:
        if _HISTORICAL_DISCLAIMER_ZH not in text:
            text = text.rstrip("。") + _HISTORICAL_DISCLAIMER_ZH

    return text


def apply_forecast_guardrail_to_list(items: list[str], forecast: ForecastBlock) -> list[str]:
    """Apply number cross-check guardrail to a list of highlight/fact strings."""
    allowed = _collect_allowed_numbers(forecast)
    if not allowed and forecast.scenario_range is None:
        # No reference numbers: strip all prediction-style price numbers
        return [_sanitize_fabricated_numbers(item, forecast) for item in items]
    return [_sanitize_fabricated_numbers(item, forecast) for item in items]
