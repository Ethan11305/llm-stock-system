from __future__ import annotations

import re

from llm_stock_system.core.enums import StanceBias, Topic, TopicTag
from llm_stock_system.core.interfaces import QueryClassifier, StockResolver
from llm_stock_system.core.models import (
    QUESTION_TYPE_TO_INTENT,
    QueryRequest,
    StructuredQuery,
    infer_intent_from_question_type,
)

_VALID_QUESTION_TYPES: frozenset[str] = frozenset(QUESTION_TYPE_TO_INTENT.keys())
_VALID_TIME_RANGE_LABELS: frozenset[str] = frozenset(
    {"1d", "7d", "30d", "latest_quarter", "1y", "3y", "5y"}
)
_VALID_TOPIC_TAG_VALUES: frozenset[str] = frozenset(tag.value for tag in TopicTag)
_VALID_STANCE_VALUES: frozenset[str] = frozenset(bias.value for bias in StanceBias)
_TIME_RANGE_DAYS: dict[str, int] = {
    "1d": 1, "7d": 7, "30d": 30, "latest_quarter": 90,
    "1y": 365, "3y": 1095, "5y": 1825,
}


class InputLayer:
    COMPANY_ALIASES = {
        "\u53f0\u7a4d\u96fb": ("2330", "\u53f0\u7a4d\u96fb"),
        "\u53f0\u7a4d": ("2330", "\u53f0\u7a4d\u96fb"),
        "tsmc": ("2330", "\u53f0\u7a4d\u96fb"),
        "2330": ("2330", "\u53f0\u7a4d\u96fb"),
        "\u5927\u540c": ("2371", "\u5927\u540c"),
        "tatung": ("2371", "\u5927\u540c"),
        "2371": ("2371", "\u5927\u540c"),
        "\u9d3b\u6d77": ("2317", "\u9d3b\u6d77"),
        "hon hai": ("2317", "\u9d3b\u6d77"),
        "foxconn": ("2317", "\u9d3b\u6d77"),
        "2317": ("2317", "\u9d3b\u6d77"),
        "\u5bb6\u767b": ("3680", "\u5bb6\u767b"),
        "3680": ("3680", "\u5bb6\u767b"),
        "\u842c\u6f64": ("6187", "\u842c\u6f64"),
        "6187": ("6187", "\u842c\u6f64"),
        "\u83ef\u90a6\u96fb": ("2344", "\u83ef\u90a6\u96fb"),
        "\u83ef\u90a6": ("2344", "\u83ef\u90a6\u96fb"),
        "winbond": ("2344", "\u83ef\u90a6\u96fb"),
        "2344": ("2344", "\u83ef\u90a6\u96fb"),
        "\u6b23\u8208": ("3037", "\u6b23\u8208"),
        "unimicron": ("3037", "\u6b23\u8208"),
        "3037": ("3037", "\u6b23\u8208"),
        "\u9577\u69ae": ("2603", "\u9577\u69ae"),
        "evergreen": ("2603", "\u9577\u69ae"),
        "2603": ("2603", "\u9577\u69ae"),
        "\u967d\u660e": ("2609", "\u967d\u660e"),
        "\u967d\u660e\u6d77\u904b": ("2609", "\u967d\u660e"),
        "yang ming": ("2609", "\u967d\u660e"),
        "yangming": ("2609", "\u967d\u660e"),
        "2609": ("2609", "\u967d\u660e"),
        "\u9577\u69ae\u822a": ("2618", "\u9577\u69ae\u822a"),
        "\u9577\u69ae\u822a\u7a7a": ("2618", "\u9577\u69ae\u822a"),
        "eva air": ("2618", "\u9577\u69ae\u822a"),
        "2618": ("2618", "\u9577\u69ae\u822a"),
        "\u806f\u767c\u79d1": ("2454", "\u806f\u767c\u79d1"),
        "\u767c\u54e5": ("2454", "\u806f\u767c\u79d1"),
        "mediatek": ("2454", "\u806f\u767c\u79d1"),
        "2454": ("2454", "\u806f\u767c\u79d1"),
        "\u4e2d\u83ef\u96fb": ("2412", "\u4e2d\u83ef\u96fb\u4fe1"),
        "\u4e2d\u83ef\u96fb\u4fe1": ("2412", "\u4e2d\u83ef\u96fb\u4fe1"),
        "cht": ("2412", "\u4e2d\u83ef\u96fb\u4fe1"),
        "2412": ("2412", "\u4e2d\u83ef\u96fb\u4fe1"),
        "\u53cb\u9054": ("2409", "\u53cb\u9054"),
        "auo": ("2409", "\u53cb\u9054"),
        "2409": ("2409", "\u53cb\u9054"),
        "\u4e2d\u92fc": ("2002", "\u4e2d\u92fc"),
        "china steel": ("2002", "\u4e2d\u92fc"),
        "2002": ("2002", "\u4e2d\u92fc"),
        "\u53f0\u6ce5": ("1101", "\u53f0\u6ce5"),
        "taiwan cement": ("1101", "\u53f0\u6ce5"),
        "1101": ("1101", "\u53f0\u6ce5"),
        "\u5b8f\u7881": ("2353", "\u5b8f\u7881"),
        "acer": ("2353", "\u5b8f\u7881"),
        "2353": ("2353", "\u5b8f\u7881"),
        "\u7f8e\u742a\u746a": ("4721", "\u7f8e\u742a\u746a"),
        "mechema": ("4721", "\u7f8e\u742a\u746a"),
        "4721": ("4721", "\u7f8e\u742a\u746a"),
        "\u7def\u7a4e": ("6669", "\u7def\u7a4e"),
        "wiwynn": ("6669", "\u7def\u7a4e"),
        "6669": ("6669", "\u7def\u7a4e"),
        "\u661f\u5b87\u822a\u7a7a": ("2646", "\u661f\u5b87\u822a\u7a7a"),
        "\u661f\u5b87": ("2646", "\u661f\u5b87\u822a\u7a7a"),
        "starlux": ("2646", "\u661f\u5b87\u822a\u7a7a"),
        "2646": ("2646", "\u661f\u5b87\u822a\u7a7a"),
    }

    TOPIC_KEYWORDS = {
        Topic.EARNINGS: ("\u8ca1\u5831", "\u6cd5\u8aaa", "\u7372\u5229", "EPS", "\u6bcf\u80a1\u76c8\u9918", "\u6bdb\u5229\u7387", "\u71df\u6536"),
        Topic.ANNOUNCEMENT: (
            "\u516c\u544a",
            "\u516c\u4f48",
            "\u80a1\u5229",
            "\u914d\u606f",
            "\u73fe\u91d1\u80a1\u5229",
            "\u9664\u606f",
            "\u8463\u4e8b\u6703",
            "\u6cd5\u8aaa\u6703",
            "\u8655\u5206",
            "\u571f\u5730",
            "\u8cc7\u7522",
            "\u696d\u5916",
            "\u5165\u5e33",
            "\u8a8d\u5217",
        ),
        Topic.NEWS: ("\u65b0\u805e", "\u6d88\u606f", "\u5e02\u5834", "\u5a92\u9ad4", "\u9810\u671f", "\u770b\u6cd5", "\u50b3\u805e"),
    }

    PRICE_RANGE_KEYWORDS = ("\u6700\u9ad8\u9ede", "\u6700\u4f4e\u9ede", "\u6700\u9ad8\u50f9", "\u6700\u4f4e\u50f9", "\u5340\u9593", "range")
    PRICE_OUTLOOK_KEYWORDS = (
        "漲跌預測",
        "會漲嗎",
        "會跌嗎",
        "走勢",
        "預測",
        "目標價",
        # directional continuation patterns
        "繼續漲",
        "繼續跌",
        "續漲",
        "續跌",
        "會不會漲",
        "會不會跌",
        "還會漲",
        "還會跌",
        "還能漲",
        "還能跌",
        "上漲空間",
        "下跌空間",
    )
    # Forward-looking time window keywords: when combined with direction/range
    # demand, the query is a forecast (price_outlook), not a historical price_range.
    FORECAST_TIME_WINDOW_KEYWORDS = (
        "未來",
        "下週",
        "下一週",
        "這週",
        "這一週",
        "這個星期",
        "這一個星期",
        "下個星期",
        "接下來",
        "明天",
        "明日",
        "後天",
        "下個月",
        "下半年",
        "上半年",
    )
    # Direction/range demand keywords that, when combined with a future time
    # window, route the query to forecast instead of historical price_range.
    FORECAST_DEMAND_KEYWORDS = (
        "預期",
        "預估",
        "預測",
        "可能",
        "會上漲",
        "會下跌",
        "會漲",
        "會跌",
        "波動如何",
        "波動",
        "漲還是跌",
        "偏多",
        "偏空",
        "看漲",
        "看跌",
        # 「區間」單獨出現不代表預測需求（e.g. 「這週股價區間？」是查歷史高低點）；
        # 需配合「預估」「預測」等明確前瞻語才應觸發 forecast。
        # 「預估區間」由上方 "預估" 項目涵蓋，故此處移除裸字 "區間"。
        "估計",
        "推估",
    )
    FUNDAMENTAL_OVERVIEW_KEYWORDS = (
        "\u57fa\u672c\u9762",
        "\u9ad4\u8cea",
        "\u71df\u904b",
        "\u8ca1\u5831",
        "\u7372\u5229",
        "\u71df\u6536",
        "eps",
        "\u6210\u9577",
        "\u6bdb\u5229",
        "\u73fe\u91d1\u6d41",
    )
    PRICE_LEVEL_ACTION_KEYWORDS = ("\u7a81\u7834", "\u7ad9\u4e0a", "\u8dcc\u7834", "\u5931\u5b88", "\u5b88\u4f4f", "\u7ad9\u7a69", "\u4e0a\u770b", "\u6311\u6230")
    PRICE_LEVEL_FUTURE_HINTS = (
        "\u672a\u4f86",
        "\u4e0b\u534a\u5e74",
        "\u4e0a\u534a\u5e74",
        "\u534a\u5e74",
        "\u6709\u6a5f\u6703",
        "\u53ef\u80fd",
        "\u80fd\u5426",
        "\u6703\u4e0d\u6703",
        "\u662f\u5426",
        "\u53ef\u4e0d\u53ef\u80fd",
    )
    EPS_KEYWORDS = ("eps", "\u6bcf\u80a1\u76c8\u9918", "\u8ca1\u5831", "\u7372\u5229", "\u6cd5\u8aaa")
    MONTHLY_REVENUE_KEYWORDS = (
        "\u6708\u71df\u6536",
        "\u7d2f\u8a08\u71df\u6536",
        "\u71df\u6536",
        "\u53bb\u5e74\u540c\u671f",
        "\u5e74\u589e",
        "\u6210\u9577\u4e86\u767e\u5206\u4e4b\u5e7e",
        "\u6708\u589e",
        "\u6708\u589e\u7387",
        "mom",
        "\u5275\u65b0\u9ad8",
        "\u65b0\u9ad8",
    )
    PE_VALUATION_KEYWORDS = ("\u672c\u76ca\u6bd4", "p/e", "pe ratio", "\u4f30\u503c", "\u8cb7\u8cb4", "\u4fbf\u5b9c", "\u9ad8\u4f4d", "\u4f4e\u4f4d", "\u9032\u5834")
    COMPARISON_KEYWORDS = ("\u6bd4\u8f03", "\u5c0d\u6bd4", "\u8ddf", "\u8207", "\u548c", "\u8ab0", "\u54ea\u4e00\u5bb6")
    GROSS_MARGIN_KEYWORDS = ("\u6bdb\u5229\u7387", "\u6bdb\u5229", "\u71df\u696d\u6bdb\u5229", "\u7d93\u71df\u6548\u7387", "\u6548\u7387")
    TURNAROUND_KEYWORDS = (
        "\u7531\u8ca0\u8f49\u6b63",
        "\u7531\u8ca0\u8f49\u6b63",
        "\u8f49\u6b63",
        "\u8ca0\u8f49\u6b63",
        "\u71df\u696d\u5229\u76ca",
        "\u71df\u696d\u640d\u76ca",
        "\u5be6\u8cea\u7372\u5229\u6539\u5584",
        "\u7372\u5229\u6539\u5584",
    )
    PROFITABILITY_STABILITY_KEYWORDS = (
        "\u9000\u4f11\u5b58\u80a1",
        "\u5b58\u80a1",
        "\u7a69\u5b9a\u7372\u5229",
        "\u6bcf\u5e74\u90fd\u6709\u7a69\u5b9a\u7372\u5229",
        "\u6bcf\u5e74\u90fd\u8cfa\u9322",
        "\u5927\u8667\u640d",
        "\u7a81\u7136\u5927\u8667",
        "\u8f49\u8667",
        "\u8667\u640d",
        "\u539f\u56e0\u662f\u4ec0\u9ebc",
    )
    FCF_KEYWORDS = ("\u81ea\u7531\u73fe\u91d1\u6d41", "fcf", "\u71df\u696d\u73fe\u91d1\u6d41", "\u73fe\u91d1\u6d41")
    DEBT_RATIO_KEYWORDS = (
        "\u8ca0\u50b5\u6bd4\u7387",
        "\u8ca0\u50b5\u6bd4",
        "\u8ca0\u50b5\u7e3d\u984d",
        "\u8cc7\u7522\u8ca0\u50b5\u8868",
        "\u69d3\u687f",
    )
    CASH_BALANCE_KEYWORDS = (
        "\u73fe\u91d1\u53ca\u7d04\u7576\u73fe\u91d1",
        "\u624b\u4e0a\u7684\u73fe\u91d1",
        "\u73fe\u91d1\u9084\u5920\u4e0d\u5920\u767c\u80a1\u5229",
        "\u5920\u4e0d\u5920\u767c\u80a1\u5229",
        "\u767c\u80a1\u5229",
    )
    DIVIDEND_TOTAL_KEYWORDS = (
        "\u73fe\u91d1\u80a1\u5229\u767c\u653e\u7e3d\u984d",
        "\u73fe\u91d1\u80a1\u5229\u7e3d\u984d",
        "\u80a1\u5229\u767c\u653e\u7e3d\u984d",
        "\u767c\u653e\u7e3d\u984d",
    )
    SUSTAINABILITY_KEYWORDS = ("\u6c38\u7e8c\u6027", "\u662f\u5426\u5177\u6709\u6c38\u7e8c\u6027", "\u53ef\u6301\u7e8c", "\u6301\u7e8c\u6027", "\u80fd\u5426\u6c38\u7e8c")
    REVENUE_GROWTH_KEYWORDS = ("\u71df\u6536\u5360\u6bd4", "\u71df\u6536\u6bd4\u91cd", "\u5360\u6bd4", "\u6bd4\u91cd", "\u6210\u9577\u9810\u6e2c", "\u6210\u9577\u5c55\u671b", "\u6210\u9577\u52d5\u80fd")
    DIVIDEND_KEYWORDS = ("\u80a1\u5229", "\u914d\u606f", "\u73fe\u91d1\u80a1\u5229", "\u9664\u606f", "\u6b96\u5229\u7387")
    DIVIDEND_YIELD_KEYWORDS = ("\u6b96\u5229\u7387", "\u73fe\u91d1\u6b96\u5229\u7387", "\u914d\u606f\u653f\u7b56", "\u80a1\u5229\u653f\u7b56")
    THEME_IMPACT_KEYWORDS = (
        "\u77ed\u7dda",
        "\u5f71\u97ff",
        "\u666e\u53ca\u7387",
        "\u9700\u6c42",
        "\u7522\u696d",
        "\u984c\u6750",
        "\u76f8\u95dc\u8cc7\u8a0a",
        "\u5229\u7a7a",
        "\u60c5\u7dd2",
        "\u5206\u6790",
        "\u5c55\u671b",
        "\u4e0d\u5982\u9810\u671f",
        "\u534a\u5c0e\u9ad4\u8a2d\u5099",
        "\u8a2d\u5099\u65cf\u7fa4",
        "\u8a2d\u5099\u80a1",
        "asml",
        "\u827e\u53f8\u6469\u723e",
    )
    ASSET_DISPOSAL_KEYWORDS = (
        "\u8655\u5206",
        "\u51fa\u552e",
        "\u571f\u5730",
        "\u8cc7\u7522",
        "\u696d\u5916",
        "\u5165\u5e33",
        "\u8a8d\u5217",
        "\u8ca2\u737b eps",
        "\u8ca2\u737beps",
    )
    SEASON_LINE_KEYWORDS = ("\u5b63\u7dda", "60\u65e5\u7dda", "60\u5747\u7dda", "60ma", "ma60")
    MARGIN_BALANCE_KEYWORDS = ("\u878d\u8cc7\u9918\u984d", "\u878d\u8cc7", "\u878d\u5238", "\u4fe1\u7528\u4ea4\u6613", "\u7c4c\u78bc")
    TECHNICAL_INDICATOR_KEYWORDS = (
        "\u6280\u8853\u6307\u6a19",
        "rsi",
        "kd",
        "macd",
        "k\u503c",
        "d\u503c",
        "\u5e03\u6797\u901a\u9053",
        "\u5747\u7dda",
        "\u4e56\u96e2",
        "\u8d85\u8cb7",
        "\u8d85\u8ce3",
    )
    EX_DIVIDEND_KEYWORDS = (
        "\u9664\u6b0a\u606f",
        "\u9664\u6b0a",
        "\u9664\u606f",
        "\u586b\u606f",
        "\u586b\u6b0a",
        "\u586b\u6b0a\u606f",
        "\u5e02\u5834\u53cd\u61c9",
    )
    LISTING_VOLATILITY_KEYWORDS = (
        "\u8f49\u4e0a\u5e02",
        "\u4e0a\u5e02",
        "\u639b\u724c",
        "\u4e0a\u5e02\u639b\u724c",
        "ipo",
        "IPO",
        "initial public offering",
        "\u65b0\u6383\u724c",
        "\u65b0\u4e0a\u5e02",
        "\u5bc6\u6708\u884c\u60c5",
        "\u627f\u92b7",
        "\u80a1\u50f9\u6ce2\u52d5",
        "\u6ce2\u52d5",
        "\u6ce2\u52d5\u539f\u56e0",
        "\u71df\u6536\u589e\u9577",
        "\u91cd\u5927\u71df\u6536",
    )
    GUIDANCE_REACTION_KEYWORDS = (
        "\u6cd5\u8aaa",
        "\u6cd5\u8aaa\u6703",
        "\u71df\u904b\u6307\u5f15",
        "\u6307\u5f15",
        "\u4e0b\u534a\u5e74",
        "\u4e0b\u534a\u5e74\u71df\u904b",
        "\u5c55\u671b",
        "\u8ca1\u6e2c",
        "\u5a92\u9ad4",
        "\u6cd5\u4eba",
        "\u5916\u8cc7",
        "\u53cd\u61c9",
        "\u6703\u5f8c",
        "\u6b63\u9762",
        "\u8ca0\u9762",
    )
    SHIPPING_RATE_IMPACT_KEYWORDS = (
        "\u7d05\u6d77",
        "\u7d05\u6d77\u822a\u7dda",
        "\u822a\u7dda\u53d7\u963b",
        "\u822a\u7dda\u53d7\u963b\u52a0\u5287",
        "scfi",
        "SCFI",
        "\u904b\u50f9",
        "\u904b\u50f9\u6307\u6578",
        "\u96c6\u904b",
        "\u822a\u904b",
        "\u652f\u6490\u529b\u9053",
        "\u5206\u6790\u5e2b",
        "\u6cd5\u4eba",
        "\u5916\u8cc7",
        "\u76ee\u6a19\u50f9",
        "\u76ee\u6a19\u50f9\u8abf\u6574",
        "\u4e0a\u4fee",
        "\u4e0b\u4fee",
    )
    ELECTRICITY_COST_IMPACT_KEYWORDS = (
        "\u5de5\u696d\u96fb\u50f9",
        "\u96fb\u50f9",
        "\u8abf\u6f32",
        "\u6f32\u50f9",
        "\u96fb\u8cbb",
        "\u7528\u96fb\u5927\u6236",
        "\u7528\u96fb",
        "\u6210\u672c",
        "\u6210\u672c\u589e\u52a0",
        "\u984d\u5ea6",
        "\u56e0\u61c9\u5c0d\u7b56",
        "\u5c0d\u7b56",
        "\u7bc0\u80fd",
        "\u7bc0\u96fb",
        "\u8f49\u5ac1",
        "\u7535\u50f9",
    )
    MACRO_YIELD_SENTIMENT_KEYWORDS = (
        "cpi",
        "CPI",
        "\u7f8e\u570bcpi",
        "\u7f8e\u570b cpi",
        "\u901a\u81a8",
        "\u901a\u81a8\u6578\u64da",
        "\u9ad8\u6b96\u5229\u7387",
        "\u6b96\u5229\u7387",
        "\u50b5\u606f",
        "\u5229\u7387",
        "\u91d1\u63a7",
        "\u91d1\u63a7\u80a1",
        "\u9632\u79a6",
        "\u8ca0\u9762\u60c5\u7dd2",
        "\u6cd5\u4eba",
        "\u89c0\u9ede",
        "\u6700\u65b0\u89c0\u9ede",
    )
    # --- Phase 1/2: topic tag infrastructure ---
    # Single authoritative QUESTION_TYPE_FALLBACK_TOPIC_TAGS
    QUESTION_TYPE_FALLBACK_TOPIC_TAGS: dict[str, tuple[str, ...]] = {
        "theme_impact_review": ("題材", "產業"),
        "shipping_rate_impact_review": ("航運", "SCFI"),
        "electricity_cost_impact_review": ("電價", "成本"),
        "macro_yield_sentiment_review": ("CPI", "殖利率"),
        "guidance_reaction_review": ("法說", "指引"),
        "listing_revenue_review": ("上市", "營收"),
        "monthly_revenue_yoy_review": ("月營收",),
        "margin_turnaround_review": ("毛利率", "轉正"),
        "gross_margin_comparison_review": ("毛利率", "比較"),
        "pe_valuation_review": ("本益比",),
        "fundamental_pe_review": ("基本面", "本益比"),
        "price_range": ("股價區間",),
        "price_outlook": ("股價", "展望"),
        "dividend_yield_review": ("股利", "殖利率"),
        "ex_dividend_performance": ("除息", "填息"),
        "fcf_dividend_sustainability_review": ("股利", "現金流"),
        "debt_dividend_safety_review": ("股利", "負債"),
        "profitability_stability_review": ("獲利", "穩定性"),
        "revenue_growth_review": ("營收", "成長"),
        "technical_indicator_review": ("技術面",),
        "season_line_margin_review": ("季線", "籌碼"),
        "earnings_summary": ("財報",),
        "eps_dividend_review": ("EPS", "股利"),
        "investment_support": ("投資評估",),
        "risk_review": ("風險",),
        "announcement_summary": ("公告",),
    }

    _TOPIC_TAG_KEYWORD_MAP: dict[TopicTag, tuple[str, ...]] = {
        # ── 原有分類（擴充關鍵字）──
        TopicTag.SHIPPING: (
            "航運", "集運", "航線", "SCFI", "scfi", "運價", "運價指數", "紅海",
        ),
        TopicTag.ELECTRICITY: (
            "工業電價", "電價", "電費", "調漲", "漲價", "用電",
        ),
        TopicTag.MACRO: (
            "CPI", "cpi", "通膨", "美債", "利率", "殖利率", "降息", "升息",
            "消費者信心", "貿易制裁", "聯準會", "Fed", "fed", "GDP", "gdp",
            "PMI", "pmi", "失業率", "非農",
        ),
        TopicTag.GUIDANCE: (
            "法說", "法說會", "營運指引", "指引", "財測", "展望",
            "業績展望", "營運展望",
        ),
        TopicTag.TECHNICAL: (
            "RSI", "rsi", "KD", "kd", "MACD", "macd", "技術指標", "均線", "超買", "超賣",
            "布林通道", "Bollinger", "bollinger", "ADX", "adx", "ATR", "atr",
            "Beta", "beta", "斐波那契", "Fibonacci", "fibonacci",
            "支撐", "壓力", "支撐位", "壓力位", "跳空", "缺口",
            "K線", "k線", "長紅", "吞噬", "十字星", "錘子線",
            "成交量", "量價背離", "量縮", "量增", "爆量",
            "Volume Profile", "volume profile", "籌碼密集區",
            "資金流向", "Money Flow", "money flow", "MFI", "mfi",
            "股價波動率", "相對強弱",
        ),
        TopicTag.MARGIN_FLOW: (
            "融資", "融券", "信用交易", "籌碼", "季線", "60MA", "60ma", "ma60",
            "借券", "借券賣出", "還券", "券資比",
        ),
        TopicTag.SEMICON_EQUIP: (
            "半導體設備", "設備股", "設備族群", "ASML", "asml", "艾司摩爾",
            "先進封裝", "CoWoS", "cowos",
        ),
        TopicTag.EV: (
            "電動車", "EV", "ev", "電池", "充電樁", "儲能",
        ),
        TopicTag.AI: (
            "AI", "ai", "伺服器", "server", "GPU", "gpu", "HBM", "hbm",
            "大語言模型", "LLM", "llm", "資料中心", "data center",
        ),
        TopicTag.DIVIDEND: (
            "股利", "配息", "現金股利", "除息", "除權", "殖利率",
            "填息", "除權息", "股票股利", "配股", "再投資",
        ),
        TopicTag.REVENUE: (
            "營收", "月營收", "累計營收", "MoM", "mom", "年增", "月增",
            "YoY", "yoy", "QoQ", "qoq", "營收成長率", "季營收",
        ),
        TopicTag.GROSS_MARGIN: (
            "毛利率", "毛利", "營業毛利",
            "產品組合", "成本控管", "毛利率提升",
        ),
        TopicTag.VALUATION: (
            "本益比", "P/E", "p/e", "pe ratio", "估值",
            "本淨比", "P/B", "p/b", "pb ratio",
            "EV/EBITDA", "ev/ebitda", "企業價值倍數",
            "股價營收比", "P/S", "p/s", "ps ratio",
            "PEG", "peg", "股價成長比",
            "DCF", "dcf", "現金流量折現", "內含價值",
            "Forward P/E", "forward p/e", "前瞻本益比",
            "安全邊際", "庫藏股",
        ),
        TopicTag.FUNDAMENTAL: (
            "基本面", "體質", "營運", "EPS", "eps",
            "每股盈餘", "淨利", "Net Income", "net income",
            "財報", "季報", "年報",
        ),
        TopicTag.CASH_FLOW: (
            "現金流", "自由現金流", "FCF", "fcf", "營業現金流",
            "現金流對淨利比", "盈餘品質", "Quality of Earnings",
        ),
        TopicTag.DEBT: (
            "負債", "負債比", "負債比率", "槓桿",
            "利息保障倍數", "Interest Coverage", "interest coverage",
            "財務槓桿", "Debt-to-Equity", "debt-to-equity",
        ),
        TopicTag.LISTING: (
            "上市", "掛牌", "轉上市", "IPO", "ipo",
            "減資", "增資", "現金增資", "GDR", "gdr", "海外存託憑證",
        ),

        # ── 新增分類 ──
        TopicTag.PROFITABILITY: (
            "獲利能力", "淨利率", "營業利益率", "Operating Margin", "operating margin",
            "ROE", "roe", "股東權益報酬率",
            "ROA", "roa", "資產報酬率",
            "業外損益", "業外收入", "業外支出",
        ),
        TopicTag.OPERATING_EFFICIENCY: (
            "營運效率", "存貨週轉", "存貨週轉天數", "Inventory Turnover", "inventory turnover",
            "應收帳款", "應收帳款週轉", "AR Turnover", "ar turnover",
            "速動比率", "流動比率", "流動性",
        ),
        TopicTag.CAPEX_RD: (
            "資本支出", "CAPEX", "capex", "產能擴張", "擴產",
            "研發費用", "研發", "R&D", "r&d", "研發投入", "研發轉化率",
        ),
        TopicTag.INSTITUTIONAL: (
            "外資", "外資買賣超", "投信", "共同基金",
            "內部人", "內部人持股", "董監持股",
            "大股東", "散戶", "散戶持股",
            "官股", "國家隊",
            "目標價", "分析師", "研究報告", "研報",
            "集中度", "持股集中度",
        ),
        TopicTag.COMPETITIVE: (
            "市佔率", "市場佔有率", "Market Share", "market share",
            "護城河", "Moat", "moat", "品牌優勢", "專利",
            "定價能力", "Pricing Power", "pricing power", "轉嫁能力",
            "產品生命週期", "替代技術", "破壞式創新",
            "規模經濟", "網絡效應",
            "垂直整合", "水平擴張",
            "特許經營", "特許門檻", "進入障礙",
            "管理階層", "資本配置", "Capital Allocation",
        ),
        TopicTag.SUPPLY_CHAIN: (
            "供應鏈", "Supply Chain", "supply chain",
            "供應商", "客戶集中度", "主要客戶",
            "供應鏈韌性", "集中風險", "關鍵供應商",
        ),
        TopicTag.ESG: (
            "ESG", "esg", "永續", "永續經營",
            "碳排", "碳中和", "淨零", "減碳",
            "社會責任", "CSR", "csr",
            "公司治理", "數位轉型",
        ),
        TopicTag.REGULATORY: (
            "政策", "監管", "法規", "政策利多", "政策利空",
            "地緣政治", "貿易戰", "制裁",
            "產業政策", "補貼", "反壟斷",
        ),
        TopicTag.EVENT: (
            "併購", "M&A", "m&a", "收購", "合併",
            "訴訟", "官司", "法律糾紛",
            "策略聯盟", "簽約", "合作案",
            "經營權", "董事會改選", "委託書",
            "新產品", "產品發佈", "技術突破",
        ),
        TopicTag.INDEX_REBAL: (
            "MSCI", "msci", "富時", "FTSE", "ftse",
            "權重調整", "指數調整", "成分股", "納入指數", "剔除指數",
        ),
        TopicTag.FX: (
            "匯率", "外匯", "台幣", "美元", "匯損", "匯兌",
            "Exchange Rate", "exchange rate", "日圓", "人民幣",
        ),
        TopicTag.SENTIMENT: (
            "VIX", "vix", "恐慌", "恐慌指數",
            "市場情緒", "情緒過熱", "市場敘事",
            "鈍化", "利空鈍化", "利多鈍化",
            "空單平倉", "Short Squeeze", "short squeeze", "軋空",
        ),
        TopicTag.RISK_MGMT: (
            "停損", "停利", "停損位", "停利點",
            "對沖", "避險", "Hedging", "hedging",
            "最大回撤", "Max Drawdown", "max drawdown",
            "風險承受", "風險承受度",
            "部位配置", "Position Sizing", "position sizing",
            "持有時間", "短線", "長線", "波段",
            "轉機股", "成長股", "定存股", "題材股", "價值股",
        ),
    }

    def __init__(
        self,
        stock_resolver: StockResolver | None = None,
        classifier: QueryClassifier | None = None,
    ) -> None:
        self._stock_resolver = stock_resolver
        self._classifier = classifier

    def parse(self, request: QueryRequest) -> StructuredQuery:
        query_text = request.query.strip()
        stock_mentions = self._extract_stock_mentions(query_text)
        named_company_match = self._extract_named_company_match(query_text)
        explicit_ticker = request.ticker or self._extract_ticker(query_text)
        company_name = None
        comparison_ticker = None
        comparison_company_name = None
        has_name_ticker_conflict = (
            request.ticker is None
            and named_company_match is not None
            and explicit_ticker is not None
            and named_company_match[0] != explicit_ticker
        )

        if has_name_ticker_conflict:
            ticker, company_name = named_company_match
            stock_mentions = [named_company_match]
        else:
            ticker = request.ticker or (stock_mentions[0][0] if stock_mentions else explicit_ticker)

        if stock_mentions and not has_name_ticker_conflict:
            if request.ticker:
                company_name = self._company_name_from_ticker(ticker)
                for mentioned_ticker, mentioned_name in stock_mentions:
                    if mentioned_ticker != ticker:
                        comparison_ticker = mentioned_ticker
                        comparison_company_name = mentioned_name
                        break
            else:
                company_name = stock_mentions[0][1]
                if len(stock_mentions) >= 2:
                    comparison_ticker = stock_mentions[1][0]
                    comparison_company_name = stock_mentions[1][1]

        if ticker is None:
            ticker, company_name = self._extract_company(query_text)
        else:
            company_name = company_name or self._company_name_from_ticker(ticker)
            if company_name is None and self._stock_resolver is not None:
                resolved_ticker, resolved_company_name = self._stock_resolver.resolve(query_text)
                if resolved_ticker == ticker:
                    company_name = resolved_company_name

        classification = self._classify_semantics(
            query_text=query_text,
            request=request,
            stock_mention_count=len(stock_mentions),
        )

        return StructuredQuery(
            user_query=query_text,
            ticker=ticker,
            company_name=company_name,
            comparison_ticker=comparison_ticker,
            comparison_company_name=comparison_company_name,
            topic=classification["topic"],
            time_range_label=classification["time_range_label"],
            time_range_days=classification["time_range_days"],
            intent=classification["intent"],
            controlled_tags=classification["controlled_tags"],
            free_keywords=classification["free_keywords"],
            tag_source=classification["tag_source"],
            question_type=classification["question_type"],
            stance_bias=classification["stance_bias"],
            classifier_source=classification["classifier_source"],
            is_forecast_query=classification["is_forecast_query"],
            wants_direction=classification["wants_direction"],
            wants_scenario_range=classification["wants_scenario_range"],
            forecast_horizon_label=classification["forecast_horizon_label"],
            forecast_horizon_days=classification["forecast_horizon_days"],
        )

    def _classify_semantics(
        self,
        query_text: str,
        request: QueryRequest,
        stock_mention_count: int,
    ) -> dict:
        """LLM-first + B2 per-field validation。

        流程：
          1. 先呼叫規則產出所有欄位（always，作為 fallback source of truth）
          2. 若 self._classifier 存在，呼叫 LLM 取得候選值
          3. 對 LLM 每個欄位獨立驗證，合法就採用、否則沿用規則
          4. 根據「多少欄位最終來自 LLM」決定 classifier_source
        """
        # ── Rule-based fallback values (always computed) ──
        topic = request.topic or self._detect_topic(query_text)
        rule_qtype = self._detect_question_type(query_text, topic, stock_mention_count)
        rule_time_label, rule_time_days = self._detect_time_range(
            query_text, request.time_range, topic, rule_qtype,
        )
        rule_stance = self._detect_stance_bias(query_text)
        rule_is_fc, rule_dir, rule_range, rule_hz_label, rule_hz_days = (
            self._detect_forecast_semantics(query_text, rule_qtype)
        )

        # ── LLM classification (None = pure rule mode) ──
        llm_result: dict | None = None
        if self._classifier is not None:
            llm_result = self._classifier.classify(query_text)
            if not isinstance(llm_result, dict):
                llm_result = None

        # 預設值：任何欄位只在 LLM 值通過驗證時才覆蓋 rule 結果。
        final_qtype: str = rule_qtype
        final_topic_tag_values: list[str] | None = None  # None = 沿用規則的 controlled/free
        final_time_label: str = rule_time_label
        final_stance: StanceBias = rule_stance
        final_is_fc: bool = rule_is_fc
        final_dir: bool = rule_dir
        final_range: bool = rule_range
        final_hz_label: str | None = rule_hz_label
        final_hz_days: int | None = rule_hz_days
        llm_used_count = 0
        rule_used_count = 0

        if llm_result is not None:
            def pick(llm_value, rule_value, validator, transform=lambda value: value):
                if validator(llm_value):
                    return transform(llm_value), True
                return rule_value, False

            final_qtype, used_llm = pick(
                llm_result.get("question_type"),
                rule_qtype,
                lambda value: value in _VALID_QUESTION_TYPES,
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            filtered_topic_tags = (
                list(
                    dict.fromkeys(
                        tag
                        for tag in llm_result.get("topic_tags", [])
                        if isinstance(tag, str) and tag in _VALID_TOPIC_TAG_VALUES
                    )
                )
                if isinstance(llm_result.get("topic_tags"), list)
                else []
            )
            final_topic_tag_values, used_llm = (filtered_topic_tags, True) if filtered_topic_tags else (None, False)
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            final_time_label, used_llm = pick(
                llm_result.get("time_range_label"),
                rule_time_label,
                lambda value: value in _VALID_TIME_RANGE_LABELS,
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            final_stance, used_llm = pick(
                llm_result.get("stance_bias"),
                rule_stance,
                lambda value: value in _VALID_STANCE_VALUES,
                StanceBias,
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            final_is_fc, used_llm = pick(
                llm_result.get("is_forecast_query"),
                rule_is_fc,
                lambda value: isinstance(value, bool),
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            final_dir, used_llm = pick(
                llm_result.get("wants_direction"),
                rule_dir,
                lambda value: isinstance(value, bool),
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            final_range, used_llm = pick(
                llm_result.get("wants_scenario_range"),
                rule_range,
                lambda value: isinstance(value, bool),
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            final_hz_label, used_llm = pick(
                llm_result.get("forecast_horizon_label"),
                rule_hz_label,
                lambda value: value is None or isinstance(value, str),
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)
            final_hz_days, used_llm = pick(
                llm_result.get("forecast_horizon_days"),
                rule_hz_days,
                lambda value: value is None or (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                ),
            )
            llm_used_count += int(used_llm)
            rule_used_count += int(not used_llm)

        # ── 組裝衍生欄位 ──
        intent = infer_intent_from_question_type(final_qtype)
        final_time_days = _TIME_RANGE_DAYS.get(final_time_label, rule_time_days)

        # topic_tags：若 LLM 提供合法清單，轉為 TopicTag；否則用規則抽取
        if final_topic_tag_values is not None:
            controlled_tags = [TopicTag(v) for v in final_topic_tag_values]
            free_keywords: list[str] = []
        else:
            controlled_tags, free_keywords = self._extract_topic_tags(query_text, final_qtype)
        tag_source = "matched" if controlled_tags else "fallback" if free_keywords else "empty"

        # ── classifier_source 判定 ──
        if self._classifier is None or llm_result is None:
            classifier_source = "rule"
        elif rule_used_count == 0:
            classifier_source = "llm"
        else:
            classifier_source = "mixed"

        return {
            "topic": topic,
            "question_type": final_qtype,
            "intent": intent,
            "controlled_tags": controlled_tags,
            "free_keywords": free_keywords,
            "tag_source": tag_source,
            "time_range_label": final_time_label,
            "time_range_days": final_time_days,
            "stance_bias": final_stance,
            "classifier_source": classifier_source,
            "is_forecast_query": final_is_fc,
            "wants_direction": final_dir,
            "wants_scenario_range": final_range,
            "forecast_horizon_label": final_hz_label,
            "forecast_horizon_days": final_hz_days,
        }

    def _extract_ticker(self, query: str) -> str | None:
        match = re.search(r"(?<!\d)(\d{4,6})(?!\d)", query)
        if match:
            return match.group(1)
        return None

    def _extract_company(self, query: str) -> tuple[str | None, str | None]:
        stock_mentions = self._extract_stock_mentions(query)
        if stock_mentions:
            return stock_mentions[0]

        lowered = query.lower()
        compacted = self._compact_query(query)
        for alias, result in self.COMPANY_ALIASES.items():
            if alias.isascii():
                normalized_alias = alias.lower().replace(" ", "")
                if alias in lowered or normalized_alias in compacted:
                    return result
            elif alias in query or alias in compacted:
                return result
        if self._stock_resolver is not None:
            return self._stock_resolver.resolve(query)
        return None, None

    def _extract_named_company_match(self, query: str) -> tuple[str, str] | None:
        lowered = query.lower()
        compacted = self._compact_query(query)
        matches: list[tuple[int, str, str]] = []

        for alias, (ticker, company_name) in self.COMPANY_ALIASES.items():
            if alias.isdigit():
                continue
            position = self._find_alias_position(query, lowered, compacted, alias)
            if position is None:
                continue
            matches.append((position, ticker, company_name))

        if matches:
            _, _, ticker, company_name = min(
                (
                    (position, -len(company_name.replace(" ", "")), ticker, company_name)
                    for position, ticker, company_name in matches
                ),
                key=lambda item: (item[0], item[1]),
            )
            return ticker, company_name

        if self._stock_resolver is not None:
            resolved_ticker, resolved_company_name = self._stock_resolver.resolve(query)
            if resolved_ticker is not None and resolved_company_name is not None:
                return resolved_ticker, resolved_company_name

        return None

    def _extract_stock_mentions(self, query: str) -> list[tuple[str, str]]:
        lowered = query.lower()
        compacted = self._compact_query(query)
        mentions: dict[str, tuple[int, str]] = {}

        for match in re.finditer(r"(?<!\d)(\d{4,6})(?!\d)", query):
            ticker = match.group(1)
            company_name = self._company_name_from_ticker(ticker)
            if company_name is None:
                continue
            existing = mentions.get(ticker)
            if existing is None or match.start() < existing[0]:
                mentions[ticker] = (match.start(), company_name)

        for alias, (ticker, company_name) in self.COMPANY_ALIASES.items():
            position = self._find_alias_position(query, lowered, compacted, alias)
            if position is None:
                continue
            existing = mentions.get(ticker)
            if existing is None or position < existing[0]:
                mentions[ticker] = (position, company_name)

        if not mentions and self._stock_resolver is not None:
            resolved_ticker, resolved_company_name = self._stock_resolver.resolve(query)
            if resolved_ticker is not None and resolved_company_name is not None:
                return [(resolved_ticker, resolved_company_name)]

        ordered = sorted(
            ((position, ticker, company_name) for ticker, (position, company_name) in mentions.items()),
            key=lambda item: (item[0], -len(item[2].replace(" ", ""))),
        )
        return [(ticker, company_name) for _, ticker, company_name in ordered]

    def _find_alias_position(self, query: str, lowered: str, compacted: str, alias: str) -> int | None:
        if alias.isascii():
            lowered_alias = alias.lower()
            direct_position = lowered.find(lowered_alias)
            if direct_position != -1:
                return direct_position
            compact_alias = lowered_alias.replace(" ", "")
            compact_position = compacted.find(compact_alias)
            return compact_position if compact_position != -1 else None

        direct_position = query.find(alias)
        if direct_position != -1:
            return direct_position
        compact_position = compacted.find(alias)
        return compact_position if compact_position != -1 else None

    def _company_name_from_ticker(self, ticker: str) -> str | None:
        for _, result in self.COMPANY_ALIASES.items():
            mapped_ticker, company_name = result
            if mapped_ticker == ticker:
                return company_name
        return None

    def _detect_topic(self, query: str) -> Topic:
        lowered = query.lower()
        compacted = self._compact_query(query)
        matched_topics = [
            topic
            for topic, keywords in self.TOPIC_KEYWORDS.items()
            if any(
                keyword.lower() in lowered
                or keyword in query
                or keyword.lower().replace(" ", "") in compacted
                for keyword in keywords
            )
        ]
        if len(matched_topics) == 1:
            return matched_topics[0]
        return Topic.COMPOSITE

    def _detect_time_range(
        self,
        query: str,
        explicit_range: str | None,
        topic: Topic,
        question_type: str,
    ) -> tuple[str, int]:
        mapping = {
            "1d": ("1d", 1),
            "7d": ("7d", 7),
            "30d": ("30d", 30),
            "1y": ("1y", 365),
            "3y": ("3y", 1095),
            "5y": ("5y", 1825),
            "latest_quarter": ("latest_quarter", 90),
        }

        # Step 1：使用者明確指定時間範圍，最高優先
        if explicit_range in mapping:
            return mapping[explicit_range]

        # Step 2：自然語言關鍵字偵測（移至 policy 查表前，避免被 policy 覆蓋）
        day_match = re.search(r"(?<!\d)(\d{1,3})\s*\u5929", query)
        if day_match:
            days = int(day_match.group(1))
            return f"{days}d", days

        if any(token in query for token in ("\u4eca\u5929", "\u4eca\u65e5", "\u8fd1 1 \u5929", "\u6700\u8fd1 1 \u5929", "\u8fd11\u5929", "\u6700\u8fd11\u5929")):
            return mapping["1d"]
        if any(token in query for token in ("\u6700\u8fd1\u4e00\u9031", "\u8fd1\u4e00\u9031", "\u6700\u8fd1 7 \u5929", "\u8fd1 7 \u5929", "\u8fd17\u5929", "\u6700\u8fd17\u5929", "\u4e00\u9031")):
            return mapping["7d"]
        if any(token in query for token in ("\u6700\u8fd1\u4e00\u500b\u6708", "\u8fd1\u4e00\u500b\u6708", "\u6700\u8fd1 30 \u5929", "\u8fd1 30 \u5929", "\u8fd130\u5929", "\u6700\u8fd130\u5929")):
            return mapping["30d"]
        if any(token in query for token in ("\u904e\u53bb\u4e09\u5e74", "\u8fd1\u4e09\u5e74", "\u6700\u8fd1\u4e09\u5e74", "\u904e\u53bb3\u5e74", "\u8fd13\u5e74", "\u4e09\u5e74")):
            return mapping["3y"]
        if any(token in query for token in ("\u904e\u53bb\u4e94\u5e74", "\u8fd1\u4e94\u5e74", "\u6700\u8fd1\u4e94\u5e74", "\u904e\u53bb5\u5e74", "\u8fd15\u5e74", "\u4e94\u5e74", "\u6b77\u53f2")):
            return mapping["5y"]
        if any(token in query for token in ("\u53bb\u5e74", "\u5168\u5e74", "\u5e74\u5ea6", "\u8fd1\u4e00\u5e74", "\u6700\u8fd1\u4e00\u5e74")):
            return mapping["1y"]
        if any(token in query for token in ("\u6700\u65b0\u4e00\u5b63", "\u6700\u8fd1\u4e00\u5b63", "\u4e0a\u4e00\u5b63", "\u672c\u5b63")):
            return mapping["latest_quarter"]

        # Step 3：向 PolicyRegistry 查詢 question_type 對應的預設時間窗口
        intent = QUESTION_TYPE_TO_INTENT.get(question_type)
        if intent is not None:
            try:
                from llm_stock_system.core.query_policy import get_policy_registry

                policy = get_policy_registry().resolve(intent, question_type)
                if policy.default_time_range_days is not None:
                    label = policy.default_time_range_label or f"{policy.default_time_range_days}d"
                    return label, policy.default_time_range_days
            except Exception:
                pass  # fallback 到下方 question_type 對照表

        # Step 4：原有 question_type 對照表（暫時保留為 fallback，待所有 policy 補齊後移除）
        if question_type == "monthly_revenue_yoy_review":
            return mapping["1y"]
        if question_type == "fundamental_pe_review":
            return mapping["1y"]
        if question_type == "price_outlook":
            return mapping["30d"]
        if question_type == "guidance_reaction_review":
            return mapping["30d"]
        if question_type == "shipping_rate_impact_review":
            return mapping["30d"]
        if question_type == "electricity_cost_impact_review":
            return mapping["30d"]
        if question_type == "macro_yield_sentiment_review":
            return mapping["30d"]
        if question_type == "listing_revenue_review":
            return mapping["30d"]
        if question_type == "profitability_stability_review":
            return mapping["5y"]
        if question_type == "fcf_dividend_sustainability_review":
            return mapping["3y"]
        if question_type == "pe_valuation_review":
            return mapping["1y"]
        if question_type == "gross_margin_comparison_review":
            return mapping["latest_quarter"]
        if question_type == "debt_dividend_safety_review":
            return mapping["3y"]
        if question_type in {"eps_dividend_review", "dividend_yield_review", "ex_dividend_performance"}:
            return mapping["1y"]
        if question_type == "revenue_growth_review":
            return mapping["latest_quarter"]
        if question_type == "theme_impact_review":
            return mapping["30d"]
        if question_type == "season_line_margin_review":
            return "90d", 90
        if question_type == "technical_indicator_review":
            return mapping["30d"]

        # Step 5：topic 與最終 fallback
        if topic == Topic.EARNINGS:
            return mapping["latest_quarter"]
        return mapping["7d"]

    def _detect_question_type(self, query: str, topic: Topic, stock_mention_count: int = 0) -> str:
        lowered = query.lower()
        compacted = self._compact_query(query)
        has_comparison = stock_mention_count >= 2 or any(
            keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.COMPARISON_KEYWORDS
        )
        has_gross_margin = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.GROSS_MARGIN_KEYWORDS
        )
        has_turnaround = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.TURNAROUND_KEYWORDS
        )
        has_profitability_stability = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.PROFITABILITY_STABILITY_KEYWORDS
        )
        has_eps = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.EPS_KEYWORDS
        )
        has_monthly_revenue = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.MONTHLY_REVENUE_KEYWORDS
        )
        has_pe_valuation = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.PE_VALUATION_KEYWORDS
        )
        has_fundamental_overview = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.FUNDAMENTAL_OVERVIEW_KEYWORDS
        )
        has_combined_fundamental_prompt = any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in ("\u540c\u6642", "\u4e00\u8d77", "\u7d9c\u5408", "\u4e00\u8d77\u770b")
        )
        has_fcf = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.FCF_KEYWORDS
        )
        has_debt_ratio = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.DEBT_RATIO_KEYWORDS
        )
        has_cash_balance = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.CASH_BALANCE_KEYWORDS
        )
        has_revenue_growth = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.REVENUE_GROWTH_KEYWORDS
        )
        has_dividend = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.DIVIDEND_KEYWORDS
        )
        has_dividend_total = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.DIVIDEND_TOTAL_KEYWORDS
        )
        has_sustainability = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.SUSTAINABILITY_KEYWORDS
        )
        has_technical = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.TECHNICAL_INDICATOR_KEYWORDS
        )
        has_theme_impact = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.THEME_IMPACT_KEYWORDS
        )
        has_season_line = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.SEASON_LINE_KEYWORDS
        )
        has_margin_balance = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.MARGIN_BALANCE_KEYWORDS
        )
        has_dividend_yield = any(
            keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.DIVIDEND_YIELD_KEYWORDS
        )
        has_shipping_rate_impact = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.SHIPPING_RATE_IMPACT_KEYWORDS
        )
        has_electricity_cost_impact = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.ELECTRICITY_COST_IMPACT_KEYWORDS
        )
        has_macro_yield_sentiment = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.MACRO_YIELD_SENTIMENT_KEYWORDS
        )
        has_asset_disposal = any(
            keyword in lowered or keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.ASSET_DISPOSAL_KEYWORDS
        )
        has_ex_dividend = any(
            keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.EX_DIVIDEND_KEYWORDS
        )
        has_price_level_query = bool(
            re.search(
                r"(\u7a81\u7834|\u7ad9\u4e0a|\u8dcc\u7834|\u5931\u5b88|\u5b88\u4f4f|\u7ad9\u7a69|\u4e0a\u770b|\u6311\u6230)\s*\d[\d,]*(?:\.\d+)?",
                query,
            )
        ) and any(
            token in query or token.lower().replace(" ", "") in compacted or "\u55ce" in query
            for token in self.PRICE_LEVEL_FUTURE_HINTS
        )

        # --- Forecast pre-check: future time window + direction/range demand ---
        # Must run BEFORE price_range so that "這一個星期預估區間" routes to
        # price_outlook, not price_range.
        has_forecast_time_window = any(
            keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.FORECAST_TIME_WINDOW_KEYWORDS
        )
        has_forecast_demand = any(
            keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.FORECAST_DEMAND_KEYWORDS
        )
        if has_forecast_time_window and has_forecast_demand:
            return "price_outlook"

        if any(keyword in query or keyword.lower().replace(" ", "") in compacted for keyword in self.PRICE_RANGE_KEYWORDS):
            return "price_range"
        if has_gross_margin and has_turnaround and stock_mention_count < 2:
            return "margin_turnaround_review"
        if has_comparison and has_gross_margin and stock_mention_count >= 2:
            return "gross_margin_comparison_review"
        if has_profitability_stability and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in ("\u4e94\u5e74", "5\u5e74", "\u904e\u53bb\u4e94\u5e74", "\u8fd1\u4e94\u5e74", "\u6bcf\u5e74", "\u54ea\u4e00\u5e74")
        ):
            return "profitability_stability_review"
        has_guidance_core = any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in ("\u6cd5\u8aaa", "\u6cd5\u8aaa\u6703", "\u71df\u904b\u6307\u5f15", "\u6307\u5f15", "\u8ca1\u6e2c")
        )
        has_guidance_reaction = any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in ("\u4e0b\u534a\u5e74", "\u5a92\u9ad4", "\u6cd5\u4eba", "\u5916\u8cc7", "\u53cd\u61c9", "\u6703\u5f8c", "\u6b63\u9762", "\u8ca0\u9762")
        )
        if has_guidance_core and has_guidance_reaction and any(
            keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.GUIDANCE_REACTION_KEYWORDS
        ):
            return "guidance_reaction_review"
        if has_shipping_rate_impact and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "\u7d05\u6d77",
                "\u822a\u7dda",
                "scfi",
                "\u904b\u50f9",
                "\u904b\u50f9\u6307\u6578",
                "\u96c6\u904b",
                "\u822a\u904b",
            )
        ) and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "\u652f\u6490",
                "\u652f\u6490\u529b\u9053",
                "\u5206\u6790\u5e2b",
                "\u6cd5\u4eba",
                "\u5916\u8cc7",
                "\u76ee\u6a19\u50f9",
                "\u8abf\u6574",
                "\u4e0a\u4fee",
                "\u4e0b\u4fee",
            )
        ):
            return "shipping_rate_impact_review"
        if has_electricity_cost_impact and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "\u5de5\u696d\u96fb\u50f9",
                "\u96fb\u50f9",
                "\u8abf\u6f32",
                "\u6f32\u50f9",
                "\u96fb\u8cbb",
                "\u7528\u96fb\u5927\u6236",
            )
        ) and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "\u6210\u672c",
                "\u6210\u672c\u589e\u52a0",
                "\u984d\u5ea6",
                "\u56e0\u61c9\u5c0d\u7b56",
                "\u5c0d\u7b56",
                "\u7bc0\u80fd",
                "\u8f49\u5ac1",
            )
        ):
            return "electricity_cost_impact_review"
        if has_macro_yield_sentiment and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "cpi",
                "CPI",
                "\u901a\u81a8",
                "\u7f8e\u570b",
                "\u7f8e\u570bcpi",
                "\u7f8e\u50b5",
                "\u50b5\u606f",
                "\u964d\u606f",
                "\u5347\u606f",
            )
        ) and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "\u9ad8\u6b96\u5229\u7387",
                "\u6b96\u5229\u7387",
                "\u91d1\u63a7",
                "\u91d1\u63a7\u80a1",
                "\u4e2d\u83ef\u96fb",
                "\u8ca0\u9762\u60c5\u7dd2",
                "\u6cd5\u4eba",
                "\u89c0\u9ede",
            )
        ):
            return "macro_yield_sentiment_review"
        if any(
            keyword in query or keyword.lower().replace(" ", "") in compacted
            for keyword in self.LISTING_VOLATILITY_KEYWORDS
        ) and has_monthly_revenue:
            return "listing_revenue_review"
        if has_monthly_revenue and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "\u7d2f\u8a08",
                "\u540c\u671f",
                "\u5e74\u589e",
                "\u524d\u4e09\u500b\u6708",
                "\u524d3\u500b\u6708",
                "\u524d\u4e09\u6708",
                "\u6708\u589e",
                "\u6708\u589e\u7387",
                "mom",
                "\u5275\u65b0\u9ad8",
                "\u8fd1\u4e00\u5e74\u65b0\u9ad8",
                "\u8fd11\u5e74\u65b0\u9ad8",
                "\u55ae\u6708",
                "\u5e7e\u6708",
                "\u5e02\u5834\u89e3\u8b80",
                "\u7a81\u767c\u6210\u9577",
                "\u51fa\u8868",
                "\u516c\u4f48",
            )
        ):
            return "monthly_revenue_yoy_review"
        if has_pe_valuation and (
            has_fundamental_overview
            or (
                has_combined_fundamental_prompt
                and any(
                    (
                        has_eps,
                        has_monthly_revenue,
                        has_dividend,
                        has_fcf,
                        has_revenue_growth,
                        has_gross_margin,
                    )
                )
            )
        ):
            return "fundamental_pe_review"
        if has_pe_valuation:
            return "pe_valuation_review"
        if has_debt_ratio and (has_cash_balance or has_dividend):
            return "debt_dividend_safety_review"
        if has_fcf and has_dividend and (has_dividend_total or has_sustainability):
            return "fcf_dividend_sustainability_review"
        if has_revenue_growth and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in ("AI", "ai", "\u4f3a\u670d\u5668", "server", "\u71df\u6536")
        ):
            return "revenue_growth_review"
        if has_asset_disposal and any(
            token in lowered or token in query or token.lower().replace(" ", "") in compacted
            for token in ("\u8655\u5206", "\u571f\u5730", "\u8cc7\u7522", "\u696d\u5916", "\u5165\u5e33", "\u8a8d\u5217", "eps")
        ):
            return "announcement_summary"
        if has_price_level_query:
            return "price_outlook"
        if any(keyword in query or keyword.lower().replace(" ", "") in compacted for keyword in self.PRICE_OUTLOOK_KEYWORDS):
            return "price_outlook"
        if has_theme_impact and any(
            token in query or token.lower().replace(" ", "") in compacted
            for token in (
                "\u96fb\u52d5\u8eca",
                "ev",
                "\u96fb\u6c60",
                "\u4f9b\u61c9\u93c8",
                "\u666e\u53ca\u7387",
                "\u9700\u6c42",
                "\u534a\u5c0e\u9ad4\u8a2d\u5099",
                "\u8a2d\u5099\u65cf\u7fa4",
                "\u8a2d\u5099\u80a1",
                "\u8a2d\u5099",
                "asml",
                "\u827e\u53f8\u6469\u723e",
                "\u5229\u7a7a",
                "\u60c5\u7dd2",
                "\u4e0d\u5982\u9810\u671f",
                "\u5c55\u671b",
            )
        ):
            return "theme_impact_review"
        if has_season_line or has_margin_balance:
            return "season_line_margin_review"
        if has_technical:
            return "technical_indicator_review"
        if has_ex_dividend and any(
            token in query or token in compacted
            for token in ("\u586b\u606f", "\u586b\u6b0a", "\u586b\u6b0a\u606f", "\u5e02\u5834\u53cd\u61c9", "\u7576\u5929")
        ):
            return "ex_dividend_performance"
        if has_dividend and has_dividend_yield:
            return "dividend_yield_review"
        if has_eps and has_dividend:
            return "eps_dividend_review"
        if any(token in query for token in ("\u53ef\u4ee5\u8cb7", "\u503c\u5f97\u8cb7", "\u80fd\u8cb7\u55ce", "\u8cb7\u9032", "\u9032\u5834")):
            return "investment_support"
        if any(token in query for token in ("\u98a8\u96aa", "\u6ce8\u610f\u4ec0\u9ebc", "\u6709\u54ea\u4e9b\u98a8\u96aa")):
            return "risk_review"
        if topic == Topic.EARNINGS:
            return "earnings_summary"
        if topic == Topic.ANNOUNCEMENT:
            return "announcement_summary"
        return "market_summary"

    def _extract_topic_tags(
        self, query: str, question_type: str
    ) -> tuple[list["TopicTag"], list[str]]:
        """Extract topic tags from the user query.

        Returns ``(controlled_tags, free_keywords)`` where:
        - ``controlled_tags``: matched ``TopicTag`` enum values (categorical signal
          for routing and observability)
        - ``free_keywords``: the specific raw keywords that triggered each tag
          (used as search terms in the Retrieval Layer)

        Falls back to ``QUESTION_TYPE_FALLBACK_TOPIC_TAGS`` (as plain strings in
        ``free_keywords``) when no keyword in the query matches the map.
        """
        from llm_stock_system.core.enums import TopicTag as _TopicTag

        lowered = query.lower()
        compacted = self._compact_query(query)
        seen_kw: set[str] = set()

        # 第一輪：收集所有命中的 (tag, keyword) 對（原始匹配，未過濾子字串）
        raw_hits: list[tuple[_TopicTag, str]] = []

        for tag, keywords in self._TOPIC_TAG_KEYWORD_MAP.items():
            for kw in keywords:
                if kw in seen_kw:
                    continue
                if (
                    kw.lower() in lowered
                    or kw in query
                    or kw.lower().replace(" ", "") in compacted
                ):
                    seen_kw.add(kw)
                    raw_hits.append((tag, kw))

        # 第二輪：過濾子字串誤判（Bug 2 修正）
        # 若某關鍵字是另一個已命中關鍵字的嚴格子字串，則視為誤判並移除。
        # 例：「毛利率」命中 GROSS_MARGIN 後，「利率」在 MACRO 的命中應被過濾。
        all_matched_kw: set[str] = {kw for _, kw in raw_hits}
        filtered_hits = [
            (tag, kw) for (tag, kw) in raw_hits
            if not any(kw != other and kw in other for other in all_matched_kw)
        ]

        # 整理回傳，保留原始出現順序
        tag_order: list[_TopicTag] = []
        seen_tags: set[_TopicTag] = set()
        matched_keywords: list[str] = []

        for tag, kw in filtered_hits:
            if tag not in seen_tags:
                seen_tags.add(tag)
                tag_order.append(tag)
            matched_keywords.append(kw)

        if tag_order:
            return tag_order, matched_keywords

        fallback = self.QUESTION_TYPE_FALLBACK_TOPIC_TAGS.get(question_type, ())
        return [], list(fallback)

    def _detect_forecast_semantics(
        self, query: str, question_type: str,
    ) -> tuple[bool, bool, bool, str | None, int | None]:
        """Return (is_forecast, wants_direction, wants_range, horizon_label, horizon_days).

        A query is a forecast query if question_type == price_outlook AND
        it contains forward-looking time/demand signals.
        """
        if question_type != "price_outlook":
            return False, False, False, None, None

        compacted = self._compact_query(query)

        # Determine if the query has forward-looking signals
        has_time_window = any(
            kw in query or kw.lower().replace(" ", "") in compacted
            for kw in self.FORECAST_TIME_WINDOW_KEYWORDS
        )
        has_demand = any(
            kw in query or kw.lower().replace(" ", "") in compacted
            for kw in self.FORECAST_DEMAND_KEYWORDS
        )
        has_outlook_kw = any(
            kw in query or kw.lower().replace(" ", "") in compacted
            for kw in self.PRICE_OUTLOOK_KEYWORDS
        )
        is_forecast = has_outlook_kw or (has_time_window and has_demand)

        if not is_forecast:
            return False, False, False, None, None

        # Direction demand
        direction_tokens = (
            "漲", "跌", "偏多", "偏空", "看漲", "看跌", "會上漲", "會下跌",
            "漲還是跌", "走勢", "目標價",
        )
        wants_direction = any(t in query for t in direction_tokens)

        # Range demand
        range_tokens = ("區間", "預估區間", "波動", "波動如何", "高低點")
        wants_range = any(t in query for t in range_tokens)

        # Horizon detection
        horizon_label, horizon_days = self._detect_forecast_horizon(query)

        return is_forecast, wants_direction, wants_range, horizon_label, horizon_days

    def _detect_forecast_horizon(self, query: str) -> tuple[str | None, int | None]:
        """Parse forward-looking horizon from the query text."""
        compacted = self._compact_query(query)

        horizon_map: list[tuple[tuple[str, ...], str, int]] = [
            (("明天", "明日"), "明天", 1),
            (("後天",), "後天", 2),
            (("這週", "這一週", "這個星期", "這一個星期"), "本週", 7),
            (("下週", "下一週", "下個星期"), "下週", 7),
            (("一週", "一個星期", "一星期", "7天"), "未來一週", 7),
            (("兩週", "兩個星期", "二週", "14天"), "未來兩週", 14),
            (("一個月", "下個月", "30天"), "未來一個月", 30),
            (("一季", "三個月"), "未來一季", 90),
            (("半年", "六個月"), "未來半年", 180),
            (("一年",), "未來一年", 365),
        ]

        for tokens, label, days in horizon_map:
            if any(t in query or t.lower().replace(" ", "") in compacted for t in tokens):
                return label, days

        # Default: if a future window keyword exists but no specific length, assume one week
        if any(t in query or t.lower().replace(" ", "") in compacted for t in self.FORECAST_TIME_WINDOW_KEYWORDS):
            return "未來", 7

        return None, None

    def _detect_stance_bias(self, query: str) -> StanceBias:
        compacted = self._compact_query(query)
        if any(token in query or token in compacted for token in ("\u770b\u591a", "\u5229\u591a", "\u6703\u6f32", "\u53ef\u4ee5\u8cb7", "\u503c\u5f97\u8cb7")):
            return StanceBias.BULLISH
        if any(token in query or token in compacted for token in ("\u770b\u7a7a", "\u5229\u7a7a", "\u6703\u8dcc", "\u4e0d\u80fd\u8cb7", "\u8ce3\u51fa")):
            return StanceBias.BEARISH
        return StanceBias.NEUTRAL

    def _compact_query(self, query: str) -> str:
        return "".join(query.lower().split())
