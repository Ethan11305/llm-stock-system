import re

from llm_stock_system.core.enums import StanceBias, Topic, TopicTag
from llm_stock_system.core.interfaces import StockResolver
from llm_stock_system.core.models import QueryRequest, StructuredQuery, infer_intent_from_question_type


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
    PRICE_OUTLOOK_KEYWORDS = ("\u6f32\u8dcc\u9810\u6e2c", "\u6703\u6f32\u55ce", "\u6703\u8dcc\u55ce", "\u8d70\u52e2", "\u9810\u6e2c", "\u76ee\u6a19\u50f9")
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
        TopicTag.SHIPPING: ("\u822a\u904b", "\u96c6\u904b", "\u822a\u7dda", "SCFI", "scfi", "\u904b\u50f9", "\u904b\u50f9\u6307\u6578", "\u7d05\u6d77"),
        TopicTag.ELECTRICITY: ("\u5de5\u696d\u96fb\u50f9", "\u96fb\u50f9", "\u96fb\u8cbb", "\u8abf\u6f32", "\u6f32\u50f9", "\u7528\u96fb"),
        TopicTag.MACRO: ("CPI", "cpi", "\u901a\u81a8", "\u7f8e\u50b5", "\u5229\u7387", "\u6b96\u5229\u7387", "\u964d\u606f", "\u5347\u606f"),
        TopicTag.GUIDANCE: ("\u6cd5\u8aaa", "\u6cd5\u8aaa\u6703", "\u71df\u904b\u6307\u5f15", "\u6307\u5f15", "\u8ca1\u6e2c", "\u5c55\u671b"),
        TopicTag.TECHNICAL: ("RSI", "rsi", "KD", "kd", "MACD", "macd", "\u6280\u8853\u6307\u6a19", "\u5747\u7dda", "\u8d85\u8cb7", "\u8d85\u8ce3"),
        TopicTag.MARGIN_FLOW: ("\u878d\u8cc7", "\u878d\u5238", "\u4fe1\u7528\u4ea4\u6613", "\u7c4c\u78bc", "\u5b63\u7dda", "60MA", "60ma", "ma60"),
        TopicTag.SEMICON_EQUIP: ("\u534a\u5c0e\u9ad4\u8a2d\u5099", "\u8a2d\u5099\u80a1", "\u8a2d\u5099\u65cf\u7fa4", "ASML", "asml", "\u827e\u53f8\u6469\u723e"),
        TopicTag.EV: ("\u96fb\u52d5\u8eca", "EV", "ev", "\u96fb\u6c60", "\u4f9b\u61c9\u93c8"),
        TopicTag.AI: ("AI", "ai", "\u4f3a\u670d\u5668", "server"),
        TopicTag.DIVIDEND: ("\u80a1\u5229", "\u914d\u606f", "\u73fe\u91d1\u80a1\u5229", "\u9664\u606f", "\u9664\u6b0a", "\u6b96\u5229\u7387"),
        TopicTag.REVENUE: ("\u71df\u6536", "\u6708\u71df\u6536", "\u7d2f\u8a08\u71df\u6536", "MoM", "mom", "\u5e74\u589e", "\u6708\u589e"),
        TopicTag.GROSS_MARGIN: ("\u6bdb\u5229\u7387", "\u6bdb\u5229", "\u71df\u696d\u6bdb\u5229"),
        TopicTag.VALUATION: ("\u672c\u76ca\u6bd4", "P/E", "p/e", "pe ratio", "\u4f30\u503c"),
        TopicTag.FUNDAMENTAL: ("\u57fa\u672c\u9762", "\u9ad4\u8cea", "\u71df\u904b", "EPS", "eps"),
        TopicTag.CASH_FLOW: ("\u73fe\u91d1\u6d41", "\u81ea\u7531\u73fe\u91d1\u6d41", "FCF", "fcf", "\u71df\u696d\u73fe\u91d1\u6d41"),
        TopicTag.DEBT: ("\u8ca0\u50b5", "\u8ca0\u50b5\u6bd4", "\u8ca0\u50b5\u6bd4\u7387", "\u69d3\u687f"),
        TopicTag.LISTING: ("\u4e0a\u5e02", "\u639b\u724c", "\u8f49\u4e0a\u5e02", "IPO", "ipo"),
    }

    def __init__(self, stock_resolver: StockResolver | None = None) -> None:
        self._stock_resolver = stock_resolver

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

        topic = request.topic or self._detect_topic(query_text)
        question_type = self._detect_question_type(query_text, topic, len(stock_mentions))
        intent = infer_intent_from_question_type(question_type)
        controlled_tags, free_keywords = self._extract_topic_tags(query_text, question_type)
        tag_source = "matched" if controlled_tags else "fallback" if free_keywords else "empty"
        time_range_label, time_range_days = self._detect_time_range(
            query_text,
            request.time_range,
            topic,
            question_type,
        )
        stance_bias = self._detect_stance_bias(query_text)

        return StructuredQuery(
            user_query=query_text,
            ticker=ticker,
            company_name=company_name,
            comparison_ticker=comparison_ticker,
            comparison_company_name=comparison_company_name,
            topic=topic,
            time_range_label=time_range_label,
            time_range_days=time_range_days,
            intent=intent,
            controlled_tags=controlled_tags,
            free_keywords=free_keywords,
            tag_source=tag_source,
            question_type=question_type,
            stance_bias=stance_bias,
        )

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

        if explicit_range in mapping:
            return mapping[explicit_range]

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
            or any(
                token in query or token.lower().replace(" ", "") in compacted
                for token in ("\u540c\u6642", "\u4e00\u8d77", "\u7d9c\u5408", "\u4e00\u8d77\u770b", "\u503c\u4e0d\u503c\u5f97", "\u6295\u8cc7")
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
        matched_tags: list[_TopicTag] = []
        matched_keywords: list[str] = []

        for tag, keywords in self._TOPIC_TAG_KEYWORD_MAP.items():
            kws_hit = [
                kw for kw in keywords
                if (
                    kw.lower() in lowered
                    or kw in query
                    or kw.lower().replace(" ", "") in compacted
                )
                and kw not in seen_kw
            ]
            if kws_hit:
                matched_tags.append(tag)
                seen_kw.update(kws_hit)
                matched_keywords.extend(kws_hit)

        if matched_tags:
            return matched_tags, matched_keywords

        fallback = self.QUESTION_TYPE_FALLBACK_TOPIC_TAGS.get(question_type, ())
        return [], list(fallback)

    def _detect_stance_bias(self, query: str) -> StanceBias:
        compacted = self._compact_query(query)
        if any(token in query or token in compacted for token in ("\u770b\u591a", "\u5229\u591a", "\u6703\u6f32", "\u53ef\u4ee5\u8cb7", "\u503c\u5f97\u8cb7")):
            return StanceBias.BULLISH
        if any(token in query or token in compacted for token in ("\u770b\u7a7a", "\u5229\u7a7a", "\u6703\u8dcc", "\u4e0d\u80fd\u8cb7", "\u8ce3\u51fa")):
            return StanceBias.BEARISH
        return StanceBias.NEUTRAL

    def _compact_query(self, query: str) -> str:
        return "".join(query.lower().split())