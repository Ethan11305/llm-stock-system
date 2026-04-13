import re

from llm_stock_system.core.models import GovernanceReport, StructuredQuery


_TARGET_PRICE_KEYWORDS = (
    "\u76ee\u6a19\u50f9",
    "\u76ee\u6807\u50f9",
    "\u76ee\u6a19\u50f9\u4f4d",
    "target price",
)
_TARGET_CONTEXT_KEYWORDS = _TARGET_PRICE_KEYWORDS + (
    "\u6cd5\u4eba",
    "\u5206\u6790\u5e2b",
    "\u5916\u8cc7",
    "\u5238\u5546",
    "\u6295\u9867",
    "\u8a55\u7b49",
    "\u4e0a\u4fee",
    "\u4e0b\u4fee",
)
_PRICE_LEVEL_ACTION_KEYWORDS = (
    "\u7a81\u7834",
    "\u7ad9\u4e0a",
    "\u8dcc\u7834",
    "\u5931\u5b88",
    "\u5b88\u4f4f",
    "\u7ad9\u7a69",
    "\u4e0a\u770b",
    "\u6311\u6230",
)
_PRICE_LEVEL_FUTURE_HINTS = (
    "\u672a\u4f86",
    "\u4e0b\u534a\u5e74",
    "\u4e0a\u534a\u5e74",
    "\u534a\u5e74",
    "\u4e00\u5e74",
    "\u4e00\u5b63",
    "\u4e00\u500b\u6708",
    "\u4eca\u5e74",
    "\u660e\u5e74",
    "\u6709\u6a5f\u6703",
    "\u53ef\u80fd",
    "\u80fd\u5426",
    "\u6703\u4e0d\u6703",
    "\u662f\u5426",
    "\u53ef\u4e0d\u53ef\u80fd",
    "\u53ef\u4ee5\u55ce",
)
_FORWARD_PRICE_CONTEXT_KEYWORDS = _TARGET_CONTEXT_KEYWORDS + (
    "\u58d3\u529b",
    "\u652f\u6490",
    "\u95dc\u5361",
    "\u6574\u6578\u95dc\u5361",
    "\u524d\u9ad8",
    "\u6b77\u53f2\u9ad8\u9ede",
    "\u6280\u8853\u9762",
    "\u5747\u7dda",
    "\u6ce2\u6bb5",
    "\u4ef7\u4f4d",
)
_TARGET_PRICE_RANGE_PATTERNS = (
    re.compile(
        "(?:\u76ee\u6a19\u50f9(?:\u4f4d)?|\u76ee\u6807\u50f9(?:\u4f4d)?|target\\s*price)[^0-9]{0,12}"
        "(\\d+(?:\\.\\d+)?)\\s*(?:-|~|\u81f3|\u5230)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:\u5143)?",
        re.IGNORECASE,
    ),
)
_TARGET_PRICE_SINGLE_PATTERNS = (
    re.compile(
        "(?:\u76ee\u6a19\u50f9(?:\u4f4d)?|\u76ee\u6807\u50f9(?:\u4f4d)?|target\\s*price)[^0-9]{0,12}"
        "(\\d+(?:\\.\\d+)?)\\s*(?:\u5143)?",
        re.IGNORECASE,
    ),
)
_PRICE_LEVEL_QUERY_PATTERNS = (
    re.compile(
        "(\u7a81\u7834|\u7ad9\u4e0a|\u8dcc\u7834|\u5931\u5b88|\u5b88\u4f4f|\u7ad9\u7a69|\u4e0a\u770b|\u6311\u6230)"
        "\\s*(\\d[\\d,]*(?:\\.\\d+)?)",
    ),
    re.compile(
        "(\\d[\\d,]*(?:\\.\\d+)?)\\s*(?:\u5143)?[^0-9]{0,8}"
        "(\u95dc\u5361|\u58d3\u529b|\u652f\u6490)",
    ),
)


def is_target_price_question(query: StructuredQuery) -> bool:
    compact_query = _compact_text(query.user_query)
    return any(keyword.replace(" ", "") in compact_query for keyword in _TARGET_PRICE_KEYWORDS)


def is_price_level_question(query: StructuredQuery) -> bool:
    compact_query = _compact_text(query.user_query)
    has_action = any(keyword in compact_query for keyword in _PRICE_LEVEL_ACTION_KEYWORDS)
    has_future_hint = any(keyword in compact_query for keyword in _PRICE_LEVEL_FUTURE_HINTS) or "\u55ce" in compact_query
    return has_action and has_future_hint and extract_price_level_value(query) is not None


def is_forward_price_question(query: StructuredQuery) -> bool:
    return is_target_price_question(query) or is_price_level_question(query)


def extract_target_price_values(governance_report: GovernanceReport) -> list[float]:
    values: set[float] = set()

    for evidence in governance_report.evidence:
        haystack = f"{evidence.title} {evidence.excerpt}"
        for pattern in _TARGET_PRICE_RANGE_PATTERNS:
            for match in pattern.finditer(haystack):
                for group in match.groups():
                    _append_target_price(values, group)
        for pattern in _TARGET_PRICE_SINGLE_PATTERNS:
            for match in pattern.finditer(haystack):
                _append_target_price(values, match.group(1))

    return sorted(values)


def has_target_price_context(governance_report: GovernanceReport) -> bool:
    compact_haystack = _compact_text(
        " ".join(f"{evidence.title} {evidence.excerpt}" for evidence in governance_report.evidence)
    )
    return any(keyword.replace(" ", "") in compact_haystack for keyword in _TARGET_CONTEXT_KEYWORDS)


def has_forward_price_context(query: StructuredQuery, governance_report: GovernanceReport) -> bool:
    compact_haystack = _compact_text(
        " ".join(f"{evidence.title} {evidence.excerpt}" for evidence in governance_report.evidence)
    )
    return (
        has_target_price_context(governance_report)
        or has_price_level_context(query, governance_report)
        or any(keyword.replace(" ", "") in compact_haystack for keyword in _FORWARD_PRICE_CONTEXT_KEYWORDS)
    )


def has_price_level_context(query: StructuredQuery, governance_report: GovernanceReport) -> bool:
    level_value = extract_price_level_value(query)
    if level_value is None:
        return False

    integer_value = int(level_value) if float(level_value).is_integer() else None
    variants = {
        _format_price(level_value),
        f"{level_value:.2f}".rstrip("0").rstrip("."),
    }
    if integer_value is not None:
        variants.add(str(integer_value))
        variants.add(f"{integer_value:,}")

    for evidence in governance_report.evidence:
        haystack = f"{evidence.title} {evidence.excerpt}"
        compact_haystack = _compact_text(haystack)
        mentions_level = any(variant.replace(",", "") in compact_haystack for variant in variants)
        mentions_level_keywords = any(keyword in compact_haystack for keyword in _PRICE_LEVEL_ACTION_KEYWORDS) or any(
            keyword in compact_haystack for keyword in ("\u95dc\u5361", "\u58d3\u529b", "\u652f\u6490", "\u524d\u9ad8")
        )
        if mentions_level and mentions_level_keywords:
            return True
    return False


def target_price_horizon_label(query: StructuredQuery) -> str:
    compact_query = _compact_text(query.user_query)
    if any(token in compact_query for token in ("\u534a\u5e74", "\u516d\u500b\u6708", "6\u500b\u6708")):
        return "\u672a\u4f86\u534a\u5e74"
    if any(token in compact_query for token in ("\u4e00\u5e74", "1\u5e74", "\u5341\u4e8c\u500b\u6708", "12\u500b\u6708")):
        return "\u672a\u4f86\u4e00\u5e74"
    if any(token in compact_query for token in ("\u4e09\u500b\u6708", "3\u500b\u6708", "\u4e00\u5b63", "\u4e00\u500b\u5b63\u5ea6")):
        return "\u672a\u4f86\u4e00\u5b63"
    return "\u672a\u4f86"


def extract_price_level_value(query: StructuredQuery) -> float | None:
    for pattern in _PRICE_LEVEL_QUERY_PATTERNS:
        match = pattern.search(query.user_query)
        if not match:
            continue
        numeric_groups = [group for group in match.groups() if group and any(char.isdigit() for char in group)]
        if not numeric_groups:
            continue
        return _parse_price_value(numeric_groups[0])
    return None


def price_level_action_label(query: StructuredQuery) -> str:
    compact_query = _compact_text(query.user_query)
    for keyword in _PRICE_LEVEL_ACTION_KEYWORDS:
        if keyword in compact_query:
            return keyword
    return "\u89f8\u53ca"


def build_forward_price_summary(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    if is_target_price_question(query):
        return build_target_price_summary(query, governance_report)
    if is_price_level_question(query):
        return build_price_level_summary(query, governance_report)
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    return (
        f"{label}\u7684\u77ed\u671f\u6f32\u8dcc\u7121\u6cd5\u50c5\u9760\u76ee\u524d\u516c\u958b\u8cc7\u8a0a\u76f4\u63a5\u9810\u6e2c\uff0c"
        "\u4ecd\u9700\u7d50\u5408\u5f8c\u7e8c\u516c\u544a\u3001\u8ca1\u5831\u3001\u65b0\u805e\u8207\u50f9\u683c\u884c\u70ba\u6301\u7e8c\u89c0\u5bdf\u3002"
    )


def build_forward_price_highlight(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    if is_target_price_question(query):
        return build_target_price_highlight(query, governance_report)
    if is_price_level_question(query):
        return build_price_level_highlight(query, governance_report)
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    return f"\u73fe\u6709\u4f86\u6e90\u53ef\u63d0\u4f9b{label}\u7684\u65b9\u5411\u6027\u8a0a\u865f\uff0c\u4f46\u4e0d\u8db3\u4ee5\u76f4\u63a5\u9810\u6e2c\u672a\u4f86\u50f9\u4f4d\u3002"


def build_forward_price_fact(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    if is_target_price_question(query):
        return build_target_price_fact(query, governance_report)
    if is_price_level_question(query):
        return build_price_level_fact(query, governance_report)
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    return f"\u73fe\u6709\u4f86\u6e90\u4e3b\u8981\u63d0\u4f9b{label}\u7684\u57fa\u672c\u9762\u8207\u65b0\u805e\u8a0a\u865f\uff0c\u4e0d\u5c6c\u65bc\u76f4\u63a5\u50f9\u4f4d\u9810\u6e2c\u8b49\u64da\u3002"


def build_target_price_summary(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    target_price_values = extract_target_price_values(governance_report)
    horizon_label = target_price_horizon_label(query)

    if target_price_values:
        if len(target_price_values) == 1:
            return (
                f"\u73fe\u6709\u4f86\u6e90\u6709\u63d0\u5230{label}\u7684\u76ee\u6a19\u50f9\u7d04\u70ba"
                f"{_format_price(target_price_values[0])}\u5143\uff0c\u4f46\u9019\u985e\u6578\u503c\u901a\u5e38\u4f86\u81ea\u55ae\u4e00\u6cd5\u4eba"
                f"\u6216\u7279\u5b9a\u6642\u9ede\uff0c\u4e0d\u5b9c\u76f4\u63a5\u8996\u70ba{horizon_label}\u7684\u4e00\u81f4\u9810\u671f\u3002"
            )
        return (
            f"\u73fe\u6709\u4f86\u6e90\u63d0\u5230{label}\u7684\u76ee\u6a19\u50f9\u7d04\u843d\u5728"
            f"{_format_price(target_price_values[0])}\u5230{_format_price(target_price_values[-1])}\u5143\u5340\u9593\uff0c"
            f"\u4f46\u9019\u4e9b\u6578\u503c\u591a\u534a\u4ecd\u5c6c\u65bc\u55ae\u4e00\u6cd5\u4eba\u6216\u5831\u5c0e\u89c0\u9ede\uff0c"
            f"\u4e0d\u5b9c\u76f4\u63a5\u8996\u70ba{horizon_label}\u7684\u4e00\u81f4\u76ee\u6a19\u50f9\u3002"
        )

    if has_target_price_context(governance_report):
        return (
            f"\u73fe\u6709\u4f86\u6e90\u63d0\u5230{label}\u7684\u6cd5\u4eba\u6216\u5206\u6790\u5e2b\u89c0\u9ede\uff0c"
            f"\u4f46\u6c92\u6709\u76f4\u63a5\u63ed\u9732\u53ef\u5c0d\u61c9{horizon_label}\u7684\u660e\u78ba\u76ee\u6a19\u50f9\u6578\u503c\uff0c"
            "\u66ab\u6642\u53ea\u80fd\u6574\u7406\u65b9\u5411\u6027\u8a0a\u865f\u3002"
        )

    return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}{horizon_label}\u7684\u660e\u78ba\u76ee\u6a19\u50f9\u3002"


def build_price_level_summary(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    horizon_label = target_price_horizon_label(query)
    action = price_level_action_label(query)
    level_value = extract_price_level_value(query)
    formatted_level = _format_price(level_value) if level_value is not None else "\u6307\u5b9a\u50f9\u4f4d"

    if has_price_level_context(query, governance_report):
        return (
            f"\u73fe\u6709\u4f86\u6e90\u6709\u63d0\u5230{label}{formatted_level}\u5143\u9644\u8fd1\u7684\u95dc\u5361\u6216\u50f9\u4f4d\u8a0e\u8ad6\uff0c"
            f"\u4f46\u4ecd\u4e0d\u8db3\u4ee5\u76f4\u63a5\u78ba\u8a8d{horizon_label}\u80fd\u5426{action}{formatted_level}\u5143\u3002"
        )

    if extract_target_price_values(governance_report):
        return (
            f"\u73fe\u6709\u4f86\u6e90\u63d0\u5230{label}\u7684\u76ee\u6a19\u50f9\u6216\u50f9\u4f4d\u6578\u503c\uff0c"
            f"\u4f46\u6c92\u6709\u76f4\u63a5\u8ad6\u8b49{horizon_label}\u80fd\u5426{action}{formatted_level}\u5143\uff0c"
            "\u66ab\u6642\u53ea\u80fd\u6574\u7406\u65b9\u5411\u6027\u8a0a\u865f\u3002"
        )

    if has_forward_price_context(query, governance_report):
        return (
            f"\u73fe\u6709\u4f86\u6e90\u53ea\u63d0\u4f9b{label}\u7684\u57fa\u672c\u9762\u3001\u6cd5\u4eba\u6216\u6280\u8853\u9762\u65b9\u5411\u6027\u8a0a\u865f\uff0c"
            f"\u6c92\u6709\u76f4\u63a5\u8a0e\u8ad6{formatted_level}\u5143\u95dc\u5361\u80fd\u5426\u5728{horizon_label}{action}\uff0c\u66ab\u6642\u53ea\u80fd\u4fdd\u5b88\u89e3\u8b80\u3002"
        )

    return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d{label}{horizon_label}\u80fd\u5426{action}{formatted_level}\u5143\u3002"


def build_target_price_highlight(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    horizon_label = target_price_horizon_label(query)

    if extract_target_price_values(governance_report):
        return (
            f"\u4f86\u6e90\u4e2d\u6709\u76f4\u63a5\u63d0\u5230{label}\u7684\u76ee\u6a19\u50f9\u6578\u503c\uff0c"
            f"\u4f46\u5c0d{horizon_label}\u4ecd\u9700\u6ce8\u610f\u5176\u6642\u9ede\u8207\u6cd5\u4eba\u80cc\u666f\u3002"
        )
    if has_target_price_context(governance_report):
        return (
            f"\u4f86\u6e90\u53ea\u63d0\u5230{label}\u7684\u6cd5\u4eba\u6216\u5206\u6790\u5e2b\u65b9\u5411\u6027\u770b\u6cd5\uff0c"
            "\u6c92\u6709\u76f4\u63a5\u7d66\u51fa\u660e\u78ba\u76ee\u6a19\u50f9\u3002"
        )
    return f"\u76ee\u524d\u4f86\u6e90\u6c92\u6709\u8db3\u5920\u7684\u76ee\u6a19\u50f9\u6216\u6cd5\u4eba\u8b49\u64da\uff0c\u7121\u6cd5\u652f\u6490\u56de\u7b54{label}\u7684\u76ee\u6a19\u50f9\u554f\u984c\u3002"


def build_price_level_highlight(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    level_value = extract_price_level_value(query)
    formatted_level = _format_price(level_value) if level_value is not None else "\u6307\u5b9a\u50f9\u4f4d"
    action = price_level_action_label(query)

    if has_price_level_context(query, governance_report):
        return f"\u4f86\u6e90\u6709\u63d0\u5230{label}{formatted_level}\u5143\u9644\u8fd1\u7684\u95dc\u5361\u8a0e\u8ad6\uff0c\u4f46\u4ecd\u4e0d\u80fd\u8996\u70ba\u78ba\u5b9a{action}\u7684\u8b49\u64da\u3002"
    if has_forward_price_context(query, governance_report):
        return f"\u4f86\u6e90\u53ea\u63d0\u4f9b{label}\u7684\u65b9\u5411\u6027\u8a0a\u865f\uff0c\u6c92\u6709\u76f4\u63a5\u8a0e\u8ad6{formatted_level}\u5143\u95dc\u5361\u3002"
    return f"\u76ee\u524d\u4f86\u6e90\u672a\u63d0\u4f9b\u53ef\u5c0d\u61c9{label}{formatted_level}\u5143\u50f9\u4f4d\u7684\u524d\u77bb\u8b49\u64da\u3002"


def build_target_price_fact(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    target_price_values = extract_target_price_values(governance_report)

    if target_price_values:
        if len(target_price_values) == 1:
            return f"\u53ef\u64f7\u53d6\u5230\u7684{label}\u76ee\u6a19\u50f9\u6578\u503c\u7d04\u70ba{_format_price(target_price_values[0])}\u5143\u3002"
        return (
            f"\u53ef\u64f7\u53d6\u5230\u7684{label}\u76ee\u6a19\u50f9\u5340\u9593\u7d04\u70ba"
            f"{_format_price(target_price_values[0])}\u5230{_format_price(target_price_values[-1])}\u5143\u3002"
        )
    if has_target_price_context(governance_report):
        return f"\u4f86\u6e90\u63d0\u5230{label}\u7684\u6cd5\u4eba\u6216\u5206\u6790\u5e2b\u89c0\u9ede\uff0c\u4f46\u672a\u63d0\u4f9b\u76f4\u63a5\u76ee\u6a19\u50f9\u6578\u5b57\u3002"
    return f"\u73fe\u6709\u4f86\u6e90\u672a\u63d0\u4f9b\u53ef\u76f4\u63a5\u5c0d\u61c9{label}\u7684\u76ee\u6a19\u50f9\u8cc7\u6599\u3002"


def build_price_level_fact(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    level_value = extract_price_level_value(query)
    formatted_level = _format_price(level_value) if level_value is not None else "\u6307\u5b9a\u50f9\u4f4d"

    if has_price_level_context(query, governance_report):
        return f"\u4f86\u6e90\u4e2d\u6709\u63d0\u5230{label}{formatted_level}\u5143\u95dc\u5361\u6216\u9644\u8fd1\u50f9\u4f4d\u7684\u76f8\u95dc\u8a0e\u8ad6\u3002"
    if extract_target_price_values(governance_report):
        return f"\u4f86\u6e90\u63d0\u5230{label}\u7684\u76ee\u6a19\u50f9\u6216\u50f9\u4f4d\u6578\u503c\uff0c\u4f46\u4e0d\u662f\u76f4\u63a5\u5c0d\u61c9{formatted_level}\u5143\u95dc\u5361\u7684\u7a81\u7834\u8b49\u64da\u3002"
    if has_forward_price_context(query, governance_report):
        return f"\u4f86\u6e90\u4e3b\u8981\u63d0\u4f9b{label}\u7684\u65b9\u5411\u6027\u6216\u6280\u8853\u9762\u89c0\u9ede\uff0c\u672a\u76f4\u63a5\u5206\u6790{formatted_level}\u5143\u50f9\u4f4d\u3002"
    return f"\u73fe\u6709\u4f86\u6e90\u672a\u76f4\u63a5\u63d0\u4f9b{label}{formatted_level}\u5143\u50f9\u4f4d\u7684\u5206\u6790\u8b49\u64da\u3002"


def _append_target_price(values: set[float], raw_value: str) -> None:
    try:
        value = float(raw_value)
    except ValueError:
        return
    if 0 < value < 10000:
        values.add(round(value, 2))


def _compact_text(text: str) -> str:
    return "".join(text.lower().split())


def _parse_price_value(raw_value: str) -> float | None:
    try:
        value = float(raw_value.replace(",", ""))
    except ValueError:
        return None
    if 0 < value < 100000:
        return round(value, 2)
    return None


def _format_price(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")
