import re

from llm_stock_system.core.enums import Intent, TopicTag
from llm_stock_system.core.models import GovernanceReport, StructuredQuery


_FUNDAMENTAL_KEYWORDS = (
    "eps",
    "\u71df\u6536",
    "\u7372\u5229",
    "\u8ca1\u5831",
    "\u57fa\u672c\u9762",
    "\u9ad4\u8cea",
    "\u6bdb\u5229",
    "\u6bdb\u5229\u7387",
    "\u73fe\u91d1\u6d41",
)
_VALUATION_KEYWORDS = (
    "\u672c\u76ca\u6bd4",
    "pe ratio",
    "p/e",
    "\u4f30\u503c",
    "\u6b77\u53f2\u5206\u4f4d",
    "valuation",
)
def is_fundamental_valuation_question(query: StructuredQuery) -> bool:
    """判斷是否為需要基本面估值視角的問題。

    路由依據：intent + topic_tags（含 QUESTION_TYPE_FALLBACK_TOPIC_TAGS 注入的最小集），
    不再讀取 question_type。

    topic_tags 由 StructuredQuery model validator 保證永遠包含該 question_type 的
    QUESTION_TYPE_FALLBACK_TOPIC_TAGS，因此 fundamental_pe_review / investment_support
    等型別不需要額外的 question_type fallback。
    """
    topic_tags = set(query.topic_tags or [])

    if query.intent == Intent.VALUATION_CHECK:
        # 同時含基本面（TopicTag.FUNDAMENTAL）與估值（TopicTag.VALUATION）標籤
        return {
            TopicTag.FUNDAMENTAL.value,
            TopicTag.VALUATION.value,
        }.issubset(topic_tags)

    if query.intent == Intent.INVESTMENT_ASSESSMENT:
        # 含基本面、估值或投資評估任一標籤即觸發
        return any(
            tag in topic_tags
            for tag in (
                TopicTag.FUNDAMENTAL.value,
                TopicTag.VALUATION.value,
                "投資評估",
            )
        )

    return False


def has_fundamental_evidence(governance_report: GovernanceReport) -> bool:
    haystack = _combined_text(governance_report)
    lowered = haystack.lower()
    return any(keyword in lowered for keyword in _FUNDAMENTAL_KEYWORDS)


def has_valuation_evidence(governance_report: GovernanceReport) -> bool:
    haystack = _combined_text(governance_report)
    lowered = haystack.lower()
    return any(keyword in lowered for keyword in _VALUATION_KEYWORDS)


def build_fundamental_valuation_summary(query: StructuredQuery, governance_report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    fundamental_text = _build_fundamental_summary_segment(governance_report)
    valuation_text = _build_valuation_summary_segment(governance_report)

    if fundamental_text and valuation_text:
        return (
            f"{label}\u57fa\u672c\u9762\u4f86\u770b\uff0c{fundamental_text}\uff1b"
            f"\u4f30\u503c\u9762\u4f86\u770b\uff0c{valuation_text}\u3002"
            "\u82e5\u8981\u5224\u65b7\u662f\u5426\u9069\u5408\u9032\u5834\uff0c\u8981\u628a\u7372\u5229\u5ef6\u7e8c\u6027\u548c\u76ee\u524d\u672c\u76ca\u6bd4\u4f4d\u7f6e\u4e00\u8d77\u770b\u3002"
        )
    if fundamental_text:
        return (
            f"{label}\u57fa\u672c\u9762\u4f86\u770b\uff0c{fundamental_text}\uff1b"
            "\u4f46\u76ee\u524d\u7f3a\u5c11\u53ef\u76f4\u63a5\u7528\u4f86\u5224\u8b80\u672c\u76ca\u6bd4\u4f4d\u7f6e\u7684\u4f30\u503c\u8b49\u64da\uff0c"
            "\u9084\u4e0d\u9069\u5408\u55ae\u7378\u5c0d\u6295\u8cc7\u5438\u5f15\u529b\u4e0b\u7d50\u8ad6\u3002"
        )
    if valuation_text:
        return (
            f"{label}\u4f30\u503c\u9762\u4f86\u770b\uff0c{valuation_text}\uff1b"
            "\u4f46\u76ee\u524d\u7f3a\u5c11\u8db3\u5920\u7684\u57fa\u672c\u9762\u91cf\u5316\u8b49\u64da\uff0c"
            "\u4e0d\u9069\u5408\u53ea\u9760\u672c\u76ca\u6bd4\u9ad8\u4f4e\u5c31\u5224\u65b7\u662f\u5426\u503c\u5f97\u9032\u5834\u3002"
        )
    return f"\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u540c\u6642\u78ba\u8a8d{label}\u7684\u57fa\u672c\u9762\u8207\u672c\u76ca\u6bd4\u4f30\u503c\u3002"


def build_fundamental_valuation_highlights(
    query: StructuredQuery,
    governance_report: GovernanceReport,
) -> list[str]:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    highlights = [
        _build_fundamental_highlight(label, governance_report),
        _build_valuation_highlight(label, governance_report),
        "\u6574\u9ad4\u5224\u8b80\u8981\u540c\u6642\u770b\u7372\u5229\u5ef6\u7e8c\u6027\u8207\u672c\u76ca\u6bd4\u6240\u8655\u4f4d\u7f6e\uff0c\u4e0d\u5efa\u8b70\u55ae\u770b\u55ae\u4e00\u908a\u3002",
    ]
    return [item for item in highlights if item][:3]


def build_fundamental_valuation_facts(
    query: StructuredQuery,
    governance_report: GovernanceReport,
) -> list[str]:
    label = query.company_name or query.ticker or "\u8a72\u516c\u53f8"
    facts = [
        _build_fundamental_fact(label, governance_report),
        _build_valuation_fact(label, governance_report),
    ]
    return [item for item in facts if item][:3]


def _build_fundamental_summary_segment(governance_report: GovernanceReport) -> str:
    annual_eps = _extract_number(governance_report, r"\u5168\u5e74 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
    latest_quarter_eps = _extract_number(governance_report, r"\u6700\u65b0\u4e00\u5b63 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
    revenue_yoy = _extract_number(governance_report, r"\u5e74\u589e (\d+(?:\.\d+)?)%")

    if annual_eps:
        return f"\u53bb\u5e74\u5168\u5e74 EPS \u7d04 {annual_eps} \u5143"
    if latest_quarter_eps:
        return f"\u6700\u65b0\u4e00\u5b63 EPS \u7d04 {latest_quarter_eps} \u5143"
    if revenue_yoy:
        return f"\u8fd1\u671f\u71df\u6536\u5e74\u589e\u7d04 {revenue_yoy}%"
    if has_fundamental_evidence(governance_report):
        return "\u6709\u57fa\u672c\u9762\u76f8\u95dc\u8b49\u64da\uff0c\u4f46\u76ee\u524d\u7f3a\u5c11\u53ef\u76f4\u63a5\u91cf\u5316\u7684 EPS \u6216\u71df\u6536\u6210\u9577\u6307\u6a19"
    return ""


def _build_valuation_summary_segment(governance_report: GovernanceReport) -> str:
    current_pe = _extract_number(governance_report, r"\u672c\u76ca\u6bd4\u7d04 (\d+(?:\.\d+)?) \u500d")
    low_pe = _extract_number(governance_report, r"\u672c\u76ca\u6bd4\u5340\u9593\u7d04 (\d+(?:\.\d+)?) \u81f3")
    high_pe = _extract_number(governance_report, r"\u81f3 (\d+(?:\.\d+)?) \u500d")
    percentile = _extract_number(governance_report, r"\u6b77\u53f2\u5206\u4f4d (\d+(?:\.\d+)?)%")
    valuation_zone = _extract_text(governance_report, r"\u5c6c(\u6b77\u53f2\u504f\u9ad8\u5340|\u6b77\u53f2\u504f\u4f4e\u5340|\u6b77\u53f2\u4e2d\u6bb5\u5340)")

    if current_pe and low_pe and high_pe and valuation_zone:
        return (
            f"\u76ee\u524d\u672c\u76ca\u6bd4\u7d04 {current_pe} \u500d\uff0c"
            f"\u82e5\u653e\u56de\u8fd1 13 \u500b\u6708\u6b77\u53f2\u5340\u9593 {low_pe} \u81f3 {high_pe} \u500d\u89c0\u5bdf\uff0c"
            f"\u76ee\u524d\u5c6c{valuation_zone}"
        )
    if current_pe and percentile:
        return f"\u76ee\u524d\u672c\u76ca\u6bd4\u7d04 {current_pe} \u500d\uff0c\u7d04\u843d\u5728\u8fd1 13 \u500b\u6708\u6b77\u53f2\u5206\u4f4d {percentile}%"
    if current_pe:
        return f"\u76ee\u524d\u672c\u76ca\u6bd4\u7d04 {current_pe} \u500d"
    if has_valuation_evidence(governance_report):
        return "\u6709\u4f30\u503c\u76f8\u95dc\u8b49\u64da\uff0c\u4f46\u76ee\u524d\u4ecd\u7f3a\u5c11\u53ef\u76f4\u63a5\u5b9a\u4f4d\u6b77\u53f2\u9ad8\u4f4e\u7684\u5b8c\u6574\u672c\u76ca\u6bd4\u6578\u5b57"
    return ""


def _build_fundamental_highlight(label: str, governance_report: GovernanceReport) -> str:
    segment = _build_fundamental_summary_segment(governance_report)
    if segment:
        return f"\u57fa\u672c\u9762\uff1a{label}{segment}\u3002"
    return f"\u57fa\u672c\u9762\uff1a\u76ee\u524d\u7f3a\u5c11{label}\u53ef\u76f4\u63a5\u5f15\u7528\u7684 EPS \u6216\u71df\u6536\u6210\u9577\u91cf\u5316\u8b49\u64da\u3002"


def _build_valuation_highlight(label: str, governance_report: GovernanceReport) -> str:
    segment = _build_valuation_summary_segment(governance_report)
    if segment:
        return f"\u672c\u76ca\u6bd4\uff1a{label}{segment}\u3002"
    return f"\u672c\u76ca\u6bd4\uff1a\u76ee\u524d\u7f3a\u5c11{label}\u53ef\u76f4\u63a5\u5b9a\u4f4d\u6b77\u53f2\u5340\u9593\u7684\u4f30\u503c\u6578\u64da\u3002"


def _build_fundamental_fact(label: str, governance_report: GovernanceReport) -> str:
    annual_eps = _extract_number(governance_report, r"\u5168\u5e74 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
    latest_quarter_eps = _extract_number(governance_report, r"\u6700\u65b0\u4e00\u5b63 EPS \u7d04 (\d+(?:\.\d+)?) \u5143")
    revenue_yoy = _extract_number(governance_report, r"\u5e74\u589e (\d+(?:\.\d+)?)%")

    if annual_eps:
        return f"{label}\u53bb\u5e74\u5168\u5e74 EPS \u7d04 {annual_eps} \u5143\u3002"
    if latest_quarter_eps:
        return f"{label}\u6700\u65b0\u4e00\u5b63 EPS \u7d04 {latest_quarter_eps} \u5143\u3002"
    if revenue_yoy:
        return f"{label}\u8fd1\u671f\u71df\u6536\u5e74\u589e\u7d04 {revenue_yoy}%\u3002"
    if has_fundamental_evidence(governance_report):
        return f"\u4f86\u6e90\u4e2d\u6709{label}\u7684\u57fa\u672c\u9762\u8a0a\u606f\uff0c\u4f46\u7f3a\u5c11\u53ef\u76f4\u63a5\u5f15\u7528\u7684 EPS \u6216\u71df\u6536\u6210\u9577\u6578\u5b57\u3002"
    return f"\u73fe\u6709\u4f86\u6e90\u672a\u63d0\u4f9b{label}\u8db3\u5920\u7684\u57fa\u672c\u9762\u91cf\u5316\u8b49\u64da\u3002"


def _build_valuation_fact(label: str, governance_report: GovernanceReport) -> str:
    current_pe = _extract_number(governance_report, r"\u672c\u76ca\u6bd4\u7d04 (\d+(?:\.\d+)?) \u500d")
    percentile = _extract_number(governance_report, r"\u6b77\u53f2\u5206\u4f4d (\d+(?:\.\d+)?)%")
    valuation_zone = _extract_text(governance_report, r"\u5c6c(\u6b77\u53f2\u504f\u9ad8\u5340|\u6b77\u53f2\u504f\u4f4e\u5340|\u6b77\u53f2\u4e2d\u6bb5\u5340)")

    if current_pe and percentile:
        return f"{label}\u76ee\u524d\u672c\u76ca\u6bd4\u7d04 {current_pe} \u500d\uff0c\u6b77\u53f2\u5206\u4f4d\u7d04 {percentile}%\u3002"
    if current_pe and valuation_zone:
        return f"{label}\u76ee\u524d\u672c\u76ca\u6bd4\u7d04 {current_pe} \u500d\uff0c\u76ee\u524d\u5c6c{valuation_zone}\u3002"
    if current_pe:
        return f"{label}\u76ee\u524d\u672c\u76ca\u6bd4\u7d04 {current_pe} \u500d\u3002"
    if has_valuation_evidence(governance_report):
        return f"\u4f86\u6e90\u4e2d\u6709{label}\u7684\u672c\u76ca\u6bd4\u6216\u4f30\u503c\u8a0e\u8ad6\uff0c\u4f46\u7f3a\u5c11\u5b8c\u6574\u5b9a\u4f4d\u6578\u5b57\u3002"
    return f"\u73fe\u6709\u4f86\u6e90\u672a\u63d0\u4f9b{label}\u8db3\u5920\u7684\u672c\u76ca\u6bd4\u4f30\u503c\u8b49\u64da\u3002"


def _extract_number(governance_report: GovernanceReport, pattern: str) -> str | None:
    compiled = re.compile(pattern)
    for evidence in governance_report.evidence:
        haystack = f"{evidence.title} {evidence.excerpt}"
        match = compiled.search(haystack)
        if match:
            return match.group(1)
    return None


def _extract_text(governance_report: GovernanceReport, pattern: str) -> str | None:
    compiled = re.compile(pattern)
    for evidence in governance_report.evidence:
        haystack = f"{evidence.title} {evidence.excerpt}"
        match = compiled.search(haystack)
        if match:
            return match.group(1)
    return None


def _combined_text(governance_report: GovernanceReport) -> str:
    return " ".join(f"{item.title} {item.excerpt}" for item in governance_report.evidence)
