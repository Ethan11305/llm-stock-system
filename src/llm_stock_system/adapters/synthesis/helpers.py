import re

from llm_stock_system.core.enums import SourceTier, StanceBias
from llm_stock_system.core.models import GovernanceReport, StructuredQuery


def extract_number(report: GovernanceReport, pattern: str) -> str | None:
    compiled = re.compile(pattern)
    for evidence in report.evidence:
        match = compiled.search(evidence.excerpt)
        if match:
            return match.group(1)
    return None


def extract_text(report: GovernanceReport, pattern: str) -> str | None:
    compiled = re.compile(pattern)
    for evidence in report.evidence:
        match = compiled.search(evidence.excerpt)
        if match:
            return match.group(1)
    return None


def extract_price_range(report: GovernanceReport) -> tuple[str | None, str | None]:
    high_pattern = re.compile(r"最高價(?:為)?\s*(\d+(?:\.\d+)?)\s*元")
    low_pattern = re.compile(r"最低價(?:為)?\s*(\d+(?:\.\d+)?)\s*元")
    for evidence in report.evidence:
        high_match = high_pattern.search(evidence.excerpt)
        low_match = low_pattern.search(evidence.excerpt)
        if high_match and low_match:
            return high_match.group(1), low_match.group(1)
    return None, None


def evidence_contains(report: GovernanceReport, keywords: tuple[str, ...]) -> bool:
    for evidence in report.evidence:
        haystack = f"{evidence.title} {evidence.excerpt}".lower()
        if any(keyword.lower() in haystack for keyword in keywords):
            return True
    return False


def listing_event_label(query: StructuredQuery) -> str:
    lowered = query.user_query.lower()
    if "ipo" in lowered or "initial public offering" in lowered:
        return "IPO 後"
    if "掛牌" in query.user_query:
        return "掛牌後"
    return "上市後"


def build_summary_fallback(query: StructuredQuery, report: GovernanceReport) -> str:
    label = query.company_name or query.ticker or "此標的"
    official_count = sum(1 for item in report.evidence if item.source_tier == SourceTier.HIGH)
    if official_count:
        return f"{label} 目前有官方或高可信來源可供整理，但現有資訊仍應與後續公告一併觀察。"
    return f"{label} 目前有部分可參考資料，但來源強度與一致性仍需持續確認。"


def build_risks_generic(query: StructuredQuery) -> list[str]:
    if query.stance_bias != StanceBias.NEUTRAL:
        return [
            "當問題帶有單一立場時，容易忽略相反證據。",
            "若只挑選支持既有看法的資訊，判讀偏誤會被放大。",
            "投資判斷仍應同時檢查利多、利空與不確定因素。",
        ]
    return [
        "若資料主要集中在單一時間窗，可能忽略更長期的基本面變化。",
        "若後續出現更新公告，現有摘要需要重新驗證。",
        "不同來源對同一事件的解讀可能不同，需持續比對。",
    ]


def build_impacts_generic() -> list[str]:
    return [
        "現有資料可作為快速掌握脈絡的起點。",
        "若不同來源能互相印證，判讀穩定度會更高。",
        "後續仍應持續追蹤新的公告、財報與新聞變化。",
    ]
