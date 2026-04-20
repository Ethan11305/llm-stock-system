from llm_stock_system.core.models import GovernanceReport, StructuredQuery
from llm_stock_system.core.fundamental_valuation import (
    build_fundamental_valuation_summary,
    is_fundamental_valuation_question,
)

from .helpers import build_risks_generic, build_summary_fallback


class InvestmentAssessmentStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        tags = set(query.topic_tags)
        label = query.company_name or query.ticker or "此標的"

        if "公告" in tags:
            lead_title = report.evidence[0].title if report.evidence else "最新公告"
            return f"{label} 近期有可追蹤的公告或高可信資料更新，重點包括：{lead_title}。"

        if is_fundamental_valuation_question(query):
            return build_fundamental_valuation_summary(query, report)

        return build_summary_fallback(query, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        if is_fundamental_valuation_question(query):
            return [
                "基本面可以幫助判斷獲利與營運動能是否具備延續性。",
                "本益比則可以拿來定位市場目前給這檔公司的估值水位。",
                "若要做進場判斷，最好是把獲利趨勢和估值位置一起看。",
            ]
        tags = set(query.topic_tags)
        if "公告" in tags:
            return [
                "公告通常會影響市場對公司短期事件與決策節奏的理解。",
                "若公告涉及股利、董事會或法說，可能提高後續關注度。",
                "公告本身仍需搭配財報與新聞交叉驗證，避免過度解讀。",
            ]
        return [
            "短線漲跌通常同時受到基本面、資金面與消息面影響。",
            "若近期有公告或財報更新，市場反應可能放大波動。",
            "缺少多來源佐證時，任何方向判斷都應保守看待。",
        ]

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        if is_fundamental_valuation_question(query):
            return [
                "估值高低會隨股價與獲利預期變化，今天的本益比位置不代表之後不會再調整。",
                "基本面如果只有單一季或單月證據，不一定能代表中長期獲利趨勢。",
                "單看本益比或單看營收都可能失真，仍要搭配現金流、資本支出和產業景氣一起解讀。",
            ]
        tags = set(query.topic_tags)
        if "公告" in tags:
            return [
                "公告標題容易被快速解讀，仍需回看完整揭露內容。",
                "若只有單一公告而缺乏交叉來源，市場解讀可能偏差。",
                "公告對股價的影響常受時間點與市場情緒放大或鈍化。",
            ]
        return build_risks_generic(query)
