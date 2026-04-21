from uuid import uuid4

from llm_stock_system.core.enums import ConfidenceLight, QueryProfile
from llm_stock_system.core.models import (
    AnswerDraft,
    DataStatus,
    GovernanceReport,
    QueryResponse,
    StructuredQuery,
    ValidationResult,
)


class PresentationLayer:
    DISCLAIMER = "\u672c\u7cfb\u7d71\u50c5\u6574\u7406\u516c\u958b\u8cc7\u8a0a\uff0c\u4e0d\u69cb\u6210\u6295\u8cc7\u5efa\u8b70\u3002"

    def present(
        self,
        query: StructuredQuery | None,
        answer_draft: AnswerDraft,
        governance_report: GovernanceReport,
        validation_result: ValidationResult,
    ) -> QueryResponse:
        """組裝 QueryResponse。

        P3 UI parity：從 ``query`` 拉 ``classifier_source`` 與 ``query_profile``，
        從 ``validation_result`` 拉 ``warnings``，讓即時回應與 ``QueryLogDetail``
        看得到相同的 meta。為避免破壞既有單元測試，``query`` 允許傳 ``None``，
        此時三個欄位全部用預設值（warnings=[], classifier_source="rule",
        query_profile=LEGACY）。生產流程請務必傳入。
        """
        if validation_result.confidence_light == ConfidenceLight.RED:
            summary = self._red_summary(answer_draft.summary)
            highlights = answer_draft.highlights[:3] or [
                "\u73fe\u6709\u8b49\u64da\u4e0d\u8db3\u6216\u4e00\u81f4\u6027\u4e0d\u8db3\uff0c\u7cfb\u7d71\u5df2\u964d\u7d1a\u56de\u7b54\u3002"
            ]
            facts = answer_draft.facts[:1] or [
                "\u5c1a\u672a\u53d6\u5f97\u8db3\u5920\u7684\u5b98\u65b9\u516c\u544a\u3001\u65b0\u805e\u6216\u8ca1\u5831\u8cc7\u6599\u3002"
            ]
            impacts = answer_draft.impacts[:1] or [
                "\u8cc7\u6599\u4e0d\u8db3\u6642\uff0c\u4e0d\u61c9\u5c07\u55ae\u4e00\u8a0a\u606f\u76f4\u63a5\u8996\u70ba\u6295\u8cc7\u5224\u65b7\u4f9d\u64da\u3002"
            ]
            risks = answer_draft.risks[:4] or [
                "\u8cc7\u6599\u4e0d\u8db3\u6642\uff0c\u5bb9\u6613\u8aa4\u628a\u55ae\u4e00\u8a0a\u606f\u7576\u6210\u8da8\u52e2\u3002",
                "\u82e5\u53ea\u4f9d\u8cf4\u672a\u9a57\u8b49\u8cc7\u8a0a\uff0c\u53ef\u80fd\u653e\u5927\u5224\u65b7\u504f\u8aa4\u3002",
                "\u5efa\u8b70\u7b49\u5f85\u66f4\u591a\u516c\u544a\u3001\u8ca1\u5831\u6216\u4e3b\u6d41\u4f86\u6e90\u66f4\u65b0\u3002",
            ]
        else:
            summary = answer_draft.summary
            highlights = answer_draft.highlights[:3]
            facts = answer_draft.facts[:3]
            impacts = answer_draft.impacts[:3]
            risks = answer_draft.risks[:4]

        classifier_source = query.classifier_source if query is not None else "rule"
        query_profile = query.query_profile if query is not None else QueryProfile.LEGACY

        return QueryResponse(
            query_id=str(uuid4()),
            summary=summary,
            highlights=highlights,
            facts=facts,
            impacts=impacts,
            risks=risks,
            dataStatus=DataStatus(
                sufficiency=governance_report.sufficiency,
                consistency=governance_report.consistency,
                freshness=governance_report.freshness,
            ),
            confidenceLight=validation_result.confidence_light,
            confidenceScore=validation_result.confidence_score,
            sources=answer_draft.sources,
            disclaimer=self.DISCLAIMER,
            warnings=list(validation_result.warnings),
            classifierSource=classifier_source,
            queryProfile=query_profile,
        )

    def _red_summary(self, summary: str) -> str:
        if summary.startswith("初步判讀："):
            return summary
        if any(token in summary for token in ("\u8cc7\u6599\u4e0d\u8db3", "\u7121\u6cd5\u78ba\u8a8d")):
            return summary
        return "\u8cc7\u6599\u4e0d\u8db3\uff0c\u7121\u6cd5\u78ba\u8a8d\u3002"
