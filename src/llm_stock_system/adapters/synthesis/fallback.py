from llm_stock_system.core.models import GovernanceReport, StructuredQuery

from .helpers import build_impacts_generic, build_risks_generic, build_summary_fallback


class FallbackStrategy:
    def build_summary(self, query: StructuredQuery, report: GovernanceReport) -> str:
        return build_summary_fallback(query, report)

    def build_impacts(self, query: StructuredQuery) -> list[str]:
        return build_impacts_generic()

    def build_risks(self, query: StructuredQuery, report: GovernanceReport) -> list[str]:
        return build_risks_generic(query)
