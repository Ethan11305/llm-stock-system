from pathlib import Path
from typing import Protocol

from llm_stock_system.core.models import (
    AnswerDraft,
    Document,
    GovernanceReport,
    PriceBar,
    QueryResponse,
    StockInfo,
    SourceResponse,
    StructuredQuery,
    ValidationResult,
)


class DocumentRepository(Protocol):
    def upsert_documents(self, documents: list[Document]) -> int:
        ...

    def search_documents(self, query: StructuredQuery) -> list[Document]:
        ...


class QueryLogStore(Protocol):
    def save(
        self,
        query: StructuredQuery,
        response: QueryResponse,
        governance_report: GovernanceReport,
        validation_result: ValidationResult,
    ) -> str:
        ...

    def get_sources(self, query_id: str) -> SourceResponse | None:
        ...


class QueryHydrator(Protocol):
    def hydrate(self, query: StructuredQuery) -> None:
        ...


class StockResolver(Protocol):
    def resolve(self, query_text: str) -> tuple[str | None, str | None]:
        ...


class LLMClient(Protocol):
    def synthesize(
        self,
        query: StructuredQuery,
        governance_report: GovernanceReport,
        system_prompt: str,
    ) -> AnswerDraft:
        ...


class PromptLoader(Protocol):
    def load(self, path: Path) -> str:
        ...


class MarketDataGateway(Protocol):
    def sync_stock_info(self, force: bool = False) -> int:
        ...

    def resolve_company(self, query_text: str) -> tuple[str | None, str | None]:
        ...

    def sync_price_history(self, ticker: str, start_date, end_date) -> int:
        ...

    def get_price_bars(self, ticker: str, start_date, end_date) -> list[PriceBar]:
        ...
