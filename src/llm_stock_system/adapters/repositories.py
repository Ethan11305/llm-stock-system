from datetime import datetime, timedelta, timezone
from uuid import uuid4

from llm_stock_system.core.enums import SourceTier, Topic
from llm_stock_system.core.interfaces import DocumentRepository, QueryLogStore
from llm_stock_system.core.models import (
    Document,
    GovernanceReport,
    QueryResponse,
    SourceResponse,
    StructuredQuery,
    ValidationResult,
)


class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self, seed_documents: list[Document] | None = None) -> None:
        self._documents = list(seed_documents or [])

    def upsert_documents(self, documents: list[Document]) -> int:
        by_url = {item.url: item for item in self._documents}
        for document in documents:
            by_url[document.url] = document
        self._documents = list(by_url.values())
        return len(documents)

    def search_documents(self, query: StructuredQuery) -> list[Document]:
        if not query.ticker:
            return []

        now = datetime.now(timezone.utc)
        min_published_at = now - timedelta(days=query.time_range_days)
        target_tickers = {query.ticker}
        if query.comparison_ticker and query.question_type in {"gross_margin_comparison_review", "theme_impact_review"}:
            target_tickers.add(query.comparison_ticker)
        candidates: list[tuple[int, Document]] = []

        for document in self._documents:
            if document.ticker not in target_tickers:
                continue
            if document.published_at < min_published_at:
                continue
            if query.topic != Topic.COMPOSITE and query.topic not in document.topics:
                continue

            score = self._score_document(document, query)
            candidates.append((score, document))

        candidates.sort(
            key=lambda item: (
                item[0],
                self._source_tier_weight(item[1].source_tier),
                item[1].published_at,
            ),
            reverse=True,
        )
        return [document for _, document in candidates]

    def _score_document(self, document: Document, query: StructuredQuery) -> int:
        score = 0
        if document.ticker == query.ticker:
            score += 5
        if query.comparison_ticker and document.ticker == query.comparison_ticker:
            score += 4
        if query.topic in document.topics:
            score += 3
        if query.question_type == "earnings_summary" and Topic.EARNINGS in document.topics:
            score += 2
        if query.question_type == "announcement_summary" and Topic.ANNOUNCEMENT in document.topics:
            score += 2
        if query.question_type == "theme_impact_review" and document.ticker in {query.ticker, query.comparison_ticker}:
            score += 2
        if query.question_type in {"investment_support", "price_outlook", "price_range"}:
            score += 1
        return score

    def _source_tier_weight(self, tier: SourceTier) -> int:
        return {
            SourceTier.HIGH: 3,
            SourceTier.MEDIUM: 2,
            SourceTier.LOW: 1,
        }[tier]


class InMemoryQueryLogStore(QueryLogStore):
    def __init__(self) -> None:
        self._source_index: dict[str, SourceResponse] = {}

    def save(
        self,
        query: StructuredQuery,
        response: QueryResponse,
        governance_report: GovernanceReport,
        validation_result: ValidationResult,
    ) -> str:
        _ = validation_result
        query_id = response.query_id or str(uuid4())
        self._source_index[query_id] = SourceResponse(
            query_id=query_id,
            ticker=query.ticker,
            topic=query.topic,
            source_count=len(governance_report.evidence),
            sources=response.sources,
        )
        return query_id

    def get_sources(self, query_id: str) -> SourceResponse | None:
        return self._source_index.get(query_id)


class HybridDocumentRepository(DocumentRepository):
    def __init__(
        self,
        primary_repository: DocumentRepository,
        fallback_repository: DocumentRepository,
    ) -> None:
        self._primary_repository = primary_repository
        self._fallback_repository = fallback_repository

    def upsert_documents(self, documents: list[Document]) -> int:
        return self._primary_repository.upsert_documents(documents)

    def search_documents(self, query: StructuredQuery) -> list[Document]:
        primary_documents = self._primary_repository.search_documents(query)
        if primary_documents:
            return primary_documents
        return self._fallback_repository.search_documents(query)
