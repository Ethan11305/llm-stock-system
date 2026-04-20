from datetime import datetime, timedelta, timezone
from uuid import uuid4

from llm_stock_system.core.enums import Intent, SourceTier, Topic
from llm_stock_system.core.fundamental_valuation import is_fundamental_valuation_question
from llm_stock_system.core.interfaces import DocumentRepository, QueryLogStore
from llm_stock_system.core.models import (
    Document,
    GovernanceReport,
    QueryLogDetail,
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
        if self._should_include_comparison(query):
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
        if self._prefers_earnings_documents(query) and Topic.EARNINGS in document.topics:
            score += 2
        if self._prefers_announcement_documents(query) and Topic.ANNOUNCEMENT in document.topics:
            score += 2
        if self._is_theme_comparison_query(query) and document.ticker in {query.ticker, query.comparison_ticker}:
            score += 2
        if self._prefers_valuation_documents(query):
            score += 1
        return score

    def _source_tier_weight(self, tier: SourceTier) -> int:
        return {
            SourceTier.HIGH: 3,
            SourceTier.MEDIUM: 2,
            SourceTier.LOW: 1,
        }[tier]

    def _topic_tags(self, query: StructuredQuery) -> set[str]:
        return set(query.topic_tags or [])

    def _is_theme_comparison_query(self, query: StructuredQuery) -> bool:
        tags = self._topic_tags(query)
        return (
            query.intent == Intent.NEWS_DIGEST
            and bool(query.comparison_ticker)
            and bool(tags & {"題材", "產業", "AI", "電動車", "半導體設備"})
        )

    def _is_gross_margin_comparison_query(self, query: StructuredQuery) -> bool:
        tags = self._topic_tags(query)
        return (
            query.intent == Intent.FINANCIAL_HEALTH
            and bool(query.comparison_ticker)
            and "毛利率" in tags
        )

    def _should_include_comparison(self, query: StructuredQuery) -> bool:
        return self._is_theme_comparison_query(query) or self._is_gross_margin_comparison_query(query)

    def _prefers_earnings_documents(self, query: StructuredQuery) -> bool:
        tags = self._topic_tags(query)
        return query.topic == Topic.EARNINGS or "財報" in tags

    def _prefers_announcement_documents(self, query: StructuredQuery) -> bool:
        tags = self._topic_tags(query)
        return query.topic == Topic.ANNOUNCEMENT or "公告" in tags

    def _prefers_valuation_documents(self, query: StructuredQuery) -> bool:
        tags = self._topic_tags(query)
        return (
            is_fundamental_valuation_question(query)
            or "股價區間" in tags
            or ("股價" in tags and "展望" in tags)
        )


class InMemoryQueryLogStore(QueryLogStore):
    def __init__(self) -> None:
        self._source_index: dict[str, SourceResponse] = {}
        self._detail_index: dict[str, QueryLogDetail] = {}

    def save(
        self,
        query: StructuredQuery,
        response: QueryResponse,
        governance_report: GovernanceReport,
        validation_result: ValidationResult,
    ) -> str:
        query_id = response.query_id or str(uuid4())
        source_count = len(governance_report.evidence)

        self._source_index[query_id] = SourceResponse(
            query_id=query_id,
            ticker=query.ticker,
            topic=query.topic,
            source_count=source_count,
            sources=response.sources,
        )

        self._detail_index[query_id] = QueryLogDetail(
            query_id=query_id,
            query_profile=query.query_profile,
            classifier_source=query.classifier_source,
            validation_status=validation_result.validation_status,
            warnings=list(validation_result.warnings),
            source_count=source_count,
            schema_version=1,
            structured_query=query.model_dump(mode="json"),
            response=response,
        )
        return query_id

    def get_sources(self, query_id: str) -> SourceResponse | None:
        return self._source_index.get(query_id)

    def get_query_log(self, query_id: str) -> QueryLogDetail | None:
        return self._detail_index.get(query_id)


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
