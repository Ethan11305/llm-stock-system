from collections import Counter
from datetime import datetime, timezone

from llm_stock_system.core.enums import ConsistencyStatus, FreshnessStatus, SourceTier, SufficiencyStatus
from llm_stock_system.core.models import Document, Evidence, GovernanceReport, StructuredQuery


class DataGovernanceLayer:
    def curate(self, query: StructuredQuery, documents: list[Document]) -> GovernanceReport:
        del query
        report = GovernanceReport()
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()

        valid_documents: list[Document] = []

        for document in documents:
            normalized_title = self._normalize_title(document.title)
            if not self._is_document_valid(document):
                report.dropped_document_ids.append(document.id)
                continue
            if document.url in seen_urls or normalized_title in seen_titles:
                report.dropped_document_ids.append(document.id)
                continue
            seen_urls.add(document.url)
            seen_titles.add(normalized_title)
            valid_documents.append(document)

        corroboration_counter = Counter(self._normalize_title(item.title) for item in valid_documents)

        report.evidence = [
            Evidence(
                document_id=document.id,
                title=document.title,
                excerpt=self._build_excerpt(document.content),
                full_content=self._build_full_content(document.content),
                source_name=document.source_name,
                source_tier=document.source_tier,
                source_type=document.source_type,
                url=document.url,
                published_at=document.published_at,
                support_score=self._support_score(document),
                corroboration_count=corroboration_counter[self._normalize_title(document.title)],
            )
            for document in valid_documents
        ]

        report.sufficiency = (
            SufficiencyStatus.SUFFICIENT if len(report.evidence) >= 2 else SufficiencyStatus.INSUFFICIENT
        )
        report.consistency = self._consistency_status(report.evidence)
        report.freshness = self._freshness_status(valid_documents)
        report.high_trust_ratio = self._high_trust_ratio(valid_documents)
        return report

    def _is_document_valid(self, document: Document) -> bool:
        return all(
            (
                document.is_valid,
                document.url,
                document.title,
                document.content,
                document.published_at,
            )
        )

    def _normalize_title(self, title: str) -> str:
        return "".join(char.lower() for char in title if char.isalnum())

    def _build_excerpt(self, content: str, limit: int = 220) -> str:
        trimmed = " ".join(content.split())
        return trimmed if len(trimmed) <= limit else f"{trimmed[:limit].rstrip()}..."

    def _build_full_content(self, content: str, limit: int = 1200) -> str:
        trimmed = " ".join(content.split())
        return trimmed if len(trimmed) <= limit else f"{trimmed[:limit].rstrip()}..."

    def _support_score(self, document: Document) -> float:
        tier_weight = {
            SourceTier.HIGH: 1.0,
            SourceTier.MEDIUM: 0.75,
            SourceTier.LOW: 0.4,
        }[document.source_tier]
        return round(tier_weight, 2)

    def _consistency_status(self, evidence: list[Evidence]) -> ConsistencyStatus:
        unique_sources = len({e.source_name for e in evidence})
        if unique_sources >= 3:
            return ConsistencyStatus.CONSISTENT
        if unique_sources == 2:
            return ConsistencyStatus.MOSTLY_CONSISTENT
        return ConsistencyStatus.CONFLICTING

    def _freshness_status(self, documents: list[Document]) -> FreshnessStatus:
        if not documents:
            return FreshnessStatus.OUTDATED

        now = datetime.now(timezone.utc)
        # published_at is always UTC-aware (normalised by Document field_validator)
        latest = max(document.published_at for document in documents)
        age_days = (now - latest).days

        if age_days <= 7:
            return FreshnessStatus.RECENT
        if age_days <= 30:
            return FreshnessStatus.STALE
        return FreshnessStatus.OUTDATED

    def _high_trust_ratio(self, documents: list[Document]) -> float:
        if not documents:
            return 0.0
        high_trust = sum(1 for document in documents if document.source_tier == SourceTier.HIGH)
        return round(high_trust / len(documents), 2)
