"""PostgreSQL-backed QueryLogStore.

此 adapter 是 Digest 產品線可信閉環的核心落地點。與 ``InMemoryQueryLogStore``
功能對等，差別在於資料寫進 PostgreSQL，重啟後仍可回查 sources / warnings /
classifier_source / structured_query / response snapshot。

相依 schema:
    * db/sql/002_schema.sql               — query_logs / query_sources 基礎表
    * db/sql/003_query_log_observability.sql — intent / controlled_tags 等欄位
    * db/sql/005_query_log_digest.sql     — query_profile / response_json 等欄位
                                            + query_log_warnings 表

兩個 read method（``get_sources`` / ``get_query_log``）皆完整實作，不留 stub。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from llm_stock_system.core.enums import (
    ConsistencyStatus,
    FreshnessStatus,
    QueryProfile,
    SourceTier,
    SufficiencyStatus,
    Topic,
)
from llm_stock_system.core.interfaces import QueryLogStore
from llm_stock_system.core.models import (
    DataStatus,
    GovernanceReport,
    QueryLogDetail,
    QueryResponse,
    SourceCitation,
    SourceResponse,
    StructuredQuery,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class PostgresQueryLogStore(QueryLogStore):
    """PostgreSQL 實作，對齊 005 migration 的 schema。"""

    def __init__(
        self,
        database_url: str,
        *,
        engine: Engine | None = None,
    ) -> None:
        # 允許外部注入 engine（方便共用連線池 / 測試 mock）
        self._engine: Engine = engine if engine is not None else create_engine(database_url)

    # ──────────────────────────────────────
    # save
    # ──────────────────────────────────────

    def save(
        self,
        query: StructuredQuery,
        response: QueryResponse,
        governance_report: GovernanceReport,
        validation_result: ValidationResult,
    ) -> str:
        query_id = response.query_id or str(uuid4())
        source_count = len(governance_report.evidence)

        structured_query_json = query.model_dump(mode="json")
        response_json = response.model_dump(mode="json", by_alias=True)

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO query_logs (
                        id, user_query, ticker, company_name, topic,
                        time_range, time_range_label, time_range_days,
                        intent, query_profile, classifier_source,
                        controlled_tags, tag_source, facet_miss_list,
                        retrieved_doc_count,
                        confidence_score, confidence_light,
                        validation_status,
                        sufficiency_status, consistency_status, freshness_status,
                        summary, response_text,
                        schema_version, response_json, structured_query_json
                    ) VALUES (
                        :id, :user_query, :ticker, :company_name, :topic,
                        :time_range, :time_range_label, :time_range_days,
                        :intent, :query_profile, :classifier_source,
                        :controlled_tags, :tag_source, :facet_miss_list,
                        :retrieved_doc_count,
                        :confidence_score, :confidence_light,
                        :validation_status,
                        :sufficiency_status, :consistency_status, :freshness_status,
                        :summary, :response_text,
                        :schema_version, CAST(:response_json AS JSONB),
                        CAST(:structured_query_json AS JSONB)
                    )
                    """
                ),
                {
                    "id": query_id,
                    "user_query": query.user_query,
                    "ticker": query.ticker,
                    "company_name": query.company_name,
                    "topic": query.topic.value,
                    "time_range": query.time_range_label,       # legacy column
                    "time_range_label": query.time_range_label,
                    "time_range_days": query.time_range_days,
                    "intent": query.intent.value,
                    "query_profile": query.query_profile.value,
                    "classifier_source": query.classifier_source,
                    "controlled_tags": [tag.value for tag in query.controlled_tags],
                    "tag_source": query.tag_source,
                    "facet_miss_list": list(validation_result.facet_miss_list),
                    "retrieved_doc_count": source_count,
                    "confidence_score": response.confidence_score,
                    "confidence_light": response.confidence_light.value,
                    "validation_status": validation_result.validation_status,
                    "sufficiency_status": response.data_status.sufficiency.value,
                    "consistency_status": response.data_status.consistency.value,
                    "freshness_status": response.data_status.freshness.value,
                    "summary": response.summary,
                    "response_text": response.summary,           # legacy column
                    "schema_version": 1,
                    "response_json": json.dumps(response_json),
                    "structured_query_json": json.dumps(structured_query_json),
                },
            )

            # query_sources 的 document_id 必需來自 Evidence（SourceCitation 沒存）
            for evidence in governance_report.evidence:
                conn.execute(
                    text(
                        """
                        INSERT INTO query_sources (
                            id, query_log_id, document_id,
                            title, source_name, source_tier, url,
                            published_at, excerpt, support_score, corroboration_count
                        ) VALUES (
                            :id, :query_log_id, :document_id,
                            :title, :source_name, :source_tier, :url,
                            :published_at, :excerpt, :support_score, :corroboration_count
                        )
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "query_log_id": query_id,
                        "document_id": evidence.document_id,
                        "title": evidence.title,
                        "source_name": evidence.source_name,
                        "source_tier": evidence.source_tier.value,
                        "url": evidence.url,
                        "published_at": evidence.published_at,
                        "excerpt": evidence.excerpt,
                        "support_score": evidence.support_score,
                        "corroboration_count": evidence.corroboration_count,
                    },
                )

            for warning in validation_result.warnings:
                conn.execute(
                    text(
                        """
                        INSERT INTO query_log_warnings (
                            id, query_log_id, warning_text
                        ) VALUES (
                            :id, :query_log_id, :warning_text
                        )
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "query_log_id": query_id,
                        "warning_text": warning,
                    },
                )

        return query_id

    # ──────────────────────────────────────
    # get_sources
    # ──────────────────────────────────────

    def get_sources(self, query_id: str) -> SourceResponse | None:
        with self._engine.connect() as conn:
            log_row = conn.execute(
                text(
                    """
                    SELECT ticker, topic
                    FROM query_logs
                    WHERE id = :id
                    """
                ),
                {"id": query_id},
            ).mappings().first()

            if log_row is None:
                return None

            source_rows = conn.execute(
                text(
                    """
                    SELECT title, source_name, source_tier, url,
                           published_at, excerpt, support_score, corroboration_count
                    FROM query_sources
                    WHERE query_log_id = :id
                    ORDER BY published_at DESC
                    """
                ),
                {"id": query_id},
            ).mappings().all()

        try:
            topic = Topic(log_row["topic"])
        except ValueError:
            logger.warning(
                "PostgresQueryLogStore.get_sources: 未知 topic %r，fallback 到 COMPOSITE",
                log_row["topic"],
            )
            topic = Topic.COMPOSITE

        sources = [self._row_to_source_citation(row) for row in source_rows]

        return SourceResponse(
            query_id=query_id,
            ticker=log_row["ticker"],
            topic=topic,
            source_count=len(sources),
            sources=sources,
        )

    # ──────────────────────────────────────
    # get_query_log
    # ──────────────────────────────────────

    def get_query_log(self, query_id: str) -> QueryLogDetail | None:
        with self._engine.connect() as conn:
            log_row = conn.execute(
                text(
                    """
                    SELECT id, query_profile, classifier_source, validation_status,
                           schema_version, response_json, structured_query_json,
                           ticker, topic,
                           sufficiency_status, consistency_status, freshness_status,
                           confidence_light, confidence_score,
                           summary, response_text
                    FROM query_logs
                    WHERE id = :id
                    """
                ),
                {"id": query_id},
            ).mappings().first()

            if log_row is None:
                return None

            warning_rows = conn.execute(
                text(
                    """
                    SELECT warning_text
                    FROM query_log_warnings
                    WHERE query_log_id = :id
                    ORDER BY created_at ASC
                    """
                ),
                {"id": query_id},
            ).mappings().all()

            source_rows = conn.execute(
                text(
                    """
                    SELECT title, source_name, source_tier, url,
                           published_at, excerpt, support_score, corroboration_count
                    FROM query_sources
                    WHERE query_log_id = :id
                    ORDER BY published_at DESC
                    """
                ),
                {"id": query_id},
            ).mappings().all()

        response = self._reconstruct_response(log_row, source_rows, query_id)
        if response is None:
            logger.warning(
                "PostgresQueryLogStore.get_query_log: 無法重建 response（query_id=%s）",
                query_id,
            )
            return None

        structured_query_json = self._ensure_dict(log_row["structured_query_json"])
        query_profile_raw = log_row["query_profile"] or QueryProfile.LEGACY.value
        try:
            query_profile = QueryProfile(query_profile_raw)
        except ValueError:
            query_profile = QueryProfile.LEGACY

        return QueryLogDetail(
            query_id=query_id,
            query_profile=query_profile,
            classifier_source=log_row["classifier_source"] or "rule",
            validation_status=log_row["validation_status"],
            warnings=[row["warning_text"] for row in warning_rows],
            source_count=len(source_rows),
            schema_version=int(log_row["schema_version"] or 1),
            structured_query=structured_query_json,
            response=response,
        )

    # ──────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────

    @staticmethod
    def _row_to_source_citation(row: Any) -> SourceCitation:
        try:
            tier = SourceTier(row["source_tier"])
        except ValueError:
            tier = SourceTier.MEDIUM
        return SourceCitation(
            title=row["title"],
            source_name=row["source_name"],
            source_tier=tier,
            url=row["url"],
            published_at=row["published_at"],
            excerpt=row["excerpt"] or "",
            support_score=float(row["support_score"]),
            corroboration_count=int(row["corroboration_count"]),
        )

    @staticmethod
    def _ensure_dict(value: Any) -> dict:
        """psycopg 會把 JSONB 以 dict 形式回傳，但測試或舊資料可能是字串。"""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}

    def _reconstruct_response(
        self,
        log_row: Any,
        source_rows: list[Any],
        query_id: str,
    ) -> QueryResponse | None:
        """優先使用 response_json；缺少時 fallback 從欄位重建（維持舊資料可讀）。"""
        payload = self._ensure_dict(log_row["response_json"])
        if payload:
            payload["query_id"] = payload.get("query_id") or query_id
            try:
                return QueryResponse.model_validate(payload)
            except Exception as exc:
                logger.warning(
                    "PostgresQueryLogStore: response_json 反序列化失敗 (%s)，fallback 到欄位重建",
                    exc,
                )

        # Fallback：用 query_logs 單欄位 + query_sources 重建最小 QueryResponse
        try:
            data_status = DataStatus(
                sufficiency=SufficiencyStatus(log_row["sufficiency_status"] or "insufficient"),
                consistency=ConsistencyStatus(log_row["consistency_status"] or "mostly_consistent"),
                freshness=FreshnessStatus(log_row["freshness_status"] or "outdated"),
            )
        except ValueError:
            return None

        from llm_stock_system.core.enums import ConfidenceLight
        try:
            confidence_light = ConfidenceLight(log_row["confidence_light"] or "red")
        except ValueError:
            confidence_light = ConfidenceLight.RED

        return QueryResponse(
            query_id=query_id,
            summary=log_row["summary"] or log_row["response_text"] or "",
            highlights=[],
            facts=[],
            impacts=[],
            risks=[],
            data_status=data_status,
            confidence_light=confidence_light,
            confidence_score=float(log_row["confidence_score"] or 0.0),
            sources=[self._row_to_source_citation(row) for row in source_rows],
            disclaimer="本系統僅整理公開資訊，不構成投資建議。",
        )


__all__ = ["PostgresQueryLogStore"]
