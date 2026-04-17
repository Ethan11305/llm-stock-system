"""vector_retrieval.py

基於 pgvector 的語意檢索 adapter。

核心查詢流程：
1. 將使用者查詢文字轉成 embedding（透過 EmbeddingService）
2. 用 cosine similarity 在 document_embeddings 中找最相似的 chunk
3. JOIN documents 表取得完整 metadata
4. 支援 metadata 前置過濾（ticker、時間範圍）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchResult:
    """語意檢索結果：一個 chunk 及其相似度分數。"""

    document_id: str
    chunk_text: str
    chunk_index: int
    similarity_score: float  # cosine similarity，越高越相似（0~1）
    # 從 documents 表 JOIN 或 document_embeddings 的 metadata 欄位取得
    ticker: str | None = None
    title: str | None = None
    source_tier: str | None = None
    published_at: str | None = None


class VectorRetrievalAdapter:
    """基於 pgvector 的語意檢索。

    依賴：
    - EmbeddingService（生成查詢文字的 embedding）
    - SQLAlchemy engine（連接 Postgres + pgvector）
    """

    def __init__(self, embedding_service, db_engine) -> None:
        """
        Args:
            embedding_service: EmbeddingService 實例（用於生成查詢 embedding）
            db_engine: SQLAlchemy engine
        """
        self._embedding_service = embedding_service
        self._engine = db_engine

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        ticker: str | None = None,
        days_back: int | None = None,
        similarity_threshold: float = 0.65,
    ) -> list[SemanticSearchResult]:
        """語意檢索主方法。

        Args:
            query_text: 使用者查詢文字，會被轉為 embedding
            top_k: 最多回傳幾筆結果
            ticker: 過濾股票代碼（None = 不過濾）
            days_back: 只看最近幾天的文件（None = 不過濾）
            similarity_threshold: cosine similarity 最低門檻（預設 0.65）

        Returns:
            依相似度降序排列的 SemanticSearchResult 列表
        """
        from sqlalchemy import text

        # Step 1: 查詢文字 → embedding
        try:
            vectors = self._embedding_service.embed_texts([query_text])
            query_vector = vectors[0]
        except Exception as e:
            logger.warning("VectorRetrievalAdapter: 生成查詢 embedding 失敗：%s", e)
            return []

        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

        # Step 2: 建構 SQL
        where_clauses: list[str] = []
        params: dict = {
            "query_vec": vector_str,
            "top_k": top_k,
            "threshold": 1.0 - similarity_threshold,  # pgvector <=> 是 distance，越小越近
        }

        if ticker:
            # 優先使用 document_embeddings 上的 ticker 欄位（避免 JOIN 效能）
            # 如果 ticker 欄位為 NULL（舊資料），fallback 到 JOIN documents
            where_clauses.append("de.ticker = :ticker")
            params["ticker"] = ticker

        if days_back:
            where_clauses.append(
                "de.published_at >= CURRENT_TIMESTAMP - (:days_back * INTERVAL '1 day')"
            )
            params["days_back"] = days_back

        # cosine distance 門檻過濾（distance = 1 - similarity）
        # SQLAlchemy text() parameter binding does not work with the `:param::type`
        # form reliably under psycopg. Cast the bound vector explicitly instead.
        where_clauses.append("(de.embedding <=> CAST(:query_vec AS vector)) <= :threshold")

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        sql = text(
            f"""
            SELECT
                de.document_id::text,
                de.chunk_text,
                de.chunk_index,
                1 - (de.embedding <=> CAST(:query_vec AS vector))  AS similarity,
                COALESCE(de.ticker, d.ticker)               AS ticker,
                d.title,
                d.source_tier,
                COALESCE(de.published_at, d.published_at)::text AS published_at
            FROM document_embeddings de
            LEFT JOIN documents d ON d.id = de.document_id
            {where_sql}
            ORDER BY de.embedding <=> CAST(:query_vec AS vector) ASC
            LIMIT :top_k
            """
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            logger.warning("VectorRetrievalAdapter: DB 查詢失敗：%s", e)
            return []

        return [
            SemanticSearchResult(
                document_id=str(row[0]),
                chunk_text=row[1],
                chunk_index=int(row[2]),
                similarity_score=float(row[3]),
                ticker