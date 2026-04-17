"""embedding_service.py

負責生成 embedding 並寫入 document_embeddings 表。

設計要點：
- 批次處理（OpenAI 一次最多 2048 個輸入）
- 重複偵測（避免同一 chunk 重複 embed，用 document_id + chunk_index 唯一鍵）
- 錯誤重試（API rate limit 指數退避）
- 完全可選：embedding 失敗不阻擋主流程
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from llm_stock_system.core.models import Document
from llm_stock_system.services.document_chunker import DocumentChunk, DocumentChunker

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """單一 chunk 的 embedding 結果。"""

    chunk: DocumentChunk
    vector: list[float]  # 1536 維（text-embedding-3-small）


class EmbeddingService:
    """負責生成 embedding 並寫入 document_embeddings 表。

    依賴：
    - OpenAI Embedding API（text-embedding-3-small，維度 1536，與 schema VECTOR(1536) 一致）
    - SQLAlchemy engine（連接 Postgres + pgvector）
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        openai_api_key: str,
        model: str = DEFAULT_MODEL,
        batch_size: int = 100,
        db_engine=None,  # sqlalchemy Engine
    ) -> None:
        self.api_key = openai_api_key
        self.model = model
        self.batch_size = batch_size
        self.engine = db_engine

    # ──────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────

    def embed_and_store(
        self,
        documents: list[Document],
        chunker: DocumentChunker | None = None,
    ) -> int:
        """完整流程：Document → Chunk → Embed → Store。
        回傳寫入（新增或更新）的 chunk 筆數。
        """
        if not documents:
            return 0

        _chunker = chunker or DocumentChunker()

        # Step 1: Chunking
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(_chunker.chunk_document(doc))

        if not all_chunks:
            return 0

        # Step 2: 過濾已存在的 chunk（避免重複 embed）
        new_chunks = self._filter_existing(all_chunks)
        if not new_chunks:
            logger.debug("embed_and_store: 所有 chunk 均已存在，跳過。")
            return 0

        # Step 3: 批次生成 embedding
        results: list[EmbeddingResult] = []
        for i in range(0, len(new_chunks), self.batch_size):
            batch = new_chunks[i : i + self.batch_size]
            try:
                vectors = self._call_openai_embedding([c.chunk_text for c in batch])
            except Exception as e:
                logger.warning("embed_and_store: batch %d 失敗，跳過。原因：%s", i, e)
                continue
            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddingResult(chunk=chunk, vector=vector))

        # Step 4: 批次寫入 DB
        written = self._bulk_upsert(results)
        logger.info("embed_and_store: 寫入 %d 個 chunk embedding（來自 %d 篇文件）", written, len(documents))
        return written

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """直接 embed 文字列表（供 VectorRetrievalAdapter 使用查詢 embedding）。"""
        return self._call_openai_embedding(texts)

    # ──────────────────────────────────────
    # OpenAI API 呼叫
    # ──────────────────────────────────────

    def _call_openai_embedding(self, texts: list[str]) -> list[list[float]]:
        """呼叫 OpenAI Embedding API，帶指數退避重試（最多 3 次）。"""
        import httpx

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = httpx.post(url, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                # OpenAI 回傳的 embedding 順序與 input 一致（index 欄位對應）
                return [
                    item["embedding"]
                    for item in sorted(data["data"], key=lambda x: x["index"])
                ]
            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code == 429:
                    wait = 2**attempt
                    logger.warning("OpenAI 429 rate limit，等待 %ds 後重試（attempt %d）", wait, attempt + 1)
                    time.sleep(wait)
                else:
                    raise
            except httpx.RequestError as e:
                last_exc = e
                wait = 2**attempt
                logger.warning("OpenAI 請求失敗，等待 %ds 後重試（attempt %d）：%s", wait, attempt + 1, e)
                time.sleep(wait)

        raise RuntimeError(f"OpenAI Embedding API 重試 3 次仍失敗：{last_exc}")

    # ──────────────────────────────────────
    # DB 操作
    # ──────────────────────────────────────

    def _filter_existing(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """過濾已在 DB 中存在的 chunk，避免重複寫入。
        以 (document_id, chunk_index) 作為唯一鍵。
        如果 engine 未設定，則全部視為新的。
        """
        if self.engine is None:
            return chunks

        from sqlalchemy import text

        doc_ids = list({c.document_id for c in chunks})
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT document_id, chunk_index "
                        "FROM document_embeddings "
                        "WHERE document_id = ANY(:ids)"
                    ),
                    {"ids": doc_ids},
                )
                existing = {(str(row[0]), int(row[1])) for row in result}
        except Exception as e:
            logger.warning("_filter_existing 查詢失敗，假設全部為新 chunk：%s", e)
            return chunks

        return [c for c in chunks if (c.document_id, c.chunk_index) not in existing]

    def _bulk_upsert(self, results: list[EmbeddingResult]) -> int:
        """批次 upsert document_embeddings 表。
        ON CONFLICT (document_id, chunk_index) 更新 chunk_text 和 embedding。
        """
        if not results or self.engine is None:
            return len(results)  # engine 未設定時仍回傳筆數（便於測試）

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO document_embeddings
                (id, document_id, chunk_index, chunk_text, embedding, ticker, published_at)
            VALUES
                (:id, :document_id, :chunk_index, :chunk_text, :embedding, :ticker, :published_at)
            ON CONFLICT (document_id, chunk_index) DO UPDATE
                SET chunk_text   = EXCLUDED.chunk_text,
                    embedding    = EXCLUDED.embedding,
                    ticker       = EXCLUDED.ticker,
                    published_at = EXCLUDED.published_at
            """
        )

        rows = []
        for r in results:
            # pgvector 接受 '[0.1, 0.2, ...]' 字串格式
            vector_str = "[" + ",".join(str(v) for v in r.vector) + "]"
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "document_id": r.chunk.document_id,
                    "chunk_index": r.chunk.chunk_index,
                    "chunk_text": r.chunk.chunk_text,
                    "embedding": vector_str,
                    "ticker": r.chunk.metadata.get("ticker"),
                    "published_at": r.chunk.metadata.get("published_at"),
                }
            )

        try:
            with self.engine.begin() as conn:
                conn.execute(sql, rows)
        except Exception as e:
            logger.error("_bulk_upsert 寫入失敗：%s", e)
            raise

        return len(rows)
