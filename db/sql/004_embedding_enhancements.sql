-- 004_embedding_enhancements.sql
--
-- P0 Embedding Pipeline 所需的 Schema 強化：
--   1. 加 UNIQUE constraint，支援 upsert ON CONFLICT (document_id, chunk_index)
--   2. 加 ticker / published_at metadata 欄位，避免每次 JOIN documents
--   3. 將索引從 IVFFlat 升級為 HNSW（查詢延遲更穩定，不需預先建立索引）
--
-- 執行前提：pgvector extension 已安裝（001_extensions.sql），
--            document_embeddings 表已建立（002_schema.sql）

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Unique constraint（支援 ON CONFLICT upsert）
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_document_embeddings_doc_chunk'
          AND conrelid = 'document_embeddings'::regclass
    ) THEN
        ALTER TABLE document_embeddings
            ADD CONSTRAINT uq_document_embeddings_doc_chunk
            UNIQUE (document_id, chunk_index);
    END IF;
END
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Metadata 欄位（避免每次 JOIN documents 表）
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE document_embeddings
    ADD COLUMN IF NOT EXISTS ticker       VARCHAR(16),
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMP;

-- Ticker index：支援 WHERE ticker = :ticker 前置過濾
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_ticker
    ON document_embeddings (ticker);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. HNSW 索引（取代 IVFFlat）
--    IVFFlat 缺點：需要足夠資料量（> lists × 10）才有效率
--    HNSW 優點：插入即可查詢，延遲更穩定，適合資料量從小逐漸增長的場景
-- ─────────────────────────────────────────────────────────────────────────────
DROP INDEX IF EXISTS idx_document_embeddings_ivfflat;

CREATE INDEX IF NOT EXISTS idx_document_embeddings_hnsw
    ON document_embeddings
    USING hnsw (embedding