-- 004_embedding_enhancements.sql
--
-- P0 Embedding Pipeline 所需的 Schema 強化：
--   1. 加 UNIQUE constraint，支援 upsert ON CONFLICT (document_id, chunk_index)
--   2. 加 ticker / published_at metadata 欄位，避免每次 JOIN documents
--   3. 將索引從 IVFFlat 升級為 HNSW（查詢延遲更穩定，不需預先建立索引）
--
-- 執行前提：pgvector extension 已安裝（001_extensions.sql），
--            document_embeddings 表已建立（002_schema.sql）
--
-- 執行方式：
--   psql -U postgres -d llm_stock < db/sql/004_embedding_enhancements.sql
--
-- 檢查執行結果：
--   psql -U postgres -d llm_stock -c "\d document_embeddings"
--   psql -U postgres -d llm_stock -c "\di" | grep document_embeddings

-- ─────────────────────────────────────────────────────────────────────────────
-- 0. 檢查前置條件：pgvector extension 是否已安裝
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_extension
        WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION 'pgvector extension 尚未安裝。請先執行 001_extensions.sql';
    END IF;
END
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Unique constraint（支援 ON CONFLICT upsert）
-- ─────────────────────────────────────────────────────────────────────────────
-- 目的：使 embedding_service.py 中的 ON CONFLICT (document_id, chunk_index)
--      DO UPDATE 語法生效
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
        RAISE NOTICE 'Added UNIQUE constraint uq_document_embeddings_doc_chunk';
    ELSE
        RAISE NOTICE 'UNIQUE constraint uq_document_embeddings_doc_chunk already exists';
    END IF;
END
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Metadata 欄位（避免每次 JOIN documents 表）
-- ─────────────────────────────────────────────────────────────────────────────
-- 目的：在 vector_retrieval.py 的 search() 查詢中支援 WHERE ticker = :ticker
--      避免需要 JOIN documents 表（提升查詢速度）

-- 新增欄位（如已存在則跳過）
ALTER TABLE document_embeddings
    ADD COLUMN IF NOT EXISTS ticker       VARCHAR(16),
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMP;

-- 為 ticker 建立索引（支援 WHERE ticker = :ticker 篩選）
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_ticker
    ON document_embeddings (ticker);

-- 為 published_at 建立索引（支援時間範圍篩選：WHERE published_at > :cutoff）
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_published_at
    ON document_embeddings (published_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. HNSW 索引（取代 IVFFlat）
-- ─────────────────────────────────────────────────────────────────────────────
-- IVFFlat 缺點：
--   - 需要足夠資料量（> lists × 10）才能有效發揮
--   - 新增資料後需要重新建立索引
--   - 小資料量下效率不如全掃描
--
-- HNSW 優點：
--   - 插入即可查詢，無需重建
--   - 延遲更穩定，不受資料量影響
--   - 適合從小到大逐漸增長的場景（本系統的典型情況）
--
-- HNSW 參數說明：
--   m = 16           : 每個節點的連接數（預設 16，範圍 2-100）
--                     - 數值越大 → 更精準但記憶體更多
--                     - 數值越小 → 更快但精度降低
--   ef_construction = 64 : 建立索引時的搜尋廣度（預設 64）

-- 先檢查舊的 IVFFlat 索引，若存在則刪除
DROP INDEX IF EXISTS idx_document_embeddings_ivfflat;

-- 建立新的 HNSW 索引
CREATE INDEX IF NOT EXISTS idx_document_embeddings_hnsw
    ON document_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. 驗證與報告
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
DECLARE
    v_constraint_count INT;
    v_index_count INT;
    v_col_count INT;
BEGIN
    -- 檢查 unique constraint
    SELECT COUNT(*) INTO v_constraint_count
    FROM pg_constraint
    WHERE conname = 'uq_document_embeddings_doc_chunk'
      AND conrelid = 'document_embeddings'::regclass;

    -- 檢查 HNSW 索引
    SELECT COUNT(*) INTO v_index_count
    FROM pg_indexes
    WHERE indexname = 'idx_document_embeddings_hnsw';

    -- 檢查 metadata 欄位
    SELECT COUNT(*) INTO v_col_count
    FROM information_schema.columns
    WHERE table_name = 'document_embeddings'
      AND column_name IN ('ticker', 'published_at');

    RAISE NOTICE '========== Schema Migration 檢查報告 ==========';
    RAISE NOTICE 'UNIQUE constraint (document_id, chunk_index): %',
        CASE WHEN v_constraint_count > 0 THEN '✓ 已建立' ELSE '✗ 未建立' END;
    RAISE NOTICE 'HNSW 索引 (embedding): %',
        CASE WHEN v_index_count > 0 THEN '✓ 已建立' ELSE '✗ 未建立' END;
    RAISE NOTICE 'Metadata 欄位 (ticker, published_at): %',
        CASE WHEN v_col_count = 2 THEN '✓ 已建立' ELSE '✗ 部分缺失' END;
    RAISE NOTICE '================================================';
END
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. 後續操作指南
-- ─────────────────────────────────────────────────────────────────────────────
--
-- ✅ Schema migration 成功後的下一步：
--
-- 1. 回填現有文件的 embedding 向量（如有歷史資料）：
--    python scripts/backfill_embeddings.py --dry-run
--    python scripts/backfill_embeddings.py --batch-size 50
--
-- 2. 驗證 embedding 效果（手工測試）：
--    - 開啟 Python REPL 或編寫測試指令碼
--    - 執行 vector_retrieval_adapter.search("台積電供應鏈")
--    - 比較結果與 metadata-only 檢索的差異
--
-- 3. 監控查詢效能：
--    EXPLAIN ANALYZE
--    SELECT * FROM document_embeddings
--    WHERE ticker = '2330'
--      AND embedding <=> '[0.1, 0.2, ..., 0.9]'::vector
--    ORDER BY embedding <=> '[0.1, 0.2, ..., 0.9]'::vector
--    LIMIT 10;
--
-- ─────────────────────────────────────────────────────────────────────────────
