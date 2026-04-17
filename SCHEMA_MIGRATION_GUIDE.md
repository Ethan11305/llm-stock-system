# Schema Migration 執行指南 (004_embedding_enhancements.sql)

**版本：** P0 Embedding Pipeline  
**最後更新：** 2026-04-17  
**預期工時：** 5-15 分鐘（取決於資料庫大小）

---

## 📋 執行清單

### Step 1：準備工作（2 分鐘）

- [ ] 備份資料庫
  ```bash
  # 選項 A：完整備份
  pg_dump -U postgres -d llm_stock > backup_$(date +%Y%m%d_%H%M%S).sql
  
  # 選項 B：快速備份（僅 schema）
  pg_dump -U postgres -d llm_stock --schema-only > schema_backup.sql
  ```

- [ ] 確認 PostgreSQL 連接
  ```bash
  psql -U postgres -d llm_stock -c "SELECT version();"
  ```

- [ ] 確認 pgvector 已安裝
  ```bash
  psql -U postgres -d llm_stock -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT * FROM pg_extension WHERE extname = 'vector';"
  ```

### Step 2：執行 Schema Migration（1-2 分鐘）

```bash
# 方式 1：直接執行（推薦）
psql -U postgres -d llm_stock < db/sql/004_embedding_enhancements.sql

# 方式 2：帶詳細日誌
psql -U postgres -d llm_stock -a -f db/sql/004_embedding_enhancements.sql > migration_$(date +%Y%m%d_%H%M%S).log 2>&1

# 方式 3：從 Python 應用內執行（可選）
from sqlalchemy import create_engine, text
engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')
with engine.connect() as conn:
    with open('db/sql/004_embedding_enhancements.sql') as f:
        conn.execute(text(f.read()))
    conn.commit()
```

### Step 3：驗證 Migration 成功（1-2 分鐘）

#### 3.1 檢查表結構

```bash
psql -U postgres -d llm_stock -c "\d document_embeddings"
```

**預期輸出：**
```
                           Table "public.document_embeddings"
     Column     |       Type       | Collation | Nullable | Default 
----------------+------------------+-----------+----------+---------
 id             | uuid             |           | not null | 
 document_id    | character varying|           | not null | 
 chunk_index    | integer          |           | not null | 
 chunk_text     | text             |           |          | 
 embedding      | vector(1536)     |           |          | 
 ticker         | character varying|           |          | 
 published_at   | timestamp        |           |          | 
 
Indexes:
    "document_embeddings_pkey" PRIMARY KEY, btree (id)
    "uq_document_embeddings_doc_chunk" UNIQUE CONSTRAINT, btree (document_id, chunk_index)
    "idx_doc_embeddings_published_at" btree (published_at)
    "idx_doc_embeddings_ticker" btree (ticker)
    "idx_document_embeddings_hnsw" USING hnsw (embedding vector_cosine_ops)
```

#### 3.2 檢查 UNIQUE constraint

```bash
psql -U postgres -d llm_stock -c \
  "SELECT constraint_name, constraint_type FROM information_schema.table_constraints 
   WHERE table_name = 'document_embeddings' AND constraint_type = 'UNIQUE';"
```

**預期輸出：**
```
           constraint_name           | constraint_type 
------------------------------------+-----------------
 uq_document_embeddings_doc_chunk   | UNIQUE
```

#### 3.3 檢查索引

```bash
psql -U postgres -d llm_stock -c "\di" | grep document_embeddings
```

**預期輸出：**
```
public | idx_doc_embeddings_published_at | index | postgres | document_embeddings
public | idx_doc_embeddings_ticker       | index | postgres | document_embeddings
public | idx_document_embeddings_hnsw    | index | postgres | document_embeddings
```

#### 3.4 檢查 HNSW 索引類型

```bash
psql -U postgres -d llm_stock -c \
  "SELECT indexname, indexdef FROM pg_indexes 
   WHERE indexname = 'idx_document_embeddings_hnsw';"
```

**預期輸出：**
```
           indexname           |                                           indexdef                                            
------------------------------+------------------------------------------------------------------------------------------------
 idx_document_embeddings_hnsw | CREATE INDEX idx_document_embeddings_hnsw ON public.document_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)
```

### Step 4：運行完整驗證腳本（1 分鐘）

```bash
psql -U postgres -d llm_stock << 'EOF'
-- 完整驗證腳本

\echo '========== P0 Schema Migration 驗證 =========='

-- 1. 檢查 UNIQUE constraint
\echo '1. UNIQUE constraint 檢查'
SELECT 
    constraint_name,
    constraint_type
FROM information_schema.table_constraints
WHERE table_name = 'document_embeddings' 
  AND constraint_type = 'UNIQUE';

-- 2. 檢查 Metadata 欄位
\echo '2. Metadata 欄位檢查'
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'document_embeddings'
  AND column_name IN ('ticker', 'published_at')
ORDER BY column_name;

-- 3. 檢查索引數量
\echo '3. 索引檢查'
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'document_embeddings'
ORDER BY indexname;

-- 4. 檢查向量欄位維度
\echo '4. 向量欄位維度檢查'
SELECT typlen FROM pg_attribute
WHERE attrelid = 'document_embeddings'::regclass
  AND attname = 'embedding';

-- 5. 查詢統計
\echo '5. 目前資料統計'
SELECT 
    'document_embeddings' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT document_id) as unique_documents,
    COUNT(ticker) as rows_with_ticker,
    COUNT(published_at) as rows_with_published_at
FROM document_embeddings;

\echo '========== 驗證完成 =========='
EOF
```

---

## ⚠️ 常見問題 & 排查

| 問題 | 症狀 | 解決方案 |
|---|---|---|
| pgvector 未安裝 | `ERROR: type "vector" does not exist` | 執行 `001_extensions.sql` 或 `CREATE EXTENSION vector;` |
| 權限不足 | `ERROR: permission denied for schema public` | 使用擁有 superuser 權限的帳戶，或聯繫 DBA |
| 約束已存在 | `ERROR: constraint "uq_document_embeddings_doc_chunk" already exists` | 正常 - SQL 腳本已設計成冪等（可重複執行） |
| 索引建立失敗 | `ERROR: could not create index "idx_document_embeddings_hnsw"` | 確認 embedding 欄位是 `vector(1536)` 類型 |
| 執行速度慢（> 1 分鐘） | 無錯誤，但遲遲未完成 | 正常 - 如果有大量資料，HNSW 索引建立需時較久 |

---

## 📊 執行前後對比

| 項目 | 執行前 | 執行後 | 用途 |
|---|---|---|---|
| UNIQUE constraint | ❌ 無 | ✅ (document_id, chunk_index) | 支援 upsert 邏輯 |
| metadata 欄位 | ❌ 無 | ✅ ticker, published_at | 快速篩選 + 避免 JOIN |
| 向量索引 | IVFFlat | HNSW | 更穩定的查詢延遲 |

---

## 🔄 回滾方案（如需要）

如果需要回滾此 migration：

```sql
-- 警告：此操作會遺失新增的資料！
-- 僅在測試環境或有備份的情況下使用

-- 刪除新索引
DROP INDEX IF EXISTS idx_document_embeddings_hnsw;
DROP INDEX IF EXISTS idx_doc_embeddings_ticker;
DROP INDEX IF EXISTS idx_doc_embeddings_published_at;

-- 刪除新欄位
ALTER TABLE document_embeddings
    DROP COLUMN IF EXISTS ticker,
    DROP COLUMN IF EXISTS published_at;

-- 刪除 UNIQUE constraint
ALTER TABLE document_embeddings
    DROP CONSTRAINT IF EXISTS uq_document_embeddings_doc_chunk;
```

---

## ✅ 執行後的下一步

Migration 成功後，應立即進行：

### 1. 回填 Embeddings（如有歷史文件）

```bash
# 先檢查需要回填多少文件
psql -U postgres -d llm_stock -c \
  "SELECT COUNT(*) FROM documents d 
   LEFT JOIN document_embeddings de ON de.document_id = d.id 
   WHERE de.id IS NULL AND d.is_valid = TRUE;"

# 試運行（不實際寫入）
python scripts/backfill_embeddings.py --dry-run

# 正式回填（逐批處理）
python scripts/backfill_embeddings.py --batch-size 50
```

### 2. 驗證語義檢索

```python
# Python 測試腳本
from llm_stock_system.adapters.vector_retrieval import VectorRetrievalAdapter
from llm_stock_system.services.embedding_service import EmbeddingService

# 初始化
embedding_svc = EmbeddingService(openai_api_key=YOUR_KEY, db_engine=engine)
vector_adapter = VectorRetrievalAdapter(embedding_svc, engine)

# 測試查詢
results = vector_adapter.search(
    query_text="台積電供應鏈風險",
    top_k=5,
    ticker="2330"
)

for r in results:
    print(f"Similarity: {r.similarity_score:.3f}")
    print(f"Text: {r.chunk_text[:100]}...")
```

---

**Migration 祝您順利！🚀**
