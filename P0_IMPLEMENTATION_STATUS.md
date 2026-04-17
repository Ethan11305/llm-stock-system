# P0 Embedding Pipeline 實作現狀分析

**撰寫日期：** 2026-04-17  
**分析基礎：** 代碼實際審查 + Codex AI 確認

---

## 📊 三個 PR 的完成度

### ✅ PR 1：基礎設施（1-2 天）— **COMPLETED**

| 任務 | 狀態 | 檔案位置 | 備註 |
|---|---|---|---|
| DocumentChunker | ✅ | `src/llm_stock_system/services/document_chunker.py` | 完整實現，支援短文整篇 + 長文按段落切 |
| EmbeddingService | ✅ | `src/llm_stock_system/services/embedding_service.py` | 完整實現，包括 OpenAI API 呼叫、批量寫入、重試邏輯 |
| Schema migration | ✅ | `db/sql/004_embedding_enhancements.sql` | 已準備，含 UNIQUE constraint + HNSW 索引 |
| Unit tests | ✅ | `tests/test_*` | 基本測試已覆蓋 |

**狀態：** 可直接使用，無待辦事項。

---

### ✅ PR 2：回填 + 寫入流程（1-2 天）— **COMPLETED**

| 任務 | 狀態 | 檔案位置 | 備註 |
|---|---|---|---|
| backfill_embeddings.py | ✅ | `scripts/backfill_embeddings.py` | 完整實現，支援 --batch-size、--ticker、--all 等參數 |
| QueryDataHydrator 嵌入 | ✅ | `src/llm_stock_system/services/query_data_hydrator.py:251` | 已在 hydrate() 方法中自動觸發 `embed_and_store()` |
| Integration test | ✅ | `tests/test_query_data_hydrator_phase2.py` | 已覆蓋 build → upsert → embed 流程 |

**完整流程：**
```
build_documents(query)
  ↓ (postgres_market_data.py:967)
upsert_documents(docs)
  ↓ (postgres_market_data.py:4725)
embed_and_store(docs, chunker)
  ↓ (embedding_service.py)
document_embeddings 表
```

**狀態：** 核心流程已完成，可投入生產。

---

### ✅ PR 3：語意檢索 + 混合檢索（2-3 天）— **COMPLETED**

| 任務 | 狀態 | 檔案位置 | 備註 |
|---|---|---|---|
| VectorRetrievalAdapter | ✅ | `src/llm_stock_system/adapters/vector_retrieval.py` | 完整實現 pgvector cosine 查詢 |
| HybridRetrievalLayer | ✅ | `src/llm_stock_system/layers/retrieval_layer.py:43` | 混合排序已實現（語意 0.6 + metadata 0.4） |
| app.py 集成 | ✅ | `src/llm_stock_system/api/app.py:~380` | 條件化使用 HybridRetrievalLayer |
| End-to-end test | 🟡 | 部分覆蓋 | 需確認實際查詢結果質量 |

**狀態：** 代碼實現完整，建議進行實際測試驗證。

---

## 🔍 架構缺口分析文件中的過時說法 vs 實際代碼

### 問題 1：Documents 寫入時機

**文件說法：** ⚠️ 過時
> 目前 `build_documents()` 是程式化合成文件，`upsert_documents()` 還是 stub。embedding pipeline 就需要從其他表（`stock_news_articles` 等）直接讀取。

**實際代碼：** ✅ 已解決
```python
# postgres_market_data.py:4725
def upsert_documents(self, documents: list[Document]) -> int:
    """將 Document 物件批次寫入（或更新）documents 表。"""
    # 完整實現，非 stub
```

```python
# query_data_hydrator.py:231-251
documents = build_documents(query)
upsert_written = document_repository.upsert_documents(documents)
written = embedding_service.embed_and_store(documents, chunker)
```

**決策：** ✅ 沿用現狀 — `documents 表先寫 → 再做 embedding`，無需改動。

---

### 問題 2：Chunk 粒度是否需要 Profile-Specific

**文件說法：** ⚠️ 保守估計
> 需要新增 profile-specific chunker，分新聞、財報、法說會三個策略

**實際代碼：** ✅ 已簡化
```python
# document_chunker.py:39-48
class DocumentChunker:
    """通用策略：短文整篇、長文按段落切。"""
    def __init__(self, max_chunk_tokens: int = 500, overlap_tokens: int = 50):
        # 一個通用實現，不分 profile
```

**Codex AI 確認：**
> `build_documents()` 產出的多數是摘要型文件，不是長篇逐字稿。  
> 先不做 profile-specific chunker；如要優化，只先對 `news_article` 或超長文本加特例。

**決策：** ✅ 保持現狀 — 通用 chunker 足以應對，無需繁複化。

---

### 問題 3：HNSW 內存限制

**文件說法：** ⚠️ 架構層級的顧慮
> 現有 schema 用 IVFFlat（需資料量 > lists × 10 才有效），建議改 HNSW，但需討論內存限制

**實際代碼：** ✅ 已解決
```sql
-- db/sql/004_embedding_enhancements.sql:45-52
DROP INDEX IF EXISTS idx_document_embeddings_ivfflat;

CREATE INDEX IF NOT EXISTS idx_document_embeddings_hnsw
    ON document_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

**Codex AI 確認：**
> 程式已經從 IVFFlat 改成 HNSW。  
> 目前 repo 和 docker-compose.yml 沒有 `maintenance_work_mem`、`shared_buffers` 這類設定，  
> 所以這不是架構 blocker，而是**後續 DB 調參項**。

**決策：** ✅ HNSW 已落地，內存調參納入 **優化清單** 而非阻擋項。

---

## 🎯 現在應該做什麼？

### **Option A：質量驗證 + 文件更新（推薦）**

**理由：** P0 代碼已完成，但需確保：
1. Embedding 品質是否足以支撐後續 CoVe/Sentiment
2. Hybrid 權重（0.6/0.4）是否需要 A/B test 調整
3. 文件版本保持同步

**行動清單：**

1. **驗證 Embedding 質量**（1-2 小時）
   - 手工查詢測試：「台積電最近有什麼供應鏈風險？」
   - 對比 metadata 檢索 vs 語意檢索的結果差異
   - 確認 semantic_weight 的權重是否合理

2. **執行 Schema Migration**（15 分鐘）
   ```bash
   # 確認 pgvector extension 已安裝
   psql -U postgres -d llm_stock < db/sql/004_embedding_enhancements.sql
   ```

3. **執行 Backfill（必要時）**（30 分鐘 - 2 小時，取決於文件量）
   ```bash
   python scripts/backfill_embeddings.py --dry-run  # 先看預期結果
   python scripts/backfill_embeddings.py --batch-size 50  # 開始回填
   ```

4. **更新架構文件**（1-2 小時）
   - 修正 `docs/architecture.md`：補充 embedding pipeline 的 flow
   - 修正 `架構缺口分析文件.md`：
     - **缺口零** → 改為「已完成」，但註記後續調參項
     - 刪除 3 個過時的問題，改為「已解決」清單
     - 推薦優先順序改為：「現在應專注於 P1（並行化）和 P2（Routing）」

---

### **Option B：跳過驗證，直接進行 P1（激進）**

**理由：** 相信代碼品質，盡快推進並行化帶來的效能提升。

**風險：** 如果 embedding 質量有問題，發現時間會比較晚，修復成本更高。

**不推薦**

---

### **Option C：深度測試 + 微調（保守）**

**理由：** P0 涉及 OpenAI API 成本，需確保投入產出比。

**行動清單：**
1. 寫完整的 E2E test（查詢 → hybrid 檢索 → 生成 → 驗證結果品質）
2. 測試 5-10 個真實查詢場景
3. 記錄效果指標（recall、precision、token 消耗）
4. 決定是否需要微調 hybrid 權重

**預估工時：** 3-5 小時

---

## 📋 最終決策：**Option A + 簡化版**

### 立即行動（今天）

1. **20 分鐘**：執行 schema migration
   ```bash
   psql -U postgres -d llm_stock < db/sql/004_embedding_enhancements.sql
   ```

2. **30 分鐘**：用 dry-run 檢查 backfill 預期結果
   ```bash
   python scripts/backfill_embeddings.py --dry-run
   ```

3. **1 小時**：手工測試 1-2 個查詢
   - 「台積電本益比」（應拉財報 + 新聞 → 語意檢索有助嗎？）
   - 「航運類股最近利空」（應拉航運新聞 → 語意檢索精度檢驗）
   - 對比 metadata 只檢索 vs hybrid 的結果差異

4. **1 小時**：更新架構文件
   - 在 `docs/architecture.md` 加上 embedding pipeline diagram
   - 在 `架構缺口分析文件.md` 開頭補 `## 更新（2026-04-17）` 章節，說明 P0 已完成

### 後續順序

- **本週內**：如無異常，正式執行 backfill（預計 30 分鐘 - 1 小時）
- **下週**：開始 P1（並行化）和 P2（Policy Registry）

---

## 💡 給 Codex AI 的反饋

感謝您指出這 3 個關鍵問題已有代碼答案，避免了我基於過時文件造成的誤導。建議後續：

1. **定期同步文件**：每個 PR 完成後，檢查 architecture.md 是否需要更新
2. **標記更新時間**：在 `架構缺口分析文件.md` 加上「Last Sync」時間戳，便於判斷新鮮度
3. **分離 Current vs Target**：如您提議的，拆成兩份或兩個章節，避免混淆

---

## 檢查清單

- [ ] Schema migration 已執行
- [ ] Backfill dry-run 已驗證
- [ ] 手工測試 1-2 個查詢場景
- [ ] 架構文件已更新
- [ ] Backfill 已在生產環境執行（如無異常）
- [ ] P1 準備開始

