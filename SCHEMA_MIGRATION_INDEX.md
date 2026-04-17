# Schema Migration 資源總覽

**準備時間：** 2026-04-17  
**預期執行時間：** 5-15 分鐘

---

## 📦 已準備好的文件清單

### 1️⃣ **核心 SQL 文件** ✅

**文件：** `db/sql/004_embedding_enhancements.sql`
- ✅ 已修復（原檔被截斷，已補全）
- ✅ 包含註解和前置條件檢查
- ✅ 設計成冪等（可重複執行）
- ✅ 包含驗證和報告邏輯

**包含的變更：**
- UNIQUE constraint: `(document_id, chunk_index)`
- Metadata 欄位：`ticker`, `published_at`
- 索引升級：IVFFlat → HNSW
- 前置檢查：pgvector 已安裝

---

### 2️⃣ **執行指南** 📖

**文件：** `SCHEMA_MIGRATION_GUIDE.md`
- 詳細的分步驟教程
- 5 個執行步驟（包括驗證）
- 常見問題排查表
- 回滾方案（如需要）
- 執行後的後續工作清單

**適合：** 希望詳細了解每個步驟的用戶

---

### 3️⃣ **一鍵執行腳本** 🚀

**文件：** `scripts/execute_schema_migration.sh`
- 自動備份資料庫
- 自動執行 migration
- 自動驗證結果
- 詳細的命令行選項

**使用方式：**
```bash
# 完整流程（推薦）
bash scripts/execute_schema_migration.sh

# 僅備份
bash scripts/execute_schema_migration.sh --backup-only

# 僅驗證
bash scripts/execute_schema_migration.sh --verify-only

# 說明
bash scripts/execute_schema_migration.sh --help
```

**適合：** 想快速執行的用戶

---

### 4️⃣ **快速參考卡** 📋

**文件：** `SCHEMA_MIGRATION_QUICK_REFERENCE.txt`
- 5 分鐘版本的執行步驟
- 關鍵驗證指令
- 常見問題排查速查表
- 後續工作檢查清單

**適合：** 需要快速查閱的用戶

---

### 5️⃣ **現狀分析文件** 📊

**文件：** `P0_IMPLEMENTATION_STATUS.md`
- P0 三個 PR 的完成度統計
- 架構文件中的過時說法對比
- 決策理由與行動計劃

**適合：** 想了解 P0 整體進度的用戶

---

## 🎯 建議執行順序

### 【快速用戶】 ⏱️ 5 分鐘

```
1. 讀 SCHEMA_MIGRATION_QUICK_REFERENCE.txt（2 分鐘）
2. 執行 bash scripts/execute_schema_migration.sh（5-15 分鐘）
3. 完成 ✅
```

### 【仔細用戶】 ⏱️ 15-30 分鐘

```
1. 讀 SCHEMA_MIGRATION_GUIDE.md（5 分鐘）
2. 執行 Step 1-4（手動或自動）（5-15 分鐘）
3. 檢查 Step 5 的驗證結果（2-3 分鐘）
4. 執行後續工作（1-2 分鐘）
```

### 【需要了解的用戶】 ⏱️ 30-60 分鐘

```
1. 讀 P0_IMPLEMENTATION_STATUS.md（10 分鐘）
2. 讀 SCHEMA_MIGRATION_GUIDE.md（10 分鐘）
3. 手動執行 migration + 驗證（10 分鐘）
4. 執行 backfill --dry-run（5 分鐘）
5. 手工測試 1-2 個查詢場景（5-10 分鐘）
```

---

## 📝 執行前的清單

- [ ] 已讀過相關文件（至少快速參考卡）
- [ ] 備份命令已準備好（或使用自動腳本）
- [ ] PostgreSQL 連接已測試
- [ ] pgvector 已安裝（檢查：`psql -c "CREATE EXTENSION vector;"`）
- [ ] 有 2-5 GB 磁碟空間用於備份
- [ ] 有 15 分鐘的時間不做其他操作

---

## ✅ 執行成功的驗證標誌

Migration 完成後，應該看到：

1. **Constraint 建立：**
   ```
   uq_document_embeddings_doc_chunk | UNIQUE
   ```

2. **欄位新增：**
   ```
   ticker       | character varying
   published_at | timestamp
   ```

3. **索引升級：**
   ```
   idx_document_embeddings_hnsw | USING hnsw (embedding vector_cosine_ops)
   ```

如以上三項都出現，就是 ✅ 成功

---

## 🚨 可能遇到的問題

| 狀況 | 解決方案 |
|---|---|
| pgvector 未安裝 | 執行 `001_extensions.sql` |
| 權限不足 | 使用 postgres 超級用戶執行 |
| 索引建立超慢 | 正常現象，耐心等待 |
| 執行腳本失敗 | 查看 `/tmp/migration_*.log` 日誌 |
| 不確定是否成功 | 執行 `scripts/execute_schema_migration.sh --verify-only` |

---

## 📞 獲得幫助

### 快速問題
→ 查看 `SCHEMA_MIGRATION_QUICK_REFERENCE.txt` 的「常見問題快速排查」

### 詳細問題
→ 查看 `SCHEMA_MIGRATION_GUIDE.md` 的「常見問題 & 排查」

### 理解 P0 整體
→ 查看 `P0_IMPLEMENTATION_STATUS.md`

### 需要備份恢復
→ 參考 `SCHEMA_MIGRATION_GUIDE.md` 的「回滾方案」

---

## 🎉 完成後

Migration 成功後，下一步：

1. **回填 Embeddings：**
   ```bash
   python scripts/backfill_embeddings.py --dry-run
   python scripts/backfill_embeddings.py --batch-size 50
   ```

2. **手工測試查詢：**
   - 「台積電供應鏈風險」
   - 「航運類股最近利空」
   - 「本益比相對低」

3. **更新文件：**
   - `docs/architecture.md`
   - `架構缺口分析文件.md`

4. **進行 P1（並行化）**

---

## 📊 預期時間表

| 活動 | 工時 | 狀態 |
|---|---|---|
| 準備工作（備份） | 2 分 | ⏳ |
| Schema Migration | 1-2 分 | ⏳ |
| 驗證 | 1-2 分 | ⏳ |
| **小計** | **5-15 分** | ⏳ |
| Backfill dry-run | 30 秒 | ⏳ |
| Backfill 執行 | 30 分 - 2 小時 | ⏳ |
| 手工測試 | 15-30 分 | ⏳ |
| 文件更新 | 1-2 小時 | ⏳ |

---

**祝執行順利！🚀**

有任何問題，查閱相應的文件即可找到答案。
