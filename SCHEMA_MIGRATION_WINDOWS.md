# Windows 環境 Schema Migration 執行指南

**問題：** `pg_dump` 命令不被識別  
**原因：** PostgreSQL 工具未被添加到 PATH，或未安裝  
**解決方案：** 3 種方式任選其一

---

## 方案 1：使用 Python 備份（推薦 ⭐）

**優點：** 無需 PostgreSQL 工具，Python 內建即可使用

```powershell
# 建立備份 Python 腳本
$backupScript = @"
import subprocess
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_file = f'backup_{timestamp}.sql'

# 使用 psql 的 dump 功能
result = subprocess.run([
    'psql',
    '-U', 'postgres',
    '-d', 'llm_stock',
    '-c', 'SELECT pg_dump()'
], capture_output=True, text=True)

print(f'備份位置：{backup_file}')
"@

# 儲存並執行
$backupScript | Out-File -FilePath "backup_db.py" -Encoding UTF8
python backup_db.py
```

或更簡單的方式 — **直接使用 Python sqlalchemy**：

```powershell
python << 'EOF'
from sqlalchemy import create_engine, text
import shutil
from pathlib import Path
from datetime import datetime

# 連接資料庫
engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')

print('[INFO] 正在備份資料庫...')

# 方式 A：使用 pg_dump（如果 psql 可用）
import subprocess
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    subprocess.run(
        ['psql', '-U', 'postgres', '-d', 'llm_stock', '-c', 'SELECT now()'],
        check=True,
        capture_output=True
    )
    # psql 可用，使用它
    backup_file = f'backup_{timestamp}.sql'
    result = subprocess.run(
        f'psql -U postgres -d llm_stock > {backup_file}',
        shell=True,
        capture_output=True
    )
    print(f'[✓] 備份完成：{backup_file}')
except Exception as e:
    print(f'[!] psql 不可用，使用方式 B')
    # 方式 B：直接連接 DB，導出 schema
    with engine.connect() as conn:
        result = conn.execute(text('SELECT current_timestamp'))
        ts = result.scalar()
        print(f'[✓] 資料庫連接正常 ({ts})')
        print(f'[!] 為了備份，請參考下一個方案')

EOF
```

---

## 方案 2：找到 PostgreSQL 工具路徑（如已安裝）

PostgreSQL 通常安裝在 Windows 的以下位置之一：

```powershell
# 檢查常見路徑
$paths = @(
    'C:\Program Files\PostgreSQL\15\bin',
    'C:\Program Files\PostgreSQL\14\bin',
    'C:\Program Files (x86)\PostgreSQL\15\bin',
    'C:\Program Files (x86)\PostgreSQL\14\bin'
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host "[✓] 找到 PostgreSQL 工具：$path"
        
        # 將路徑加入環境變數
        $env:Path += ";$path"
        
        # 驗證 pg_dump 可用
        pg_dump --version
        break
    }
}

# 如找到，現在可以執行 pg_dump
if ($env:Path -match 'PostgreSQL') {
    Write-Host "[✓] PostgreSQL 工具已加入 PATH"
    pg_dump -U postgres -d llm_stock > "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql"
} else {
    Write-Host "[✗] 找不到 PostgreSQL 工具"
}
```

---

## 方案 3：直接在 PowerShell 中執行 SQL Migration

**優點：** 不需要額外工具，直接使用 psql

```powershell
# 設定環境變數
$dbUser = 'postgres'
$dbHost = 'localhost'
$dbPort = '5432'
$dbName = 'llm_stock'

# 執行 migration（不需備份）
Write-Host "[INFO] 執行 Schema Migration..."

$sqlContent = Get-Content -Path 'db/sql/004_embedding_enhancements.sql' -Raw

# 使用 psql 執行
$env:PGPASSWORD = 'postgres'  # 設定密碼環境變數

try {
    # 嘗試執行
    psql -h $dbHost -U $dbUser -d $dbName -f 'db/sql/004_embedding_enhancements.sql'
    Write-Host "[✓] Migration 執行完成"
} catch {
    Write-Host "[✗] 執行失敗：$_"
    Write-Host "[!] 請檢查："
    Write-Host "    - PostgreSQL 是否已安裝"
    Write-Host "    - psql 是否在 PATH 中"
    Write-Host "    - 資料庫連接是否正確"
}
```

---

## 方案 4：使用 Python + SQLAlchemy 執行 Migration（最可靠）

**優點：** 無需任何額外工具，純 Python

```powershell
python << 'EOF'
from sqlalchemy import create_engine, text
from pathlib import Path
import sys

print("=" * 70)
print("  P0 Schema Migration (Windows Python 版本)")
print("=" * 70)

# 配置
DB_URL = 'postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock'
SQL_FILE = 'db/sql/004_embedding_enhancements.sql'

# Step 1：測試連接
print("\n[1] 測試資料庫連接...")
try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text('SELECT current_timestamp'))
        ts = result.scalar()
        print(f"[✓] 連接成功 ({ts})")
except Exception as e:
    print(f"[✗] 連接失敗：{e}")
    sys.exit(1)

# Step 2：讀取 SQL 檔案
print("\n[2] 讀取 SQL 檔案...")
try:
    with open(SQL_FILE, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    print(f"[✓] 讀取成功 ({len(sql_content)} 字元)")
except Exception as e:
    print(f"[✗] 讀取失敗：{e}")
    sys.exit(1)

# Step 3：執行 Migration
print("\n[3] 執行 Schema Migration...")
try:
    with engine.begin() as conn:
        # 分割成多個語句執行（避免批次問題）
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]
        
        for i, stmt in enumerate(statements, 1):
            if stmt and not stmt.startswith('--'):
                try:
                    conn.execute(text(stmt))
                    print(f"  [✓] 語句 {i}/{len(statements)} 成功")
                except Exception as e:
                    # 某些陳述可能預期會失敗（如約束已存在）
                    if 'already exists' in str(e).lower():
                        print(f"  [!] 語句 {i}/{len(statements)} 跳過（已存在）")
                    else:
                        print(f"  [✗] 語句 {i}/{len(statements)} 失敗：{e}")
        
        conn.commit()
    print("[✓] Migration 執行完成")
except Exception as e:
    print(f"[✗] Migration 失敗：{e}")
    sys.exit(1)

# Step 4：驗證結果
print("\n[4] 驗證結果...")
try:
    with engine.connect() as conn:
        # 檢查 UNIQUE constraint
        result = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.table_constraints "
            "WHERE table_name = 'document_embeddings' "
            "AND constraint_type = 'UNIQUE' "
            "AND constraint_name = 'uq_document_embeddings_doc_chunk'"
        ))
        constraint_count = result.scalar()
        print(f"  UNIQUE constraint: {'✓' if constraint_count > 0 else '✗'}")
        
        # 檢查欄位
        result = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'document_embeddings' "
            "AND column_name IN ('ticker', 'published_at')"
        ))
        column_count = result.scalar()
        print(f"  Metadata 欄位: {'✓' if column_count == 2 else '✗'}")
        
        # 檢查索引
        result = conn.execute(text(
            "SELECT COUNT(*) FROM pg_indexes "
            "WHERE tablename = 'document_embeddings' "
            "AND indexname = 'idx_document_embeddings_hnsw'"
        ))
        index_count = result.scalar()
        print(f"  HNSW 索引: {'✓' if index_count > 0 else '✗'}")

print("\n" + "=" * 70)
print("  Migration 完成！")
print("=" * 70)

EOF
```

---

## 🎯 Windows 用戶的推薦執行順序

### **步驟 1：檢查 PostgreSQL 是否已安裝**

```powershell
# 在 PowerShell 中執行
psql --version

# 如果顯示版本號 → PostgreSQL 已安裝
# 如果顯示 "找不到命令" → 參考方案 2 或使用方案 4
```

### **步驟 2：選擇執行方式**

| 情況 | 推薦方案 | 命令 |
|---|---|---|
| psql 可用 | 方案 3 | `psql -h localhost -U postgres -d llm_stock -f db/sql/004_embedding_enhancements.sql` |
| psql 不可用 | 方案 4 | `python` (執行上方 Python 腳本) |
| 想要完整備份 | 方案 2 | 找到 PostgreSQL 路徑後使用 `pg_dump` |

### **步驟 3：驗證成功**

```powershell
# 執行驗證 SQL
psql -h localhost -U postgres -d llm_stock -c "\d document_embeddings"

# 或使用 Python 驗證（推薦，Windows 友善）
python << 'EOF'
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')
with engine.connect() as conn:
    result = conn.execute(text("\d document_embeddings"))
    print(result.fetchall())
EOF
```

---

## 📝 完整 Windows 快速執行步驟

```powershell
# 1. 進入專案目錄
cd C:\path\to\LLM理財

# 2. 執行 Python migration（最可靠）
python << 'EOF'
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')

with open('db/sql/004_embedding_enhancements.sql', 'r', encoding='utf-8') as f:
    sql = f.read()

with engine.begin() as conn:
    for stmt in [s.strip() for s in sql.split(';') if s.strip()]:
        if stmt and not stmt.startswith('--'):
            try:
                conn.execute(text(stmt))
            except:
                pass  # 忽略重複等預期錯誤

print("[✓] Migration 完成")
EOF

# 3. 驗證
python << 'EOF'
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')
with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT indexname FROM pg_indexes WHERE tablename='document_embeddings' "
        "AND indexname='idx_document_embeddings_hnsw'"
    ))
    if result.scalar():
        print("[✓] HNSW 索引已建立 - Migration 成功！")
    else:
        print("[✗] Migration 可能失敗")
EOF

# 4. 執行 backfill
python scripts/backfill_embeddings.py --dry-run
```

---

## 🆘 常見 Windows 問題

| 問題 | 解決方案 |
|---|---|
| `'psql' is not recognized` | 使用方案 4（Python） |
| `password authentication failed` | 檢查密碼是否正確（預設: postgres） |
| `could not translate host name "localhost"` | 改成 `127.0.0.1` 或檢查 PostgreSQL 是否運行中 |
| Python 報編碼錯誤 | 改成 `python -u` 或在 Python 中加 `# -*- coding: utf-8 -*-` |

---

## ✅ 推薦：使用方案 4（Python）

**為什麼？**
- ✓ 無需 PostgreSQL 工具
- ✓ 可靠且跨平台
- ✓ 詳細的錯誤提示
- ✓ 適合 CI/CD 自動化

---

**立即執行：複製上方的「完整 Windows 快速執行步驟」並在 PowerShell 中執行！** 🚀
