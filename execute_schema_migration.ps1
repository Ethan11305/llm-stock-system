# execute_schema_migration.ps1
#
# P0 Schema Migration 一鍵執行腳本 (Windows PowerShell 版本)
#
# 用法：
#   .\execute_schema_migration.ps1
#
# 特點：
#   - 無需 pg_dump，使用 Python + SQLAlchemy
#   - 自動驗證結果
#   - 詳細的進度報告
#
# 前置需求：
#   - Python 已安裝
#   - psycopg 模組已安裝 (pip install psycopg[binary])
#   - PostgreSQL 正在運行

param(
    [ValidateSet('full', 'migrate-only', 'verify-only')]
    [string]$Mode = 'full',

    [string]$DbHost = 'localhost',
    [string]$DbPort = '5432',
    [string]$DbUser = 'postgres',
    [string]$DbName = 'llm_stock'
)

# ─────────────────────────────────────────────────────────────────────────────
# 顏色定義
# ─────────────────────────────────────────────────────────────────────────────

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[✓] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[✗] $args" -ForegroundColor Red }
function Write-Warning { Write-Host "[!] $args" -ForegroundColor Yellow }

# ─────────────────────────────────────────────────────────────────────────────
# Python 執行函數
# ─────────────────────────────────────────────────────────────────────────────

function Invoke-PythonScript {
    param(
        [string]$Script
    )

    $tempFile = [System.IO.Path]::GetTempFileName()
    $tempFile = $tempFile -replace '\.tmp$', '.py'

    try {
        Set-Content -Path $tempFile -Value $Script -Encoding UTF8
        $output = python $tempFile 2>&1

        if ($LASTEXITCODE -eq 0) {
            return @{ Success = $true; Output = $output }
        } else {
            return @{ Success = $false; Output = $output }
        }
    } finally {
        Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 測試連接
# ─────────────────────────────────────────────────────────────────────────────

function Test-PostgresConnection {
    Write-Info "測試 PostgreSQL 連接..."

    $script = @"
import sys
try:
    from sqlalchemy import create_engine, text
    db_url = 'postgresql+psycopg://$DbUser:postgres@$DbHost`:$DbPort/$DbName'
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text('SELECT current_timestamp'))
        ts = result.scalar()
    print(f"Successfully connected: {ts}")
    sys.exit(0)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
"@

    $result = Invoke-PythonScript $script

    if ($result.Success) {
        Write-Success "PostgreSQL 連接成功"
        return $true
    } else {
        Write-Error "PostgreSQL 連接失敗"
        Write-Host $result.Output
        return $false
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 執行 Migration
# ─────────────────────────────────────────────────────────────────────────────

function Invoke-SchemaMigration {
    Write-Info "執行 Schema Migration..."

    $sqlFile = "db/sql/004_embedding_enhancements.sql"

    if (-not (Test-Path $sqlFile)) {
        Write-Error "SQL 檔案不存在：$sqlFile"
        return $false
    }

    $script = @"
import sys
from sqlalchemy import create_engine, text

try:
    db_url = 'postgresql+psycopg://$DbUser:postgres@$DbHost`:$DbPort/$DbName'
    engine = create_engine(db_url)

    with open('$sqlFile', 'r', encoding='utf-8') as f:
        sql_content = f.read()

    # 分割成多個語句
    statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]

    with engine.begin() as conn:
        for i, stmt in enumerate(statements, 1):
            if stmt:
                try:
                    conn.execute(text(stmt))
                    print(f"✓ Statement {i}/{len(statements)}")
                except Exception as e:
                    if 'already exists' in str(e).lower():
                        print(f"! Statement {i}/{len(statements)} (already exists, skipped)")
                    else:
                        raise

    print("MIGRATION_SUCCESS")
    sys.exit(0)

except Exception as e:
    print(f"MIGRATION_FAILED: {e}")
    sys.exit(1)
"@

    $result = Invoke-PythonScript $script

    Write-Host $result.Output

    if ($result.Output -match "MIGRATION_SUCCESS") {
        Write-Success "Schema Migration 執行完成"
        return $true
    } else {
        Write-Error "Schema Migration 失敗"
        return $false
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 驗證 Migration
# ─────────────────────────────────────────────────────────────────────────────

function Verify-SchemaMigration {
    Write-Info "驗證 Migration 結果..."

    $script = @"
import sys
from sqlalchemy import create_engine, text

try:
    db_url = 'postgresql+psycopg://$DbUser:postgres@$DbHost`:$DbPort/$DbName'
    engine = create_engine(db_url)

    checks = {
        'UNIQUE constraint': "SELECT COUNT(*) FROM information_schema.table_constraints WHERE table_name = 'document_embeddings' AND constraint_type = 'UNIQUE' AND constraint_name = 'uq_document_embeddings_doc_chunk'",
        'Metadata columns': "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'document_embeddings' AND column_name IN ('ticker', 'published_at')",
        'HNSW index': "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'document_embeddings' AND indexname = 'idx_document_embeddings_hnsw'"
    }

    results = {}
    with engine.connect() as conn:
        for check_name, query in checks.items():
            result = conn.execute(text(query))
            count = result.scalar()
            results[check_name] = count > 0 if check_name != 'Metadata columns' else count == 2

    all_passed = all(results.values())

    for check_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")

    if all_passed:
        print("VERIFICATION_SUCCESS")
        sys.exit(0)
    else:
        print("VERIFICATION_FAILED")
        sys.exit(1)

except Exception as e:
    print(f"VERIFICATION_ERROR: {e}")
    sys.exit(1)
"@

    $result = Invoke-PythonScript $script

    Write-Host $result.Output

    if ($result.Output -match "VERIFICATION_SUCCESS") {
        Write-Success "驗證成功"
        return $true
    } else {
        Write-Error "驗證失敗"
        return $false
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

function Main {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║   P0 Schema Migration (Windows PowerShell)             ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""

    Write-Info "配置："
    Write-Host "  Host: $DbHost"
    Write-Host "  Port: $DbPort"
    Write-Host "  User: $DbUser"
    Write-Host "  Database: $DbName"
    Write-Host "  Mode: $Mode"
    Write-Host ""

    # Step 1: 測試連接
    if ($Mode -eq 'full' -or $Mode -eq 'migrate-only') {
        if (-not (Test-PostgresConnection)) {
            Write-Error "無法連接到資料庫，請檢查配置"
            exit 1
        }
    }
    Write-Host ""

    # Step 2: 執行 Migration
    if ($Mode -eq 'full' -or $Mode -eq 'migrate-only') {
        if (-not (Invoke-SchemaMigration)) {
            Write-Error "Schema Migration 失敗"
            exit 1
        }
    }
    Write-Host ""

    # Step 3: 驗證
    if ($Mode -eq 'full' -or $Mode -eq 'verify-only') {
        if (-not (Verify-SchemaMigration)) {
            Write-Error "驗證失敗，Migration 可能未完全成功"
            exit 1
        }
    }
    Write-Host ""

    # 完成
    Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Success "P0 Schema Migration 已完成！"
    Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""

    Write-Info "下一步："
    Write-Host "  1. 執行 backfill: python scripts/backfill_embeddings.py --dry-run"
    Write-Host "  2. 手工測試查詢"
    Write-Host "  3. 更新架構文件"
    Write-Host ""
}

# 執行主程序
Main
