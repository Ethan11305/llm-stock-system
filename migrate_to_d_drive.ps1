# migrate_to_d_drive.ps1
#
# P0 Schema Migration - Windows D 槽遷移腳本
#
# ⚠️ 重要：此腳本需要以管理員身份執行
#
# 用法：
#   .\migrate_to_d_drive.ps1
#
# 此腳本會執行：
# 1. 備份 PostgreSQL 數據庫
# 2. 停止 PostgreSQL 服務
# 3. 複製數據庫檔案到 D:\PostgreSQL
# 4. 複製項目檔案到 D:\LLM理財
# 5. 重新啟動 PostgreSQL 並驗證

# 確認管理員權限
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "❌ 此腳本需要以管理員身份執行" -ForegroundColor Red
    Write-Host "請右鍵點擊 PowerShell，選擇『以管理員身份執行』" -ForegroundColor Yellow
    exit 1
}

# 顏色定義
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[✓] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[✗] $args" -ForegroundColor Red }
function Write-Warning { Write-Host "[!] $args" -ForegroundColor Yellow }

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

$SourceProject = "C:\Users\ASUS\Downloads\LLM理財"
$DestProject = "D:\LLM理財"
$PGVersion = "15"  # 改成您的 PostgreSQL 版本（通常 14 或 15）
$PGDataSource = "C:\Program Files\PostgreSQL\$PGVersion\data"
$PGDataDest = "D:\PostgreSQL\$PGVersion\data"
$PGService = "postgresql-x64-$PGVersion"
$BackupDir = "C:\backups"

# ─────────────────────────────────────────────────────────────────────────────
# 第 1 步：確認配置
# ─────────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  PostgreSQL + 項目遷移至 D 槽                             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Info "驗證配置..."
Write-Host "  源項目路徑: $SourceProject"
Write-Host "  目標項目路徑: $DestProject"
Write-Host "  PG 版本: $PGVersion"
Write-Host "  PG 服務名: $PGService"
Write-Host ""

# 檢查源項目是否存在
if (-not (Test-Path $SourceProject)) {
    Write-Error "源項目不存在：$SourceProject"
    exit 1
}

# 檢查 PostgreSQL 數據目錄是否存在
if (-not (Test-Path $PGDataSource)) {
    Write-Error "PostgreSQL 數據目錄不存在：$PGDataSource"
    Write-Warning "請檢查 PostgreSQL 版本號（當前設置為 $PGVersion）"
    exit 1
}

Write-Success "配置驗證完成"
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 2 步：備份 PostgreSQL 數據庫
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "備份 PostgreSQL 數據庫..."

New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
$BackupFile = "$BackupDir\llm_stock_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql"

try {
    # 使用 pg_dump 備份
    $PGBinPath = "C:\Program Files\PostgreSQL\$PGVersion\bin"
    $env:PGPASSWORD = 'postgres'

    & "$PGBinPath\pg_dump.exe" -U postgres -d llm_stock > $BackupFile

    if (Test-Path $BackupFile) {
        $FileSize = (Get-Item $BackupFile).Length / 1MB
        Write-Success "備份完成：$BackupFile ($([Math]::Round($FileSize, 2)) MB)"
    } else {
        throw "備份檔案創建失敗"
    }
} catch {
    Write-Error "備份失敗：$_"
    Write-Warning "繼續遷移（備份可能已部分成功）"
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 3 步：停止 PostgreSQL 服務
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "停止 PostgreSQL 服務..."

try {
    $service = Get-Service $PGService -ErrorAction SilentlyContinue

    if ($null -eq $service) {
        Write-Warning "找不到服務 '$PGService'"
        Write-Warning "嘗試自動尋找..."

        $allPGServices = Get-Service | Where-Object { $_.Name -like "*postgresql*" }
        if ($allPGServices) {
            Write-Host "找到的 PostgreSQL 服務："
            $allPGServices | ForEach-Object { Write-Host "  - $($_.Name)" }
            Write-Error "請手動更新腳本中的 \$PGService 變數"
            exit 1
        }
    }

    if ($service.Status -eq "Running") {
        Stop-Service -Name $PGService -Force
        Start-Sleep -Seconds 2
        Write-Success "PostgreSQL 服務已停止"
    } else {
        Write-Warning "PostgreSQL 服務已停止"
    }
} catch {
    Write-Error "停止服務失敗：$_"
    exit 1
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 4 步：複製 PostgreSQL 數據目錄到 D 槽
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "複製 PostgreSQL 數據目錄到 D 槽..."
Write-Warning "此過程可能需要 10-30 分鐘，請耐心等待..."

try {
    New-Item -ItemType Directory -Path $PGDataDest -Force | Out-Null

    Write-Host "  複製中...（檔案數量較多，進度較慢）"
    Copy-Item -Path "$PGDataSource\*" -Destination $PGDataDest -Recurse -Force

    # 驗證複製
    $SourceFiles = (Get-ChildItem -Path $PGDataSource -Recurse -Force | Measure-Object).Count
    $DestFiles = (Get-ChildItem -Path $PGDataDest -Recurse -Force | Measure-Object).Count

    Write-Success "PostgreSQL 數據目錄已複製"
    Write-Host "  源檔案數：$SourceFiles，目標檔案數：$DestFiles"
} catch {
    Write-Error "複製 PostgreSQL 數據失敗：$_"
    exit 1
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 5 步：設置新數據目錄的權限
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "設置 PostgreSQL 數據目錄權限..."

try {
    # 移除繼承的權限
    icacls "$PGDataDest" /inheritance:r /T /C | Out-Null

    # 給予 SYSTEM 完全控制
    icacls "$PGDataDest" /grant:r "SYSTEM:(F)" /T /C | Out-Null

    Write-Success "權限設置完成"
} catch {
    Write-Warning "權限設置失敗，但繼續嘗試啟動服務：$_"
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 6 步：更新 PostgreSQL 服務配置
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "更新 PostgreSQL 服務配置..."
Write-Warning "⚠️ 此步驟修改服務啟動參數，請確認操作"

$PGExePath = "C:\Program Files\PostgreSQL\$PGVersion\bin\pg_ctl.exe"
$PGBinPath = "C:\Program Files\PostgreSQL\$PGVersion\bin\postgres.exe"

try {
    # 構建新的啟動命令
    $NewBinPath = "`"$PGExePath`" runservice -N `"$PGService`" -D `"$PGDataDest`" -p `"$PGBinPath`""

    Write-Host "  舊啟動路徑：(從服務屬性查看)"
    Write-Host "  新啟動路徑：$NewBinPath"

    # 修改服務配置
    sc.exe config "$PGService" binPath= $NewBinPath | Out-Null

    Write-Success "服務配置已更新"
} catch {
    Write-Error "服務配置更新失敗：$_"
    Write-Warning "您可能需要手動修改服務配置"
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 7 步：複製項目檔案到 D 槽
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "複製項目檔案到 D 槽..."

try {
    Write-Host "  複製中...（可能需要 2-5 分鐘）"
    Copy-Item -Path $SourceProject -Destination $DestProject -Recurse -Force

    Write-Success "項目檔案已複製到 $DestProject"
} catch {
    Write-Error "複製項目失敗：$_"
    exit 1
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 8 步：啟動 PostgreSQL 服務並驗證
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "啟動 PostgreSQL 服務..."

try {
    Start-Service -Name $PGService
    Start-Sleep -Seconds 5

    $service = Get-Service $PGService
    if ($service.Status -eq "Running") {
        Write-Success "PostgreSQL 服務已啟動"
    } else {
        Write-Error "PostgreSQL 服務啟動失敗"
        Write-Warning "檢查服務狀態：Get-Service $PGService"
        exit 1
    }
} catch {
    Write-Error "啟動服務失敗：$_"
    exit 1
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 第 9 步：測試數據庫連接
# ─────────────────────────────────────────────────────────────────────────────

Write-Info "測試數據庫連接..."

try {
    # 使用 Python 測試
    python << 'EOF'
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM documents'))
    doc_count = result.scalar()
    print(f"[✓] 數據庫連接成功，文件總數：{doc_count}")
EOF

    Write-Success "數據庫連接驗證通過"
} catch {
    Write-Error "數據庫連接失敗：$_"
    Write-Warning "請檢查 PostgreSQL 是否正常運行"
    Write-Warning "日誌位置：$PGDataDest\log\"
}

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────────────────────

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Success "遷移完成！"
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host ""
Write-Host "下一步："
Write-Host "  1. 進入新項目目錄：cd D:\LLM理財"
Write-Host "  2. 激活虛擬環境：.\venv\Scripts\Activate.ps1"
Write-Host "  3. 繼續執行 backfill：python scripts/backfill_embeddings.py --batch-size 50"
Write-Host ""

Write-Host "清理舊檔案（可選，遷移成功後）："
Write-Host "  Remove-Item -Recurse -Force '$SourceProject'"
Write-Host ""
