# Windows 項目 + PostgreSQL 遷移至 D 槽完整指南

**目的：** 將整個 LLM理財 項目和 PostgreSQL 數據庫從 C 槽遷移到 D 槽  
**預期耗時：** 30-60 分鐘  
**風險等級：** 中（涉及系統服務）

---

## ⚠️ 準備工作（必讀）

### 1️⃣ 備份現有數據

```powershell
# 備份 PostgreSQL 數據庫
pg_dump -U postgres -d llm_stock > C:\backup_llm_stock_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql

# 或使用 Python 備份
python << 'EOF'
from sqlalchemy import create_engine
import shutil
from datetime import datetime

# 備份到 C:\
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_file = f'C:\backup_llm_stock_{timestamp}.sql'

print(f"[備份中...請稍候...]")
# 實際備份代碼需要 pg_dump
import subprocess
subprocess.run(f'pg_dump -U postgres -d llm_stock > {backup_file}', shell=True)
print(f"[✓] 備份完成：{backup_file}")
EOF
```

### 2️⃣ 確認 D 槽有足夠空間

```powershell
# 檢查磁碟空間
Get-Volume -DriveLetter D | Select-Object SizeRemaining

# 應該至少有 10 GB 可用空間（項目約 500MB + 數據庫變數）
```

---

## 📋 遷移步驟

### Step 1：停止 PostgreSQL 服務

```powershell
# 以管理員身份執行 PowerShell

# 停止 PostgreSQL
net stop postgresql-x64-15
# 或（如果版本不同）
net stop postgresql-x64-14

# 驗證已停止
Get-Service postgresql-x64-* | Select-Object Name, Status
```

如果出現「找不到服務」，使用以下方式查找正確的服務名稱：

```powershell
# 查找 PostgreSQL 服務
Get-Service | Where-Object {$_.Name -like "*postgres*"}

# 記下準確的服務名稱（例如 postgresql-x64-15）
```

---

### Step 2：查找 PostgreSQL 數據目錄

**典型位置：**
```
C:\Program Files\PostgreSQL\15\data
C:\Program Files\PostgreSQL\14\data
```

確認位置：
```powershell
# 查看 PostgreSQL 配置
notepad "C:\Program Files\PostgreSQL\15\data\postgresql.conf"

# 查找 data_directory 參數
# 預設通常就在 PostgreSQL 安裝目錄的 \data 子資料夾
```

---

### Step 3：複製 PostgreSQL 數據目錄到 D 槽

```powershell
# 建立 D:\PostgreSQL 目錄結構
New-Item -ItemType Directory -Path "D:\PostgreSQL\15\data" -Force

# 複製 PostgreSQL 數據目錄
# 注意：這可能需要 10-30 分鐘，取決於數據庫大小
Copy-Item -Path "C:\Program Files\PostgreSQL\15\data\*" `
          -Destination "D:\PostgreSQL\15\data" `
          -Recurse -Force

# 驗證複製成功
Get-ChildItem "D:\PostgreSQL\15\data" | Measure-Object

# 應該看到多個 base, global, pg_* 目錄
```

---

### Step 4：更新 PostgreSQL 配置

**編輯 postgresql.conf：**

```powershell
# 編輯新位置的配置檔案
notepad "D:\PostgreSQL\15\data\postgresql.conf"

# 找到這一行並修改：
# data_directory = 'D:/PostgreSQL/15/data'

# 保存檔案
```

**編輯 Windows 服務的啟動配置：**

```powershell
# 以管理員身份打開「服務」應用程式
services.msc

# 或用 PowerShell：
$svc = Get-Service postgresql-x64-15
$svc | Stop-Service  # 確保已停止

# 修改服務的啟動參數
# 編輯註冊表（建議用圖形界面 regedit）
# HKLM\SYSTEM\CurrentControlSet\Services\postgresql-x64-15
# 修改 ImagePath 參數的 -D 參數指向 D:\PostgreSQL\15\data
```

**或使用 sc.exe（推薦）：**

```powershell
# 以管理員身份執行

# 查看現有配置
sc qc postgresql-x64-15

# 修改服務配置（如果 -D 參數指向 C:\...）
# 找到類似的命令：
# "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" runservice -N "postgresql-x64-15" -D "C:\Program Files\PostgreSQL\15\data" ...

# 修改為：
# "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" runservice -N "postgresql-x64-15" -D "D:\PostgreSQL\15\data" ...

# 使用此命令修改：
sc config postgresql-x64-15 binPath= "\"C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe\" runservice -N \"postgresql-x64-15\" -D \"D:\PostgreSQL\15\data\" -p \"C:\Program Files\PostgreSQL\15\bin\postgres.exe\""
```

---

### Step 5：調整資料夾權限

PostgreSQL 需要特定權限。複製後需要重新設置：

```powershell
# 以管理員身份執行

# 設置資料夾所有者為 SYSTEM（或當前用戶）
icacls "D:\PostgreSQL" /grant:r "SYSTEM:(F)" /T /C

# 或
icacls "D:\PostgreSQL" /grant:r "$env:USERNAME:(F)" /T /C

# 移除其他用戶的訪問權限（安全考量）
icacls "D:\PostgreSQL" /inheritance:r /T /C
icacls "D:\PostgreSQL" /grant:r "SYSTEM:(F)" /T /C
```

---

### Step 6：啟動 PostgreSQL 服務

```powershell
# 以管理員身份執行

# 啟動服務
net start postgresql-x64-15

# 驗證狀態
Get-Service postgresql-x64-15 | Select-Object Name, Status

# 應該看到 Status: Running
```

---

### Step 7：驗證 PostgreSQL 正常運行

```powershell
# 測試連接
psql -U postgres -d postgres -c "SELECT version();"

# 或用 Python 驗證
python << 'EOF'
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')
with engine.connect() as conn:
    result = conn.execute(text('SELECT current_timestamp'))
    print(f"[✓] PostgreSQL 連接正常: {result.scalar()}")
EOF
```

如果連接失敗，檢查：
- PostgreSQL 服務是否已啟動 → `net start postgresql-x64-15`
- 防火牆是否阻止 5432 端口
- 新數據目錄的權限設置

---

### Step 8：遷移項目檔案到 D 槽

```powershell
# 複製整個項目
Copy-Item -Path "C:\Users\ASUS\Downloads\LLM理財" `
          -Destination "D:\LLM理財" `
          -Recurse -Force

# 驗證複製成功
Get-ChildItem "D:\LLM理財"

# 應該看到 src/, db/, scripts/, venv/ 等目錄
```

---

### Step 9：更新虛擬環境（如需要）

如果虛擬環境路徑已改變：

```powershell
# 進入新項目目錄
cd D:\LLM理財

# 如果 venv 已複製，直接激活
.\venv\Scripts\Activate.ps1

# 如果需要重建 venv
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install sqlalchemy psycopg[binary] httpx openai
```

---

### Step 10：驗證所有連接

```powershell
# 進入 D:\LLM理財 目錄
cd D:\LLM理財

# 激活虛擬環境
.\venv\Scripts\Activate.ps1

# 測試完整流程
python << 'EOF'
from sqlalchemy import create_engine, text

# 1. 測試資料庫連接
engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM documents'))
    doc_count = result.scalar()
    print(f"[✓] 數據庫連接成功，文件總數：{doc_count}")

# 2. 測試項目檔案
import os
if os.path.exists('db/sql/004_embedding_enhancements.sql'):
    print("[✓] 項目檔案完整")
else:
    print("[✗] 項目檔案缺失")
EOF
```

---

## ❌ 如果出現問題

| 問題 | 解決方案 |
|---|---|
| PostgreSQL 服務無法啟動 | 檢查權限設置、檢查 PostgreSQL 日誌 (D:\PostgreSQL\15\data\log\) |
| 連接被拒絕 | 確認 postgresql.conf 的 listen_addresses 設置、檢查防火牆 |
| 數據目錄損壞 | 使用備份恢復、或重新初始化：`initdb -D D:\PostgreSQL\15\data` |
| 權限問題 | 重新執行 icacls 命令設置權限 |

---

## 🗑️ 清理舊位置（遷移成功後）

```powershell
# 確認新位置正常運行後，才刪除舊位置

# 刪除 C:\Users\ASUS\Downloads\LLM理財（如已備份）
Remove-Item -Recurse -Force "C:\Users\ASUS\Downloads\LLM理財"

# 刪除 C:\Program Files\PostgreSQL\15\data（如確認備份無誤）
# 注意：只刪除 data 目錄，保留 PostgreSQL 應用程式
Remove-Item -Recurse -Force "C:\Program Files\PostgreSQL\15\data"

# 驗證磁碟空間已釋放
Get-Volume -DriveLetter C | Select-Object SizeRemaining
```

---

## ✅ 檢查清單

遷移完成後確認：

- [ ] PostgreSQL 服務已啟動（net start postgresql-x64-15）
- [ ] 數據庫可連接（psql -U postgres -d llm_stock -c "SELECT 1"）
- [ ] 項目檔案已複製到 D:\LLM理財
- [ ] 虛擬環境可激活（.\venv\Scripts\Activate.ps1）
- [ ] 依賴已安裝（pip list）
- [ ] 測試查詢成功（python run_migration.py）
- [ ] 舊檔案已刪除（清理 C 槽空間）

---

## 🚀 遷移後立即執行

```powershell
# 進入新位置
cd D:\LLM理財

# 激活虛擬環境
.\venv\Scripts\Activate.ps1

# 繼續執行 backfill（如之前未完成）
python scripts/backfill_embeddings.py --batch-size 50

# 執行 migration 驗證
python run_migration.py
```

---

**遷移過程中有問題，隨時告訴我！** 🚀
