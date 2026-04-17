# Windows 虛擬環境設置完整指南

**問題：** 缺少虛擬環境或必要的 Python 依賴  
**解決方案：** 建立並設置虛擬環境

---

## 📋 完整設置步驟（5-10 分鐘）

### Step 1：進入專案目錄

```powershell
# 進入您的 LLM理財 目錄
cd C:\path\to\LLM理財

# 確認目錄結構
dir
# 應該看到：db/, src/, scripts/ 等資料夾
```

---

### Step 2：建立虛擬環境

```powershell
# 建立名為 venv 的虛擬環境
python -m venv venv

# 如果出現錯誤，先檢查 Python 是否已安裝
python --version
# 應該顯示 Python 3.10+
```

---

### Step 3：激活虛擬環境

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# 如果出現執行策略錯誤，執行以下命令允許運行腳本
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然後重新執行激活命令
.\venv\Scripts\Activate.ps1
```

**成功的標誌：** PowerShell 提示字首會顯示 `(venv)`

```
(venv) C:\path\to\LLM理財>
```

---

### Step 4：升級 pip 和安裝基本工具

```powershell
# 確保在虛擬環境中（看到 (venv) 前綴）

# 升級 pip
python -m pip install --upgrade pip

# 安裝 wheel（用於建立二進制檔案）
pip install wheel
```

---

### Step 5：安裝必要的依賴

```powershell
# 檢查是否有 requirements.txt
dir requirements*.txt

# 如果存在 requirements.txt
pip install -r requirements.txt

# 或手動安裝核心依賴
pip install sqlalchemy psycopg[binary]

# 驗證安裝成功
python -c "import sqlalchemy; import psycopg; print('[✓] 所有依賴已安裝')"
```

---

### Step 6：驗證虛擬環境

```powershell
# 檢查已安裝的包
pip list

# 應該看到（至少）：
# sqlalchemy
# psycopg
# ...
```

---

## 🚀 虛擬環境準備完成，現在執行 Migration

```powershell
# 確認虛擬環境已激活（看到 (venv) 前綴）

# 執行 migration
python << 'EOF'
from sqlalchemy import create_engine, text

print("[1] 連接資料庫...")
engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')

print("[2] 讀取 SQL 檔案...")
with open('db/sql/004_embedding_enhancements.sql', 'r', encoding='utf-8') as f:
    sql = f.read()

print("[3] 執行 Migration...")
with engine.begin() as conn:
    statements = [s.strip() for s in sql.split(';') if s.strip() and not s.startswith('--')]
    for i, stmt in enumerate(statements, 1):
        try:
            conn.execute(text(stmt))
            print(f"   ✓ 語句 {i}/{len(statements)} 完成")
        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f"   ! 語句 {i}/{len(statements)} (已存在，跳過)")
            else:
                print(f"   ✗ 語句 {i}/{len(statements)} 錯誤：{e}")

print("\n[✓] Migration 完成！")
EOF
```

---

## ⚠️ 常見虛擬環境問題

| 問題 | 解決方案 |
|---|---|
| `(venv)` 前綴未出現 | 重新執行 `.\venv\Scripts\Activate.ps1` |
| 執行策略錯誤 | 執行 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `No module named 'sqlalchemy'` | 執行 `pip install sqlalchemy psycopg[binary]` |
| `pip: command not found` | 虛擬環境未激活，重新執行 `Activate.ps1` |
| `Python 3.8 或更舊版本` | 升級 Python 至 3.10+ |

---

## 📝 虛擬環境管理命令

```powershell
# 激活虛擬環境
.\venv\Scripts\Activate.ps1

# 停用虛擬環境（返回全域 Python）
deactivate

# 刪除虛擬環境（如需重建）
Remove-Item -Recurse -Force venv

# 檢查虛擬環境中的 Python 版本
python --version

# 列出已安裝的包
pip list

# 安裝特定包
pip install package_name

# 卸載包
pip uninstall package_name
```

---

## 🔧 從頭完整設置（如前面的步驟有問題）

```powershell
# 1. 進入專案目錄
cd C:\path\to\LLM理財

# 2. 刪除舊的虛擬環境（如有）
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue

# 3. 建立新虛擬環境
python -m venv venv

# 4. 允許執行腳本
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 5. 激活虛擬環境
.\venv\Scripts\Activate.ps1

# 6. 升級 pip
python -m pip install --upgrade pip

# 7. 安裝依賴
pip install sqlalchemy psycopg[binary]

# 8. 驗證
python -c "import sqlalchemy, psycopg; print('[✓] 準備完成')"

# 9. 現在可以執行 migration 了
python << 'EOF'
from sqlalchemy import create_engine, text
# ... (上面的 migration 代碼) ...
EOF
```

---

## ✅ 檢查清單

- [ ] 虛擬環境已建立 (`venv` 資料夾存在)
- [ ] 虛擬環境已激活 (看到 `(venv)` 前綴)
- [ ] pip 已升級
- [ ] sqlalchemy 已安裝 → `pip install sqlalchemy`
- [ ] psycopg 已安裝 → `pip install psycopg[binary]`
- [ ] PostgreSQL 正在運行
- [ ] 準備好執行 migration

---

## 🚀 準備完成！

虛擬環境設置完畢後，執行 migration：

```powershell
# 確保虛擬環境已激活（看到 (venv) 前綴）

# 簡單版 - 直接執行
python << 'EOF'
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')

with open('db/sql/004_embedding_enhancements.sql', 'r', encoding='utf-8') as f:
    sql = f.read()

with engine.begin() as conn:
    for stmt in [s.strip() for s in sql.split(';') if s.strip() and not s.startswith('--')]:
        try:
            conn.execute(text(stmt))
        except:
            pass

print("[✓] Migration 完成！")
EOF
```

---

## 📞 仍有問題？

1. **確認虛擬環境激活**
   ```powershell
   which python
   # 應該顯示 venv 路徑
   ```

2. **驗證依賴安裝**
   ```powershell
   pip list | grep -E "sqlalchemy|psycopg"
   ```

3. **測試資料庫連接**
   ```powershell
   python << 'EOF'
   from sqlalchemy import create_engine, text
   engine = create_engine('postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock')
   with engine.connect() as conn:
       result = conn.execute(text('SELECT 1'))
       print("[✓] 資料庫連接正常")
   EOF
   ```

4. **檢查 PostgreSQL 是否運行**
   ```powershell
   netstat -an | findstr 5432
   # 應該看到 LISTENING
   ```

---

**虛擬環境準備完成！現在可以安全地執行 migration 了！** 🚀
