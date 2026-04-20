#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_migration.py

P0 Schema Migration 執行腳本（Windows 友善版本）
無需複雜的 PowerShell 語法，直接執行即可

用法：
    python run_migration.py
"""

from sqlalchemy import create_engine, text
import sys

def main():
    print("\n" + "="*70)
    print("  P0 Schema Migration 執行")
    print("="*70)

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
        print("\n檢查清單：")
        print("  - PostgreSQL 是否正在運行？")
        print("  - 密碼是否正確？(預設: postgres)")
        print("  - 資料庫是否存在？(預設: llm_stock)")
        return False

    # Step 2：讀取 SQL 檔案
    print("\n[2] 讀取 SQL 檔案...")
    try:
        with open(SQL_FILE, 'r', encoding='utf-8') as f:
            sql = f.read()
        print(f"[✓] 讀取成功 ({len(sql)} 字元)")
    except Exception as e:
        print(f"[✗] 讀取失敗：{e}")
        return False

    # Step 3：執行 Migration
    print("\n[3] 執行 Schema Migration...")
    try:
        # 特殊處理：分割 SQL 語句，正確處理 DO ... $$ ... $$ 塊
        statements = []
        current_stmt = ""
        in_do_block = False
        i = 0

        lines = sql.split('\n')
        while i < len(lines):
            line = lines[i]
            current_stmt += line + '\n'

            # 檢測 DO 塊開始
            if 'DO $$' in line:
                in_do_block = True
            # 檢測 DO 塊結束
            elif in_do_block and '$$;' in line:
                in_do_block = False
                # DO 塊完成，添加到語句列表
                statements.append(current_stmt.strip())
                current_stmt = ""
            # 檢測普通語句結束
            elif not in_do_block and line.rstrip().endswith(';'):
                stmt = current_stmt.strip()
                if stmt and not stmt.startswith('--'):
                    statements.append(stmt)
                current_stmt = ""

            i += 1

        # 添加任何剩餘的語句
        if current_stmt.strip():
            statements.append(current_stmt.strip())

        # 移除空語句和註釋
        statements = [s for s in statements if s and not s.startswith('--')]

        print(f"   找到 {len(statements)} 個 SQL 語句")

        with engine.begin() as conn:
            for i, stmt in enumerate(statements, 1):
                try:
                    conn.execute(text(stmt))
                    print(f"   ✓ 語句 {i}/{len(statements)}")
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'already exists' in error_msg or 'constraint' in error_msg:
                        print(f"   ! 語句 {i}/{len(statements)} (已存在，跳過)")
                    else:
                        print(f"   ✗ 語句 {i}/{len(statements)} 錯誤：{str(e)[:100]}")
                        raise

        print("[✓] Migration 執行完成")
    except Exception as e:
        print(f"[✗] Migration 失敗：{e}")
        return False

    # Step 4：驗證結果
    print("\n[4] 驗證結果...")
    try:
        checks = {
            'UNIQUE constraint (document_id, chunk_index)':
                "SELECT COUNT(*) FROM information_schema.table_constraints "
                "WHERE table_name='document_embeddings' AND constraint_type='UNIQUE' "
                "AND constraint_name='uq_document_embeddings_doc_chunk'",
            'Metadata 欄位 (ticker, published_at)':
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name='document_embeddings' AND column_name IN ('ticker', 'published_at')",
            'HNSW 索引':
                "SELECT COUNT(*) FROM pg_indexes "
                "WHERE tablename='document_embeddings' AND indexname='idx_document_embeddings_hnsw'"
        }

        with engine.connect() as conn:
            all_passed = True
            for check_name, query in checks.items():
                result = conn.execute(text(query))
                count = result.scalar()

                # 檢查 metadata 欄位需要恰好 2 個
                if check_name == 'Metadata 欄位 (ticker, published_at)':
                    passed = count == 2
                else:
                    passed = count > 0

                status = "✓" if passed else "✗"
                print(f"   {status} {check_name}")

                if not passed:
                    all_passed = False

        if not all_passed:
            print("\n[!] 某些驗證失敗，請查看上方詳細信息")
            return False

    except Exception as e:
        print(f"[✗] 驗證失敗：{e}")
        return False

    # 完成
    print("\n" + "="*70)
    print("  ✓ P0 Schema Migration 完成！")
    print("="*70)

    print("\n下一步：")
    print("  1. 執行 backfill: python scripts/backfill_embeddings.py --dry-run")
    print("  2. 如無異常，執行: python scripts/backfill_embeddings.py --batch-size 50")
    print("  3. 手工測試查詢效果")
    print()

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
