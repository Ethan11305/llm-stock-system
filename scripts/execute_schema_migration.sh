#!/bin/bash
#
# execute_schema_migration.sh
#
# P0 Schema Migration 一鍵執行腳本
# 用途：執行 004_embedding_enhancements.sql 並驗證結果
#
# 用法：
#   bash scripts/execute_schema_migration.sh              # 完整流程
#   bash scripts/execute_schema_migration.sh --backup-only # 僅備份
#   bash scripts/execute_schema_migration.sh --verify-only # 僅驗證
#

set -e  # 任何命令失敗就停止

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-llm_stock}"
BACKUP_DIR="./db/backups"
SQL_FILE="./db/sql/004_embedding_enhancements.sql"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 顏色碼
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# 函數
# ─────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# 檢查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 '$1' 未找到，請先安裝"
        exit 1
    fi
}

# 測試 PostgreSQL 連接
test_postgres_connection() {
    log_info "測試 PostgreSQL 連接..."
    if PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" &> /dev/null; then
        log_success "PostgreSQL 連接成功"
        return 0
    else
        log_error "無法連接到 PostgreSQL"
        log_warning "檢查以下設置："
        echo "  DB_USER:   $DB_USER"
        echo "  DB_HOST:   $DB_HOST"
        echo "  DB_PORT:   $DB_PORT"
        echo "  DB_NAME:   $DB_NAME"
        return 1
    fi
}

# 備份資料庫
backup_database() {
    log_info "備份資料庫..."
    mkdir -p "$BACKUP_DIR"

    BACKUP_FILE="$BACKUP_DIR/llm_stock_${TIMESTAMP}.sql"

    if PGPASSWORD=$DB_PASSWORD pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" > "$BACKUP_FILE"; then
        log_success "資料庫已備份：$BACKUP_FILE"
        return 0
    else
        log_error "資料庫備份失敗"
        return 1
    fi
}

# 執行 Schema Migration
execute_migration() {
    log_info "執行 Schema Migration..."

    if [ ! -f "$SQL_FILE" ]; then
        log_error "SQL 檔案不存在：$SQL_FILE"
        return 1
    fi

    MIGRATION_LOG="/tmp/migration_${TIMESTAMP}.log"

    if PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -a -f "$SQL_FILE" > "$MIGRATION_LOG" 2>&1; then
        log_success "Schema Migration 執行成功"
        log_info "詳細日誌：$MIGRATION_LOG"
        return 0
    else
        log_error "Schema Migration 執行失敗"
        log_warning "錯誤日誌內容："
        cat "$MIGRATION_LOG"
        return 1
    fi
}

# 驗證 Migration 結果
verify_migration() {
    log_info "驗證 Migration 結果..."

    # 檢查 UNIQUE constraint
    CONSTRAINT_CHECK=$(PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM information_schema.table_constraints
         WHERE table_name = 'document_embeddings'
         AND constraint_type = 'UNIQUE'
         AND constraint_name = 'uq_document_embeddings_doc_chunk';" 2>/dev/null || echo "0")

    if [ "$CONSTRAINT_CHECK" == "1" ]; then
        log_success "UNIQUE constraint 已建立"
    else
        log_error "UNIQUE constraint 未建立"
        return 1
    fi

    # 檢查 Metadata 欄位
    COLUMN_CHECK=$(PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM information_schema.columns
         WHERE table_name = 'document_embeddings'
         AND column_name IN ('ticker', 'published_at');" 2>/dev/null || echo "0")

    if [ "$COLUMN_CHECK" == "2" ]; then
        log_success "Metadata 欄位已新增（ticker, published_at）"
    else
        log_error "Metadata 欄位未完整新增"
        return 1
    fi

    # 檢查 HNSW 索引
    INDEX_CHECK=$(PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM pg_indexes
         WHERE tablename = 'document_embeddings'
         AND indexname = 'idx_document_embeddings_hnsw';" 2>/dev/null || echo "0")

    if [ "$INDEX_CHECK" == "1" ]; then
        log_success "HNSW 索引已建立"
    else
        log_error "HNSW 索引未建立"
        return 1
    fi

    # 顯示完整的表結構
    log_info "當前 document_embeddings 表結構："
    PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "\d document_embeddings"

    return 0
}

# 主流程
main() {
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  P0 Schema Migration 執行腳本${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo

    # 檢查依賴
    log_info "檢查依賴..."
    check_command "psql"
    check_command "pg_dump"
    log_success "所有依賴已齊備"
    echo

    # 測試連接
    if ! test_postgres_connection; then
        exit 1
    fi
    echo

    # 解析命令行參數
    case "${1:-}" in
        --backup-only)
            backup_database
            exit $?
            ;;
        --verify-only)
            verify_migration
            exit $?
            ;;
        --help)
            echo "用法："
            echo "  bash $0              # 完整流程（備份 + 遷移 + 驗證）"
            echo "  bash $0 --backup-only # 僅備份"
            echo "  bash $0 --verify-only # 僅驗證"
            echo "  bash $0 --help        # 顯示此幫助"
            exit 0
            ;;
    esac

    # 完整流程：備份 → 遷移 → 驗證

    # 1. 備份
    if ! backup_database; then
        log_warning "備份失敗，但繼續執行遷移（風險自負）"
    fi
    echo

    # 2. 執行遷移
    if ! execute_migration; then
        log_error "Schema Migration 失敗，請檢查上述日誌"
        exit 1
    fi
    echo

    # 3. 驗證
    if ! verify_migration; then
        log_error "驗證失敗"
        exit 1
    fi
    echo

    # 完成
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
    log_success "P0 Schema Migration 已完成"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
    echo
    log_info "下一步："
    echo "  1. 執行 backfill: python scripts/backfill_embeddings.py --dry-run"
    echo "  2. 手工測試查詢效果"
    echo "  3. 監控系統運行"
    echo
}

# 執行
main "$@"
