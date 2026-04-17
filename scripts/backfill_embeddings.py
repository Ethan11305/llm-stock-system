#!/usr/bin/env python
"""backfill_embeddings.py

一次性回填腳本：為 documents 表中已存在但尚未 embed 的文件生成向量。

用法：
    python scripts/backfill_embeddings.py --help
    python scripts/backfill_embeddings.py --dry-run
    python scripts/backfill_embeddings.py --batch-size 100
    python scripts/backfill_embeddings.py --batch-size 50 --ticker 2330
    python scripts/backfill_embeddings.py --all   # 反覆執行直到所有文件都回填完畢

環境變數（或 .env 檔案）：
    OPENAI_API_KEY   - 必填
    DATABASE_URL     - 選填，預設 postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 確保專案 src 在 sys.path 中（不論從哪個目錄執行）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_embeddings")


# ─────────────────────────────────────────────────────────────────────────────
# 核心邏輯
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_document_dict(row) -> dict:
    """將 DB row 轉成可供 Document(**d) 使用的 dict。"""
    return {
        "id": str(row[0]),
        "ticker": row[1],
        "title": row[2],
        "content": row[3] or "",       # content_clean
        "source_name": row[4],
        "source_type": row[5],
        "source_tier": row[6],
        "url": row[7],
        "published_at": row[8],
        "author": None,
        "topics": [],
        "is_valid": True,
    }


def fetch_unembedded_documents(conn, batch_size: int, ticker: str | None = None):
    """從 documents 取出尚未有 embedding 的文件（一批）。"""
    from sqlalchemy import text

    where_extra = "AND d.ticker = :ticker" if ticker else ""
    sql = text(
        f"""
        SELECT
            d.id,
            d.ticker,
            d.title,
            d.content_clean,
            d.source_name,
            d.source_type,
            d.source_tier,
            d.url,
            d.published_at
        FROM documents d
        LEFT JOIN document_embeddings de ON de.document_id = d.id
        WHERE de.id IS NULL
          AND d.is_valid = TRUE
          {where_extra}
        ORDER BY d.published_at DESC
        LIMIT :batch_size
        """
    )
    params: dict = {"batch_size": batch_size}
    if ticker:
        params["ticker"] = ticker
    return conn.execute(sql, params).fetchall()


def count_unembedded(conn, ticker: str | None = None) -> int:
    """計算尚未 embed 的文件數量。"""
    from sqlalchemy import text

    where_extra = "AND d.ticker = :ticker" if ticker else ""
    sql = text(
        f"""
        SELECT COUNT(*)
        FROM documents d
        LEFT JOIN document_embeddings de ON de.document_id = d.id
        WHERE de.id IS NULL
          AND d.is_valid = TRUE
          {where_extra}
        """
    )
    params: dict = {}
    if ticker:
        params["ticker"] = ticker
    result = conn.execute(sql, params)
    return result.scalar() or 0


def backfill(
    db_engine,
    embedding_service,
    chunker,
    batch_size: int = 100,
    ticker: str | None = None,
    dry_run: bool = False,
) -> int:
    """回填一批文件的 embedding。回傳本次寫入的 chunk 數。"""
    from llm_stock_system.core.models import Document

    with db_engine.connect() as conn:
        rows = fetch_unembedded_documents(conn, batch_size, ticker)

    if not rows:
        logger.info("✅ 所有文件都已有 embedding，無需回填。")
        return 0

    logger.info("本批次取得 %d 篇待回填文件。", len(rows))

    if dry_run:
        for row in rows:
            logger.info("  [dry-run] %s | %s | %s", row[1], row[2][:40], row[8])
        return 0

    documents = [Document(**_row_to_document_dict(row)) for row in rows]
    written = embedding_service.embed_and_store(documents, chunker)
    logger.info("✅ 本批次回填 %d 個 chunk（來自 %d 篇文件）。", written, len(documents))
    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI 進入點
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="回填 document_embeddings 表：為尚未 embed 的文件生成向量。"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="每批處理的文件數（預設 100）",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="只回填指定股票代碼（如 2330）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="列出待回填文件但不實際寫入",
    )
    parser.add_argument(
        "--all", dest="run_all", action="store_true",
        help="反覆執行直到所有文件都回填完畢",
    )
    parser.add_argument(
        "--model", type=str, default="text-embedding-3-small",
        help="OpenAI embedding 模型（預設 text-embedding-3-small）",
    )
    args = parser.parse_args()

    # ── 載入設定 ──
    from llm_stock_system.core.config import get_settings
    settings = get_settings()

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY 未設定，無法生成 embedding。請在 .env 中設定。")
        sys.exit(1)

    # ── 建立 DB engine ──
    from sqlalchemy import create_engine
    engine = create_engine(settings.database_url)

    # ── 建立 EmbeddingService + Chunker ──
    from llm_stock_system.services.embedding_service import EmbeddingService
    from llm_stock_system.services.document_chunker import DocumentChunker

    embedding_service = EmbeddingService(
        openai_api_key=settings.openai_api_key,
        model=args.model,
        batch_size=50,  # OpenAI 單次 batch 建議不超過 50-100
        db_engine=engine,
    )
    chunker = DocumentChunker()

    # ── 顯示待回填數量 ──
    with engine.connect() as conn:
        total = count_unembedded(conn, args.ticker)
    logger.info("待回填文件總數：%d（ticker=%s）", total, args.ticker or "全部")

    if total == 0:
        logger.info("✅ 無需回填，結束。")
        return

    # ── 執行回填 ──
    total_written = 0
    if args.run_all:
        # 反覆執行直到沒有待回填文件
        iteration = 0
        while True:
            iteration += 1
            logger.info("── 第 %d 輪 ──", iteration)
            written = backfill(engine, embedding_service, chunker, args.batch_size, args.ticker, args.dry_run)
            total_written += written
            if written == 0:
                break
    else:
        total_written = backfill(engine, embedding_service, chunker, args.batch_size, args.ticker, args.dry_run)

    logger.info("回填完成，共寫入 %d 個 chunk embedding。", total_written)


if __name__ == "__main__":
    main()
