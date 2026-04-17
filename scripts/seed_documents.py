#!/usr/bin/env python
"""seed_documents.py

把現有結構化表（新聞 / 股價 / 財報 / 股利 / 月營收）批次轉成
documents 表記錄，再生成 document_embeddings 向量。

執行後系統才有向量可以做語意檢索（HybridRetrievalLayer）。

──────────────────────────────────────────────────────
用法
──────────────────────────────────────────────────────
# 看說明
python scripts/seed_documents.py --help

# 乾跑：列出各 ticker 會產生多少文件，不實際寫入
python scripts/seed_documents.py --dry-run

# 實際執行（全部 ticker）
python scripts/seed_documents.py

# 只跑特定 ticker
python scripts/seed_documents.py --tickers 2330 2317 2603

# 只做 documents upsert，不做 embedding（方便分段執行）
python scripts/seed_documents.py --skip-embedding

# 只對已有 document 但還沒 embedding 的補生成向量（搭配 backfill_embeddings.py）
python scripts/backfill_embeddings.py --all
──────────────────────────────────────────────────────
環境變數（.env）
──────────────────────────────────────────────────────
DATABASE_URL     必填（預設 postgresql+psycopg://postgres:postgres@localhost:5432/llm_stock）
OPENAI_API_KEY   skip-embedding 時可不填
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_documents")


# ─────────────────────────────────────────────────────────────────────────────
# 種子策略表
# 每個 SeedProfile 代表一種查詢視角，涵蓋不同資料類型
# ─────────────────────────────────────────────────────────────────────────────

class SeedProfile(NamedTuple):
    question_type: str
    topic_tags: list[str]
    time_range_days: int
    label: str          # 僅用於日誌顯示


SEED_PROFILES: list[SeedProfile] = [
    # ── 新聞（最重要，語意檢索主力）──
    SeedProfile("market_summary",             [],                  90,  "news-90d"),
    SeedProfile("announcement_summary",        [],                 365, "announce-1y"),

    # ── 財報 ──
    SeedProfile("earnings_summary",            [],                 730, "financials-2y"),
    SeedProfile("monthly_revenue_yoy_review",  ["月營收"],          365, "monthly-rev-1y"),

    # ── 股利 ──
    SeedProfile("dividend_yield_review",       ["股利", "殖利率"],  730, "dividends-2y"),

    # ── 股價 ──
    SeedProfile("price_range",                 ["股價區間"],        180, "price-6m"),
]


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_tickers_with_data(engine) -> list[tuple[str, str]]:
    """回傳 (ticker, stock_name) 列表，取「有實際資料」的 ticker。

    以 stock_news_articles + daily_price_bars + financial_statement_items
    三表的 UNION 取交集，過濾出真正有資料可 seed 的股票。
    """
    from sqlalchemy import text

    sql = text(
        """
        SELECT DISTINCT s.stock_id, s.stock_name
        FROM stock_info s
        WHERE s.stock_id IN (
            SELECT DISTINCT ticker FROM stock_news_articles
            UNION
            SELECT DISTINCT ticker FROM daily_price_bars
            UNION
            SELECT DISTINCT ticker FROM financial_statement_items
        )
        ORDER BY s.stock_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def count_existing_documents(engine, ticker: str) -> int:
    """查詢 documents 表中某 ticker 已有多少筆。"""
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM documents WHERE ticker = :ticker"),
            {"ticker": ticker},
        )
        return result.scalar() or 0


def count_existing_embeddings(engine, ticker: str) -> int:
    """查詢 document_embeddings 表中某 ticker 已有多少 chunk。"""
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM document_embeddings de
                JOIN documents d ON d.id = de.document_id
                WHERE d.ticker = :ticker
                """
            ),
            {"ticker": ticker},
        )
        return result.scalar() or 0


# ─────────────────────────────────────────────────────────────────────────────
# 核心：對單一 ticker 執行所有 SeedProfile
# ─────────────────────────────────────────────────────────────────────────────

def seed_ticker(
    ticker: str,
    company_name: str,
    gateway,
    document_repository,
    embedding_service,
    chunker,
    profiles: list[SeedProfile],
    dry_run: bool,
    skip_embedding: bool,
) -> dict:
    """對單一 ticker 跑所有 SeedProfile，回傳統計資訊。"""
    from llm_stock_system.core.models import StructuredQuery

    seen_urls: set[str] = set()
    all_docs = []

    for profile in profiles:
        query = StructuredQuery(
            user_query=f"{company_name or ticker} 資料種子查詢",
            ticker=ticker,
            company_name=company_name,
            question_type=profile.question_type,
            topic_tags=profile.topic_tags,
            time_range_days=profile.time_range_days,
        )
        try:
            docs = gateway.build_documents(query)
        except Exception as exc:
            logger.warning("  [%s] profile=%s 失敗：%s", ticker, profile.label, exc)
            continue

        new_docs = [d for d in docs if d.url not in seen_urls]
        seen_urls.update(d.url for d in new_docs)
        all_docs.extend(new_docs)

    if not all_docs:
        return {"ticker": ticker, "docs": 0, "chunks": 0}

    if dry_run:
        logger.info(
            "  [dry-run] %s：%d 篇文件（跨 %d 個 profile）",
            ticker, len(all_docs), len(profiles),
        )
        return {"ticker": ticker, "docs": len(all_docs), "chunks": 0}

    # Step 1: upsert documents
    upserted = document_repository.upsert_documents(all_docs)
    logger.info("  %s：upsert %d 篇文件", ticker, upserted)

    if skip_embedding or embedding_service is None:
        return {"ticker": ticker, "docs": upserted, "chunks": 0}

    # Step 2: embed（只 embed 剛 upsert 的，不重複）
    try:
        written = embedding_service.embed_and_store(all_docs, chunker)
        logger.info("  %s：生成 %d 個 chunk embedding", ticker, written)
    except Exception as exc:
        logger.warning("  %s embedding 失敗：%s", ticker, exc)
        written = 0

    return {"ticker": ticker, "docs": upserted, "chunks": written}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="把現有結構化表批次轉成 documents + document_embeddings。"
    )
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="只處理指定 ticker（空白分隔，如 2330 2317）；不指定則處理全部",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="列出各 ticker 預計產生的文件數，不實際寫入",
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="只做 documents upsert，跳過 embedding 生成",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="跳過已有 documents 的 ticker（加速重跑）",
    )
    parser.add_argument(
        "--model", default="text-embedding-3-small",
        help="OpenAI embedding 模型（預設 text-embedding-3-small）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="embedding batch size（預設 50）",
    )
    args = parser.parse_args()

    # ── 載入設定 ──
    from llm_stock_system.core.config import get_settings
    settings = get_settings()

    if not args.skip_embedding and not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY 未設定。請在 .env 中設定，"
            "或加上 --skip-embedding 只做 documents upsert。"
        )
        sys.exit(1)

    # ── 建立 DB engine ──
    from sqlalchemy import create_engine
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    # ── 建立 Gateway 與 Repository ──
    from llm_stock_system.adapters.postgres_market_data import (
        FinMindPostgresGateway,
        PostgresMarketDocumentRepository,
    )
    from llm_stock_system.adapters.finmind import FinMindClient
    from llm_stock_system.adapters.twse_financial import TwseCompanyFinancialClient

    finmind_client = FinMindClient(
        base_url=settings.finmind_base_url,
        api_token=settings.finmind_api_token,
    )
    twse_client = TwseCompanyFinancialClient(
        base_url=settings.twse_company_financial_url,
        monthly_revenue_url=settings.twse_monthly_revenue_url,
    )
    gateway = FinMindPostgresGateway(
        database_url=settings.database_url,
        finmind_client=finmind_client,
        twse_financial_client=twse_client,
        sync_on_query=False,   # seed 時不再觸發外部 sync，只用現有 DB 資料
    )
    document_repository = PostgresMarketDocumentRepository(gateway)

    # ── 建立 EmbeddingService ──
    embedding_service = None
    chunker = None
    if not args.skip_embedding:
        from llm_stock_system.services.embedding_service import EmbeddingService
        from llm_stock_system.services.document_chunker import DocumentChunker
        embedding_service = EmbeddingService(
            openai_api_key=settings.openai_api_key,
            model=args.model,
            batch_size=args.batch_size,
            db_engine=engine,
        )
        chunker = DocumentChunker()

    # ── 取得 ticker 清單 ──
    if args.tickers:
        # 從 stock_info 補 company_name
        from sqlalchemy import text
        ticker_map: dict[str, str] = {}
        with engine.connect() as conn:
            for tk in args.tickers:
                row = conn.execute(
                    text("SELECT stock_name FROM stock_info WHERE stock_id = :id"),
                    {"id": tk},
                ).fetchone()
                ticker_map[tk] = row[0] if row else tk
        all_tickers = list(ticker_map.items())
    else:
        logger.info("查詢有資料的 ticker 清單…")
        all_tickers = get_tickers_with_data(engine)

    logger.info("共 %d 個 ticker 待處理", len(all_tickers))

    if args.skip_existing and not args.dry_run:
        before = len(all_tickers)
        all_tickers = [
            (tk, name) for tk, name in all_tickers
            if count_existing_documents(engine, tk) == 0
        ]
        logger.info("--skip-existing：跳過 %d 個已有 documents 的 ticker，剩 %d 個",
                    before - len(all_tickers), len(all_tickers))

    if not all_tickers:
        logger.info("無需處理，結束。")
        return

    # ── 逐 ticker 執行 ──
    total_docs = 0
    total_chunks = 0
    t_start = time.perf_counter()

    for idx, (ticker, company_name) in enumerate(all_tickers, 1):
        logger.info("[%d/%d] 處理 %s（%s）…", idx, len(all_tickers), ticker, company_name)
        stats = seed_ticker(
            ticker=ticker,
            company_name=company_name,
            gateway=gateway,
            document_repository=document_repository,
            embedding_service=embedding_service,
            chunker=chunker,
            profiles=SEED_PROFILES,
            dry_run=args.dry_run,
            skip_embedding=args.skip_embedding,
        )
        total_docs += stats["docs"]
        total_chunks += stats["chunks"]

    elapsed = time.perf_counter() - t_start
    logger.info(
        "═══ 完成 ═══  ticker=%d  documents=%d  chunks=%d  耗時=%.1fs",
        len(all_tickers), total_docs, total_chunks, elapsed,
    )

    if args.dry_run:
        logger.info("（dry-run 模式：以上皆未實際寫入）")
    elif not args.skip_embedding and total_docs > 0 and total_chunks == 0:
        logger.warning(
            "documents 已寫入但 embedding 為 0。"
            "可能原因：OPENAI_API_KEY 無效或網路問題。"
            "補跑：python scripts/backfill_embeddings.py --all"
        )


if __name__ == "__main__":
    main()
