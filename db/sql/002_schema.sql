CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    title TEXT NOT NULL,
    summary_raw TEXT,
    content_raw TEXT NOT NULL,
    content_clean TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    author TEXT,
    published_at TIMESTAMP NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    topic_tags TEXT[] NOT NULL DEFAULT '{}',
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    is_superseded BOOLEAN NOT NULL DEFAULT FALSE,
    dedupe_group_id UUID,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_ticker_published_at
    ON documents (ticker, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_source_tier_published_at
    ON documents (source_tier, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_topic_tags
    ON documents USING GIN (topic_tags);

CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_ivfflat
    ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY,
    user_query TEXT NOT NULL,
    ticker VARCHAR(16),
    topic TEXT NOT NULL,
    time_range TEXT NOT NULL,
    retrieved_doc_count INT NOT NULL,
    confidence_score NUMERIC(5,4) NOT NULL,
    validation_status TEXT NOT NULL,
    response_text TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_sources (
    id UUID PRIMARY KEY,
    query_log_id UUID NOT NULL REFERENCES query_logs(id) ON DELETE CASCADE,
    document_id UUID NOT NULL,
    title TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    excerpt TEXT NOT NULL,
    support_score NUMERIC(6,4) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_info (
    stock_id VARCHAR(64) PRIMARY KEY,
    stock_name TEXT NOT NULL,
    industry_category TEXT,
    market_type TEXT,
    reference_date DATE,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stock_info_stock_name
    ON stock_info (stock_name);

CREATE TABLE IF NOT EXISTS daily_price_bars (
    ticker VARCHAR(64) NOT NULL,
    trading_date DATE NOT NULL,
    open_price NUMERIC(12,4) NOT NULL,
    high_price NUMERIC(12,4) NOT NULL,
    low_price NUMERIC(12,4) NOT NULL,
    close_price NUMERIC(12,4) NOT NULL,
    trading_volume BIGINT,
    trading_money BIGINT,
    spread NUMERIC(12,4),
    turnover BIGINT,
    source_name TEXT NOT NULL DEFAULT 'FinMind TaiwanStockPrice',
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, trading_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_price_bars_ticker_date
    ON daily_price_bars (ticker, trading_date DESC);

CREATE TABLE IF NOT EXISTS financial_statement_items (
    ticker VARCHAR(64) NOT NULL,
    statement_date DATE NOT NULL,
    item_type TEXT NOT NULL,
    value NUMERIC(20,6) NOT NULL,
    origin_name TEXT NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, statement_date, item_type)
);

CREATE INDEX IF NOT EXISTS idx_financial_statement_items_lookup
    ON financial_statement_items (ticker, statement_date DESC, item_type);

CREATE TABLE IF NOT EXISTS balance_sheet_items (
    ticker VARCHAR(64) NOT NULL,
    statement_date DATE NOT NULL,
    item_type TEXT NOT NULL,
    value NUMERIC(20,6) NOT NULL,
    origin_name TEXT NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, statement_date, item_type)
);

CREATE INDEX IF NOT EXISTS idx_balance_sheet_items_lookup
    ON balance_sheet_items (ticker, statement_date DESC, item_type);

CREATE TABLE IF NOT EXISTS cash_flow_statement_items (
    ticker VARCHAR(64) NOT NULL,
    statement_date DATE NOT NULL,
    item_type TEXT NOT NULL,
    value NUMERIC(20,6) NOT NULL,
    origin_name TEXT NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, statement_date, item_type)
);

CREATE INDEX IF NOT EXISTS idx_cash_flow_statement_items_lookup
    ON cash_flow_statement_items (ticker, statement_date DESC, item_type);

CREATE TABLE IF NOT EXISTS monthly_revenue_points (
    ticker VARCHAR(64) NOT NULL,
    revenue_month DATE NOT NULL,
    revenue NUMERIC(20,6) NOT NULL,
    prior_year_month_revenue NUMERIC(20,6),
    month_over_month_pct NUMERIC(20,6),
    year_over_year_pct NUMERIC(20,6),
    cumulative_revenue NUMERIC(20,6),
    prior_year_cumulative_revenue NUMERIC(20,6),
    cumulative_yoy_pct NUMERIC(20,6),
    report_date DATE,
    notes TEXT,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, revenue_month)
);

CREATE INDEX IF NOT EXISTS idx_monthly_revenue_points_lookup
    ON monthly_revenue_points (ticker, revenue_month DESC);

CREATE TABLE IF NOT EXISTS pe_valuation_points (
    ticker VARCHAR(64) NOT NULL,
    valuation_month DATE NOT NULL,
    pe_ratio NUMERIC(20,6),
    peer_pe_ratio NUMERIC(20,6),
    pb_ratio NUMERIC(20,6),
    peer_pb_ratio NUMERIC(20,6),
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, valuation_month)
);

CREATE INDEX IF NOT EXISTS idx_pe_valuation_points_lookup
    ON pe_valuation_points (ticker, valuation_month DESC);

CREATE TABLE IF NOT EXISTS dividend_policies (
    ticker VARCHAR(64) NOT NULL,
    base_date DATE NOT NULL,
    year_label TEXT NOT NULL,
    cash_earnings_distribution NUMERIC(20,6),
    cash_statutory_surplus NUMERIC(20,6),
    stock_earnings_distribution NUMERIC(20,6),
    stock_statutory_surplus NUMERIC(20,6),
    participate_distribution_of_total_shares NUMERIC(20,6),
    announcement_date DATE,
    announcement_time TEXT,
    cash_ex_dividend_trading_date DATE,
    cash_dividend_payment_date DATE,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, base_date, year_label)
);

CREATE INDEX IF NOT EXISTS idx_dividend_policies_lookup
    ON dividend_policies (ticker, base_date DESC);

ALTER TABLE dividend_policies
    ADD COLUMN IF NOT EXISTS participate_distribution_of_total_shares NUMERIC(20,6);

CREATE TABLE IF NOT EXISTS stock_news_articles (
    ticker VARCHAR(64) NOT NULL,
    published_at TIMESTAMP NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    source_name TEXT NOT NULL,
    url TEXT NOT NULL,
    source_tier TEXT NOT NULL DEFAULT 'medium',
    source_type TEXT NOT NULL DEFAULT 'news_article',
    provider_name TEXT NOT NULL DEFAULT 'finmind',
    tags TEXT,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, url)
);

ALTER TABLE stock_news_articles
    ADD COLUMN IF NOT EXISTS source_tier TEXT NOT NULL DEFAULT 'medium';

ALTER TABLE stock_news_articles
    ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'news_article';

ALTER TABLE stock_news_articles
    ADD COLUMN IF NOT EXISTS provider_name TEXT NOT NULL DEFAULT 'finmind';

ALTER TABLE stock_news_articles
    ADD COLUMN IF NOT EXISTS tags TEXT;

CREATE INDEX IF NOT EXISTS idx_stock_news_articles_lookup
    ON stock_news_articles (ticker, published_at DESC);

CREATE TABLE IF NOT EXISTS margin_purchase_short_sale_bars (
    ticker VARCHAR(64) NOT NULL,
    trading_date DATE NOT NULL,
    margin_purchase_buy BIGINT,
    margin_purchase_cash_repayment BIGINT,
    margin_purchase_limit BIGINT,
    margin_purchase_sell BIGINT,
    margin_purchase_today_balance BIGINT,
    margin_purchase_yesterday_balance BIGINT,
    offset_loan_and_short BIGINT,
    short_sale_buy BIGINT,
    short_sale_cash_repayment BIGINT,
    short_sale_limit BIGINT,
    short_sale_sell BIGINT,
    short_sale_today_balance BIGINT,
    short_sale_yesterday_balance BIGINT,
    note TEXT,
    source_name TEXT NOT NULL DEFAULT 'FinMind TaiwanStockMarginPurchaseShortSale',
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, trading_date)
);

CREATE INDEX IF NOT EXISTS idx_margin_purchase_short_sale_lookup
    ON margin_purchase_short_sale_bars (ticker, trading_date DESC);
