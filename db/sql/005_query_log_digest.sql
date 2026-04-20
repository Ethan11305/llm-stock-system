-- P0 Digest 產品線的可追溯閉環 schema 升級
--
-- 與既有 002_schema.sql / 003_query_log_observability.sql 相容：
--   * query_logs          ：ALTER ADD COLUMN（不重建，避免破壞既有資料）
--   * query_sources       ：ALTER ADD COLUMN corroboration_count
--   * query_log_warnings  ：新建
--
-- 舊欄位 time_range / response_text 暫時保留，維持與 legacy pipeline 的相容性。
-- 之後可選擇性 backfill：
--   time_range   -> time_range_label
--   response_text-> summary / response_json

-- 1) 擴充既有 query_logs 以支援 digest 可信閉環
ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS company_name TEXT,
    ADD COLUMN IF NOT EXISTS query_profile TEXT NOT NULL DEFAULT 'legacy',
    ADD COLUMN IF NOT EXISTS classifier_source TEXT NOT NULL DEFAULT 'rule',
    ADD COLUMN IF NOT EXISTS time_range_label TEXT,
    ADD COLUMN IF NOT EXISTS time_range_days INT,
    ADD COLUMN IF NOT EXISTS confidence_light TEXT,
    ADD COLUMN IF NOT EXISTS sufficiency_status TEXT,
    ADD COLUMN IF NOT EXISTS consistency_status TEXT,
    ADD COLUMN IF NOT EXISTS freshness_status TEXT,
    ADD COLUMN IF NOT EXISTS summary TEXT,
    ADD COLUMN IF NOT EXISTS schema_version INT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS response_json JSONB,
    ADD COLUMN IF NOT EXISTS structured_query_json JSONB;

CREATE INDEX IF NOT EXISTS idx_query_logs_query_profile
    ON query_logs (query_profile, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_query_logs_ticker_created_at
    ON query_logs (ticker, created_at DESC);

-- 2) 擴充既有 query_sources（沿用此表，不另建 query_log_sources）
ALTER TABLE query_sources
    ADD COLUMN IF NOT EXISTS corroboration_count INT NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_query_sources_query_log_id
    ON query_sources (query_log_id);

-- 3) 新建 query_log_warnings（可信閉環的第三張表）
CREATE TABLE IF NOT EXISTS query_log_warnings (
    id UUID PRIMARY KEY,
    query_log_id UUID NOT NULL REFERENCES query_logs(id) ON DELETE CASCADE,
    warning_text TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_log_warnings_query_log_id
    ON query_log_warnings (query_log_id);
