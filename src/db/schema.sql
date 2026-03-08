-- ============================================
-- QuantEdge — TimescaleDB Schema (Phase 1)
-- ============================================
-- Official TimescaleDB docs:
--   https://docs.timescale.com/self-hosted/latest/install/
--   https://docs.timescale.com/api/latest/hypertable/create_hypertable/
-- ============================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================
-- 1. OHLCV Bars
-- ============================================
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    timeframe   TEXT            NOT NULL DEFAULT '1Min',
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL,
    vwap        DOUBLE PRECISION,
    trade_count INTEGER
);

-- Convert to hypertable partitioned by time (7-day chunks)
SELECT create_hypertable(
    'ohlcv_bars',
    by_range('time', INTERVAL '7 days'),
    if_not_exists => TRUE
);

-- Composite index for fast lookups by symbol + time range
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
    ON ohlcv_bars (symbol, time DESC);

-- ============================================
-- 2. Signal Scores
-- ============================================
CREATE TABLE IF NOT EXISTS signal_scores (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    signal_name     TEXT            NOT NULL,
    score           DOUBLE PRECISION NOT NULL,  -- normalized [-1, 1]
    metadata        JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'signal_scores',
    by_range('time', INTERVAL '7 days'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_signal_symbol_time
    ON signal_scores (symbol, time DESC);

-- ============================================
-- 3. Decisions
-- ============================================
CREATE TABLE IF NOT EXISTS decisions (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    regime          TEXT            NOT NULL,
    decision_class  TEXT            NOT NULL,
    signal_scores   JSONB           NOT NULL DEFAULT '{}'::jsonb,
    action_taken    TEXT,
    order_id        TEXT,
    outcome_pnl     DOUBLE PRECISION,
    metadata        JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'decisions',
    by_range('time', INTERVAL '7 days'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_decisions_symbol_time
    ON decisions (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_decisions_class
    ON decisions (decision_class, time DESC);

-- ============================================
-- 4. Execution Logs
-- ============================================
CREATE TABLE IF NOT EXISTS execution_logs (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    order_id        TEXT            NOT NULL,
    order_type      TEXT            NOT NULL,
    side            TEXT            NOT NULL,
    qty             DOUBLE PRECISION NOT NULL,
    filled_qty      DOUBLE PRECISION,
    filled_avg_price DOUBLE PRECISION,
    status          TEXT            NOT NULL DEFAULT 'pending',
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    bracket_parent_id TEXT,
    error_message   TEXT,
    raw_response    JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'execution_logs',
    by_range('time', INTERVAL '7 days'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_exec_symbol_time
    ON execution_logs (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_exec_order_id
    ON execution_logs (order_id);

CREATE INDEX IF NOT EXISTS idx_exec_status
    ON execution_logs (status, time DESC);
