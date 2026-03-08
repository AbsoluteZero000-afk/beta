-- Phase 4 migration: extend trades table with execution columns
-- Run via: psql $DATABASE_URL -f migrations/002_phase4_trades.sql

ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS client_order_id  TEXT UNIQUE,
    ADD COLUMN IF NOT EXISTS alpaca_order_id  TEXT,
    ADD COLUMN IF NOT EXISTS status           TEXT DEFAULT 'submitted',
    ADD COLUMN IF NOT EXISTS pnl              NUMERIC(12, 4),
    ADD COLUMN IF NOT EXISTS pnl_pct          NUMERIC(10, 6),
    ADD COLUMN IF NOT EXISTS fill_price       NUMERIC(12, 4),
    ADD COLUMN IF NOT EXISTS qty              NUMERIC(12, 4);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
    ON trades (symbol, status);

CREATE INDEX IF NOT EXISTS idx_trades_client_order_id
    ON trades (client_order_id);
