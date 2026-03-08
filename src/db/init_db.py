"""
QuantEdge — Database Initialization.

Reads schema.sql and applies it to TimescaleDB.
This runs when the app container starts to ensure tables exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.db.pool import get_pool

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def init_database() -> None:
    """
    Initialize the database schema.

    The schema.sql file is also mounted into the TimescaleDB container
    via docker-entrypoint-initdb.d for first-run initialization.
    This function provides a fallback for when the app starts after
    the DB is already running.
    """
    pool = await get_pool()

    # Check if TimescaleDB extension is active
    row = await pool.fetchrow(
        "SELECT default_version, installed_version "
        "FROM pg_available_extensions WHERE name = 'timescaledb'"
    )

    if row is None:
        raise RuntimeError(
            "TimescaleDB extension is NOT available. "
            "Ensure you are using the timescale/timescaledb-ha Docker image."
        )

    if row["installed_version"] is None:
        logger.warning("TimescaleDB not yet installed — running CREATE EXTENSION")
        await pool.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    logger.info(
        "TimescaleDB version: %s (installed: %s)",
        row["default_version"],
        row["installed_version"],
    )

    # Apply schema (idempotent — uses IF NOT EXISTS everywhere)
    schema_sql = SCHEMA_PATH.read_text()
    await pool.execute(schema_sql)
    logger.info("Database schema applied successfully")


async def verify_tables() -> dict[str, bool]:
    """Verify all expected tables exist."""
    pool = await get_pool()
    expected = ["ohlcv_bars", "signal_scores", "decisions", "execution_logs"]
    results: dict[str, bool] = {}

    for table in expected:
        exists = await pool.fetchval(
            "SELECT EXISTS ("
            "  SELECT FROM information_schema.tables "
            "  WHERE table_name = $1"
            ")",
            table,
        )
        results[table] = exists

    return results
