"""
Tests for database connectivity and schema.
Run with: pytest tests/test_db.py -v
Requires Docker containers to be running.
"""

import pytest
import pytest_asyncio

from src.db.pool import get_pool, close_pool
from src.db.init_db import verify_tables


@pytest.mark.asyncio
async def test_pool_creation():
    """Test that asyncpg pool can be created."""
    pool = await get_pool()
    assert pool is not None
    assert not pool._closed
    await close_pool()


@pytest.mark.asyncio
async def test_timescaledb_extension():
    """Test that TimescaleDB extension is installed."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT installed_version FROM pg_available_extensions WHERE name = 'timescaledb'"
    )
    assert row is not None
    assert row["installed_version"] is not None
    await close_pool()


@pytest.mark.asyncio
async def test_tables_exist():
    """Test that all expected tables exist."""
    tables = await verify_tables()
    for table, exists in tables.items():
        assert exists, f"Table {table} does not exist"
    await close_pool()
