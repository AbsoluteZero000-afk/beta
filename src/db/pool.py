"""
QuantEdge — Async PostgreSQL Connection Pool.

Uses asyncpg for high-performance async Postgres access.
Official docs: https://magicstack.github.io/asyncpg/current/
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import asyncpg
from asyncpg import Pool

from src.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[Pool] = None
_pool_lock = asyncio.Lock()


async def get_pool() -> Pool:
    """
    Get or create a singleton asyncpg connection pool.

    Uses a lock to prevent multiple coroutines from creating pools
    simultaneously during startup.
    """
    global _pool
    if _pool is not None and not _pool._closed:
        return _pool

    async with _pool_lock:
        # Double-check after acquiring lock
        if _pool is not None and not _pool._closed:
            return _pool

        logger.info(
            "Creating asyncpg pool: host=%s port=%s db=%s",
            settings.database.host,
            settings.database.port,
            settings.database.db,
        )
        _pool = await asyncpg.create_pool(
            dsn=settings.database.asyncpg_dsn,
            min_size=2,
            max_size=10,
            max_inactive_connection_lifetime=300.0,
            command_timeout=30.0,
        )
        logger.info("asyncpg pool created successfully")
        return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("asyncpg pool closed")


async def execute(query: str, *args) -> str:
    """Execute a single SQL statement."""
    pool = await get_pool()
    return await pool.execute(query, *args)


async def fetch(query: str, *args) -> list[asyncpg.Record]:
    """Fetch multiple rows."""
    pool = await get_pool()
    return await pool.fetch(query, *args)


async def fetchrow(query: str, *args) -> Optional[asyncpg.Record]:
    """Fetch a single row."""
    pool = await get_pool()
    return await pool.fetchrow(query, *args)


async def fetchval(query: str, *args):
    """Fetch a single value."""
    pool = await get_pool()
    return await pool.fetchval(query, *args)
