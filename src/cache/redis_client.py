"""
QuantEdge — Async Redis Client.

Uses redis-py's built-in async support (redis.asyncio).
aioredis is DEPRECATED and was merged into redis-py >= 4.2.0.

Official docs: https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from src.config import settings

logger = logging.getLogger(__name__)

_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """
    Get or create a singleton async Redis client.

    redis.asyncio creates an internal connection pool automatically.
    Connections are established lazily on first use.
    """
    global _client
    if _client is not None:
        return _client

    logger.info("Creating async Redis client: %s", settings.redis.url)
    _client = aioredis.from_url(
        settings.redis.url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    # Verify connectivity
    pong = await _client.ping()
    logger.info("Redis connected: PING -> %s", pong)
    return _client


async def close_redis() -> None:
    """Gracefully close the Redis client and its connection pool."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Redis client closed")


async def cache_set(key: str, value: str, ttl: int = 300) -> None:
    """Set a key with TTL (seconds)."""
    client = await get_redis()
    await client.set(key, value, ex=ttl)


async def cache_get(key: str) -> Optional[str]:
    """Get a cached value by key."""
    client = await get_redis()
    return await client.get(key)


async def cache_delete(key: str) -> None:
    """Delete a cached key."""
    client = await get_redis()
    await client.delete(key)


async def cache_hset(name: str, mapping: dict[str, Any]) -> None:
    """Set multiple hash fields."""
    client = await get_redis()
    await client.hset(name, mapping=mapping)


async def cache_hgetall(name: str) -> dict[str, str]:
    """Get all fields from a hash."""
    client = await get_redis()
    return await client.hgetall(name)
