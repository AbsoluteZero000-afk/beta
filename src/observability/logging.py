"""
QuantEdge — Structured Logging (structlog).

Configures structlog with:
  - ISO timestamps (UTC)
  - Log level filtering
  - JSON output in production (LOG_FORMAT=json)
  - Human-readable ConsoleRenderer in development (LOG_FORMAT=console)
  - orjson serializer for fast JSON encoding
  - async-safe (structlog ainfo/awarning/aerror methods)

All modules get a logger via:
    import structlog
    logger = structlog.get_logger(__name__)

Official docs:
  https://www.structlog.org/en/stable/configuration.html
  https://www.structlog.org/en/stable/processors.html
"""

from __future__ import annotations

import logging
import sys

import orjson
import structlog


def configure_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """
    Configure structlog for the entire application.
    Call once at startup in main.py before any other imports use logging.

    Args:
        log_level:  "DEBUG", "INFO", "WARNING", "ERROR"
        log_format: "console" (human-readable) or "json" (production)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors — run on every log event regardless of renderer
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if log_format.lower() == "json":
        # Production: compact JSON via orjson (faster than stdlib json)
        renderer = structlog.processors.JSONRenderer(
            serializer=lambda v, **kw: orjson.dumps(v).decode()
        )
    else:
        # Development: coloured, human-readable output
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libraries
    # (alpaca-py, asyncpg, etc.) respect our log level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    # Quiet noisy third-party loggers
    logging.getLogger("alpaca").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
