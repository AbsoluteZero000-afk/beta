"""
QuantEdge — Centralized Configuration.

Loads all settings from environment variables (.env file) using pydantic-settings.
Official docs:
  - pydantic-settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
  - python-dotenv: https://pypi.org/project/python-dotenv/
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL + TimescaleDB connection settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    user: str = "quantedge"
    password: str = "quantedge_secret_change_me"
    db: str = "quantedge"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def asyncpg_dsn(self) -> str:
        """asyncpg uses the same DSN format as libpq."""
        return self.dsn


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class AlpacaSettings(BaseSettings):
    """Alpaca Markets API settings."""

    model_config = SettingsConfigDict(env_prefix="ALPACA_")

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_feed: str = "iex"


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    watchlist: str = "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,SPY,QQQ"
    signal_buffer_size: int = 200

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)

    @property
    def watchlist_symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.watchlist.split(",") if s.strip()]


# Singleton — import this everywhere
settings = AppSettings()
