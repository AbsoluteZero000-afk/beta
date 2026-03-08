"""
Fetches historical OHLCV bars from Alpaca Markets REST API.
Returns a pandas DataFrame with a UTC DatetimeIndex.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

_ALPACA_KEY    = os.environ.get("ALPACA_API_KEY", "")
_ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
_BASE_URL      = "https://data.alpaca.markets/v2"


def fetch_historical_bars(
    symbol:    str,
    start:     datetime,
    end:       datetime,
    timeframe: str = "1Day",
) -> pd.DataFrame | None:
    """
    Fetch historical OHLCV bars from Alpaca.

    Parameters
    ----------
    symbol    : Ticker e.g. "AAPL"
    start     : Start datetime (UTC-aware)
    end       : End datetime (UTC-aware)
    timeframe : Alpaca timeframe string ("1Day", "1Hour", "5Min", etc.)

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    and a UTC DatetimeIndex, or None on failure.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("pip install requests")

    headers = {
        "APCA-API-KEY-ID":     _ALPACA_KEY,
        "APCA-API-SECRET-KEY": _ALPACA_SECRET,
    }
    params = {
        "start":     start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end":       end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeframe": timeframe,
        "limit":     10000,
        "feed":      "iex",
    }

    all_bars = []
    url      = f"{_BASE_URL}/stocks/{symbol}/bars"

    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"Alpaca API error {resp.status_code}: {resp.text}")
            return None
        data       = resp.json()
        all_bars.extend(data.get("bars") or [])
        next_token = data.get("next_page_token")
        params     = {"page_token": next_token, "limit": 10000, "feed": "iex"} if next_token else None
        url        = url if next_token else None

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars).rename(
        columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()[["open", "high", "low", "close", "volume"]]
