"""
QuantEdge — Numba JIT Kernels.

All hot numerical loops are isolated here and decorated with
@numba.jit(nopython=True) for maximum performance.

Rules for numba nopython=True compatibility:
  - Input must be numpy arrays (float64), NOT pandas objects
  - No Python dicts, sets, or arbitrary objects
  - No try/except inside JIT functions
  - Use numpy math functions (np.sqrt, np.abs, np.exp, etc.)
  - No f-strings or string formatting

First call triggers JIT compilation (~1-3s). All subsequent calls
use cached compiled machine code. On M1 this compiles to ARM64.

Official docs:
  https://numba.readthedocs.io/en/stable/user/jit.html
  https://numba.readthedocs.io/en/stable/reference/numpysupported.html
"""

from __future__ import annotations

import numba
import numpy as np


# ============================================================
# MOMENTUM KERNELS
# ============================================================

@numba.jit(nopython=True, cache=True)
def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """
    Relative Strength Index (RSI) normalized to [-1, 1].

    RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss
    Normalized: (RSI - 50) / 50  => [-1, 1]

    Returns 0.0 if period > len(closes) - 1.
    """
    n = len(closes)
    if n < period + 1:
        return 0.0

    gains = np.empty(n - 1)
    losses = np.empty(n - 1)
    for i in range(n - 1):
        delta = closes[i + 1] - closes[i]
        if delta > 0.0:
            gains[i] = delta
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = -delta

    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(period):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 1.0  # All gains = maximum bullish
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Normalize: RSI 0-100 -> -1 to +1
    return (rsi - 50.0) / 50.0


@numba.jit(nopython=True, cache=True)
def compute_momentum_score(closes: np.ndarray, fast: int = 5, slow: int = 20) -> float:
    """
    Dual moving average momentum: (fast_ma - slow_ma) / slow_ma.
    Normalized via tanh to [-1, 1].
    """
    n = len(closes)
    if n < slow:
        return 0.0

    fast_sum = 0.0
    for i in range(n - fast, n):
        fast_sum += closes[i]
    fast_ma = fast_sum / fast

    slow_sum = 0.0
    for i in range(n - slow, n):
        slow_sum += closes[i]
    slow_ma = slow_sum / slow

    if slow_ma == 0.0:
        return 0.0
    raw = (fast_ma - slow_ma) / slow_ma
    # tanh maps any real number to (-1, 1)
    # Scale by 100 so 1% move maps to tanh(1) ~ 0.76
    return np.tanh(raw * 100.0)


# ============================================================
# MEAN REVERSION KERNELS
# ============================================================

@numba.jit(nopython=True, cache=True)
def compute_zscore(closes: np.ndarray, period: int = 20) -> float:
    """
    Z-score of current price vs rolling mean/std.
    z = (close - mean) / std, then clamped and normalized.

    Positive z = price above mean (overbought -> bearish signal).
    We negate so +1 = oversold (buy signal), -1 = overbought (sell signal).
    """
    n = len(closes)
    if n < period:
        return 0.0

    window = closes[n - period:]
    mean = 0.0
    for v in window:
        mean += v
    mean /= period

    var = 0.0
    for v in window:
        diff = v - mean
        var += diff * diff
    var /= period
    std = np.sqrt(var)

    if std < 1e-10:
        return 0.0

    z = (closes[-1] - mean) / std
    # Negate: high z = overbought = sell signal (-1)
    # Clamp z to [-3, 3] then normalize to [-1, 1]
    z_clamped = max(-3.0, min(3.0, z))
    return -z_clamped / 3.0


@numba.jit(nopython=True, cache=True)
def compute_bollinger_position(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    """
    Position within Bollinger Bands normalized to [-1, 1].

    0.0 = at midband (neutral)
    -1.0 = at or below lower band (oversold, bullish signal)
    +1.0 = at or above upper band (overbought, bearish signal)

    We negate for signal convention: lower band = buy.
    """
    n = len(closes)
    if n < period:
        return 0.0

    window = closes[n - period:]
    mean = 0.0
    for v in window:
        mean += v
    mean /= period

    var = 0.0
    for v in window:
        diff = v - mean
        var += diff * diff
    std = np.sqrt(var / period)

    if std < 1e-10:
        return 0.0

    upper = mean + num_std * std
    lower = mean - num_std * std
    band_width = upper - lower

    if band_width < 1e-10:
        return 0.0

    current = closes[-1]
    # Position: 0 = lower band, 0.5 = midband, 1.0 = upper band
    position = (current - lower) / band_width
    # Map to [-1, 1] and negate: upper band -> bearish (-1)
    return -(position * 2.0 - 1.0)


# ============================================================
# VOLUME ANOMALY KERNELS
# ============================================================

@numba.jit(nopython=True, cache=True)
def compute_volume_anomaly(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> float:
    """
    Volume anomaly signal combining:
      1. Volume ratio: current / rolling_avg
      2. Price direction: up or down move on the spike

    Returns positive score for bullish volume spikes (price up + high vol)
    Returns negative score for bearish volume spikes (price down + high vol)
    Normal volume returns near zero.
    """
    n = len(closes)
    if n < period + 1:
        return 0.0

    # Rolling average volume (exclude current bar)
    avg_vol = 0.0
    for i in range(n - period - 1, n - 1):
        avg_vol += volumes[i]
    avg_vol /= period

    if avg_vol < 1.0:
        return 0.0

    current_vol = volumes[-1]
    vol_ratio = current_vol / avg_vol

    # Price direction of current bar
    price_change = closes[-1] - closes[-2]
    price_pct = price_change / closes[-2] if closes[-2] != 0.0 else 0.0

    # Score: direction * volume_intensity, normalized via tanh
    raw_score = price_pct * vol_ratio
    return np.tanh(raw_score * 50.0)


# ============================================================
# VOLATILITY REGIME KERNELS
# ============================================================

@numba.jit(nopython=True, cache=True)
def compute_realized_volatility(closes: np.ndarray, period: int = 20) -> float:
    """
    Annualized realized volatility from log returns.
    Returns value in [0, inf). Typical intraday: 0.10-0.50 annualized.
    """
    n = len(closes)
    if n < period + 1:
        return 0.0

    log_returns = np.empty(period)
    for i in range(period):
        idx = n - period - 1 + i
        if closes[idx] <= 0.0:
            return 0.0
        log_returns[i] = np.log(closes[idx + 1] / closes[idx])

    mean_ret = 0.0
    for r in log_returns:
        mean_ret += r
    mean_ret /= period

    var = 0.0
    for r in log_returns:
        diff = r - mean_ret
        var += diff * diff
    var /= (period - 1)

    # Annualize: 1-min bars -> 252 * 390 bars per year
    return np.sqrt(var * 252.0 * 390.0)


@numba.jit(nopython=True, cache=True)
def compute_volatility_signal(closes: np.ndarray, short: int = 5, long: int = 20) -> float:
    """
    Volatility regime signal.

    Compares short-term vs long-term volatility.
    When short vol > long vol (vol expanding) -> bearish (-1)
    When short vol < long vol (vol contracting) -> bullish (+1)

    Returns normalized score in [-1, 1].
    """
    n = len(closes)
    if n < long + 1:
        return 0.0

    short_vol = compute_realized_volatility(closes[n - short - 1:], short)
    long_vol  = compute_realized_volatility(closes, long)

    if long_vol < 1e-10:
        return 0.0

    # Ratio of short/long vol, normalized
    ratio = short_vol / long_vol
    # ratio > 1 = expanding vol = bearish, ratio < 1 = contracting = bullish
    return np.tanh(-(ratio - 1.0) * 3.0)


@numba.jit(nopython=True, cache=True)
def compute_trend_strength(closes: np.ndarray, period: int = 20) -> float:
    """
    Trend strength via linear regression slope normalized by price.
    Returns positive for uptrend, negative for downtrend, near-zero for flat.
    Normalized to [-1, 1] via tanh.
    """
    n = len(closes)
    if n < period:
        return 0.0

    window = closes[n - period:]
    x_mean = (period - 1) / 2.0
    y_mean = 0.0
    for v in window:
        y_mean += v
    y_mean /= period

    numerator = 0.0
    denominator = 0.0
    for i in range(period):
        x_dev = i - x_mean
        numerator   += x_dev * (window[i] - y_mean)
        denominator += x_dev * x_dev

    if denominator < 1e-10 or y_mean == 0.0:
        return 0.0

    slope = numerator / denominator
    # Normalize slope by price level to get a % per bar
    normalized_slope = slope / y_mean
    return np.tanh(normalized_slope * 1000.0)
