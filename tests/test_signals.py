"""
Tests for Phase 3 — Signal Engine.
Run with: pytest tests/test_signals.py -v

Note: First run will be slow (~3-10s) due to numba JIT compilation.
Subsequent runs use cached compiled code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from src.signals.kernels import (
    compute_rsi,
    compute_momentum_score,
    compute_zscore,
    compute_bollinger_position,
    compute_volume_anomaly,
    compute_realized_volatility,
    compute_volatility_signal,
    compute_trend_strength,
)
from src.signals.momentum import MomentumSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.volume_anomaly import VolumeAnomalySignal
from src.signals.volatility_regime import VolatilityRegimeSignal
from src.signals.regime_detector import RegimeDetector
from src.signals.composer import SignalComposer, REGIME_WEIGHTS
from src.events.types import MarketRegime


# ---- Helpers ----

def make_df(closes, volumes=None, n=None):
    """Build a minimal OHLCV DataFrame from a closes array."""
    if n:
        closes = np.linspace(100, 110, n)
    closes = np.array(closes, dtype=float)
    if volumes is None:
        volumes = np.full(len(closes), 10_000, dtype=float)
    idx = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
           for i in range(len(closes))]
    return pd.DataFrame({
        "open":   closes - 0.5,
        "high":   closes + 1.0,
        "low":    closes - 1.0,
        "close":  closes,
        "volume": volumes,
    }, index=idx)


def trending_up_df(n=50):
    closes = np.linspace(100, 130, n)  # Strong uptrend
    return make_df(closes)


def trending_down_df(n=50):
    closes = np.linspace(130, 100, n)  # Strong downtrend
    return make_df(closes)


def mean_reverting_df(n=50):
    np.random.seed(42)
    closes = 100 + np.random.normal(0, 0.5, n).cumsum() * 0.1
    return make_df(closes)


def volatile_df(n=50):
    np.random.seed(99)
    closes = 100 + np.random.normal(0, 5, n).cumsum()
    return make_df(closes)


# ---- Kernel tests ----

def test_rsi_returns_in_range():
    closes = np.linspace(100, 120, 30).astype(np.float64)
    score = compute_rsi(closes, 14)
    assert -1.0 <= score <= 1.0


def test_rsi_uptrend_positive():
    closes = np.linspace(100, 130, 30).astype(np.float64)
    score = compute_rsi(closes, 14)
    assert score > 0  # Strong uptrend → RSI > 50 → positive score


def test_rsi_downtrend_negative():
    closes = np.linspace(130, 100, 30).astype(np.float64)
    score = compute_rsi(closes, 14)
    assert score < 0


def test_rsi_insufficient_data():
    closes = np.array([100.0, 101.0, 102.0])
    assert compute_rsi(closes, 14) == 0.0


def test_momentum_score_uptrend():
    closes = np.linspace(100, 120, 30).astype(np.float64)
    score = compute_momentum_score(closes, 5, 20)
    assert score > 0


def test_zscore_above_mean_negative():
    """Price well above mean → overbought → negative score."""
    closes = np.concatenate([
        np.full(19, 100.0),
        np.array([120.0])  # Far above mean
    ]).astype(np.float64)
    score = compute_zscore(closes, 20)
    assert score < 0


def test_zscore_below_mean_positive():
    """Price well below mean → oversold → positive score."""
    closes = np.concatenate([
        np.full(19, 100.0),
        np.array([80.0])  # Far below mean
    ]).astype(np.float64)
    score = compute_zscore(closes, 20)
    assert score > 0


def test_bollinger_position_range():
    closes = np.linspace(95, 105, 30).astype(np.float64)
    score = compute_bollinger_position(closes, 20, 2.0)
    assert -1.0 <= score <= 1.0


def test_volume_anomaly_bullish():
    """Up move + volume spike = positive score."""
    closes = np.concatenate([np.full(22, 100.0), np.array([101.0])]).astype(np.float64)
    volumes = np.concatenate([np.full(22, 1000.0), np.array([10000.0])]).astype(np.float64)
    score = compute_volume_anomaly(closes, volumes, 20)
    assert score > 0


def test_volume_anomaly_bearish():
    """Down move + volume spike = negative score."""
    closes = np.concatenate([np.full(22, 100.0), np.array([99.0])]).astype(np.float64)
    volumes = np.concatenate([np.full(22, 1000.0), np.array([10000.0])]).astype(np.float64)
    score = compute_volume_anomaly(closes, volumes, 20)
    assert score < 0


def test_realized_volatility_nonnegative():
    closes = np.linspace(100, 105, 30).astype(np.float64)
    vol = compute_realized_volatility(closes, 20)
    assert vol >= 0


def test_trend_strength_uptrend_positive():
    closes = np.linspace(100, 120, 25).astype(np.float64)
    score = compute_trend_strength(closes, 20)
    assert score > 0


def test_trend_strength_downtrend_negative():
    closes = np.linspace(120, 100, 25).astype(np.float64)
    score = compute_trend_strength(closes, 20)
    assert score < 0


# ---- Signal class tests ----

def test_momentum_signal_uptrend():
    sig = MomentumSignal()
    score = sig.safe_compute(trending_up_df())
    assert score > 0


def test_momentum_signal_downtrend():
    sig = MomentumSignal()
    score = sig.safe_compute(trending_down_df())
    assert score < 0


def test_mean_reversion_signal_range():
    sig = MeanReversionSignal()
    score = sig.safe_compute(mean_reverting_df())
    assert -1.0 <= score <= 1.0


def test_volume_anomaly_signal_neutral_volume():
    """Flat price + constant volume should be near zero."""
    df = make_df(np.full(30, 100.0))
    sig = VolumeAnomalySignal()
    score = sig.safe_compute(df)
    assert abs(score) < 0.2


def test_volatility_regime_signal_range():
    sig = VolatilityRegimeSignal()
    score = sig.safe_compute(volatile_df())
    assert -1.0 <= score <= 1.0


def test_signal_returns_zero_on_insufficient_data():
    sig = MomentumSignal()
    tiny_df = make_df(np.array([100.0, 101.0]))
    score = sig.safe_compute(tiny_df)
    assert score == 0.0


# ---- Regime Detector tests ----

def test_regime_detector_trending_up():
    detector = RegimeDetector()
    regime = detector.detect(trending_up_df(50))
    assert regime in (MarketRegime.TRENDING_UP, MarketRegime.MEAN_REVERTING)


def test_regime_detector_volatile():
    detector = RegimeDetector()
    regime = detector.detect(volatile_df(50))
    assert regime == MarketRegime.VOLATILE


def test_regime_detector_unknown_on_short_data():
    detector = RegimeDetector()
    regime = detector.detect(make_df(np.array([100.0, 101.0])))
    assert regime == MarketRegime.UNKNOWN


# ---- SignalComposer tests ----

def test_composer_weights_sum_to_one():
    """All regime weight tables must sum to 1.0."""
    for regime, weights in REGIME_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"{regime} weights sum to {total}"


def test_composer_composite_in_range():
    composer = SignalComposer()
    result = composer.compute("AAPL", trending_up_df())
    assert -1.0 <= result.composite_score <= 1.0


def test_composer_uptrend_positive_score():
    composer = SignalComposer()
    result = composer.compute("AAPL", trending_up_df(50))
    assert result.composite_score > 0


def test_composer_downtrend_negative_score():
    composer = SignalComposer()
    result = composer.compute("AAPL", trending_down_df(50))
    assert result.composite_score < 0


def test_composer_signal_scores_dict():
    composer = SignalComposer()
    result = composer.compute("AAPL", trending_up_df())
    d = result.signal_scores_dict
    assert set(d.keys()) == {"momentum", "mean_reversion", "volume_anomaly", "volatility_regime"}
    for v in d.values():
        assert -1.0 <= v <= 1.0


def test_composer_regime_aware_weights():
    """In trending regime, momentum weight should be highest."""
    composer = SignalComposer()
    result = composer.compute("SPY", trending_up_df(50), regime=MarketRegime.TRENDING_UP)
    momentum_result = next(r for r in result.signal_results if r.name == "momentum")
    assert momentum_result.weight == 0.50


def test_composer_mean_reverting_weights():
    """In mean_reverting regime, mean_reversion weight should be highest."""
    composer = SignalComposer()
    result = composer.compute("SPY", mean_reverting_df(50), regime=MarketRegime.MEAN_REVERTING)
    mr_result = next(r for r in result.signal_results if r.name == "mean_reversion")
    assert mr_result.weight == 0.55
