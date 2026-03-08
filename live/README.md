# QuantEdge Live Trading — Phase 7

## Quick Start

1. Add to your `.env`:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true

# Optional
SLACK_TOKEN=xoxb-...
SLACK_CHANNEL=#trading-alerts
CAPITAL_PER_TRADE=5000
MIN_GAP_PCT=2.0
MIN_VOL_RATIO=0.3
RISK_REWARD=2.0
MAX_POSITIONS=3
MONITOR_INTERVAL=60
```

2. Run the live system:
```bash
python -m src.live.live_runner
```

## Daily Schedule (ET)
| Time | Action |
|---|---|
| 4:00–9:29 AM | Gap scanner every 5 min, builds watchlist |
| 9:31 AM | First-candle confirmation → entries placed |
| 9:32 AM–4:00 PM | Continuous monitor: position management + new setups |
| 4:00 PM | All positions flattened, daily reset |

## Top Backtest Universe (Phase 6 Results)
Symbols with Sharpe > 2.4 and Profit Factor > 1.0:
UPST, AFRM, HOOD, TSLA, NFLX, AMD, COIN, LYFT, SNAP, ETHA, SQQQ, TZA, GLD, USO

## Files
- `position_manager.py` — tracks stops, targets, trailing stops
- `gap_trader.py`       — pre-market scan + open entry logic
- `continuous_monitor.py` — intraday loop, Slack alerts
- `live_runner.py`      — main entry point
