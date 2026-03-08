"""
QuantEdge — Backtest Results Page.

Run a backtest directly from the dashboard.
Fetches historical bars from Alpaca, runs the engine,
and renders tearsheet + equity curves + trade log.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="Backtest Results", page_icon="🔬", layout="wide")
st.title("🔬 Backtest Results")

with st.sidebar:
    st.header("Parameters")
    symbols_raw  = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,NVDA")
    days         = st.slider("Lookback (days)", 90, 730, 365, step=30)
    capital      = st.number_input("Initial Capital ($)", value=100_000, step=10_000)
    threshold    = st.slider("Signal Threshold", 0.05, 0.60, 0.25, step=0.05)
    max_hold     = st.slider("Max Hold Bars", 5, 50, 20)
    pos_size     = st.slider("Position Size %", 0.05, 0.30, 0.10, step=0.01)
    walk_forward = st.checkbox("Walk-Forward Validation", value=False)
    n_splits     = st.slider("WF Splits", 3, 10, 5) if walk_forward else 5
    run_btn      = st.button("▶ Run Backtest", type="primary")

if not run_btn:
    st.info("Configure parameters in the sidebar and click **▶ Run Backtest**.")
    st.stop()

symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
end     = datetime.now(timezone.utc)
start   = end - timedelta(days=days)

engine_kwargs = {
    "initial_capital":   float(capital),
    "signal_threshold":  threshold,
    "max_hold_bars":     max_hold,
    "position_size_pct": pos_size,
}

from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardValidator
from src.data.alpaca_fetcher import fetch_historical_bars

tearsheets = []
equity_map = {}
trades_map = {}
progress   = st.progress(0, text="Fetching data...")

for idx, sym in enumerate(symbols):
    progress.progress(idx / len(symbols), text=f"Running {sym}...")
    bars = fetch_historical_bars(sym, start, end)
    if bars is None or len(bars) < 100:
        st.warning(f"Insufficient data for {sym} — skipped.")
        continue

    if walk_forward:
        wf = WalkForwardValidator(n_splits=n_splits, engine_kwargs=engine_kwargs)
        r  = wf.run(bars, symbol=sym)
        tearsheets.append(r.combined_tearsheet)
        equity_map[sym] = r.combined_equity
        trades_map[sym] = r.combined_trades
    else:
        engine = BacktestEngine(**engine_kwargs)
        r      = engine.run(bars, symbol=sym)
        tearsheets.append(r.tearsheet)
        equity_map[sym] = r.equity
        trades_map[sym] = r.trades

progress.progress(1.0, text="Done!")

if not tearsheets:
    st.error("No results — check symbols and Alpaca API keys.")
    st.stop()

# ---- Tearsheet ----
st.subheader("📊 Tearsheet Summary")
ts_df = pd.DataFrame(tearsheets).set_index("symbol")
metric_cols = [
    "total_return_pct", "ann_return_pct", "sharpe", "sortino",
    "max_drawdown_pct", "calmar", "total_trades", "win_rate_pct",
    "profit_factor", "avg_trade_pnl", "best_trade", "worst_trade",
]
ts_df = ts_df[[c for c in metric_cols if c in ts_df.columns]]
ts_df.columns = [
    "Total Ret %", "Ann Ret %", "Sharpe", "Sortino",
    "Max DD %", "Calmar", "Trades", "Win Rate %",
    "Profit Factor", "Avg Trade $", "Best $", "Worst $",
]
st.dataframe(
    ts_df.style.applymap(
        lambda v: "color: #34d399" if isinstance(v, (int, float)) and v > 0
        else "color: #f87171" if isinstance(v, (int, float)) and v < 0 else ""
    ),
    use_container_width=True,
)

# ---- Equity curves ----
st.subheader("📈 Equity Curves")
colors = ["#7c3aed", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
fig    = go.Figure()
for i, (sym, eq) in enumerate(equity_map.items()):
    if eq.empty:
        continue
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name=sym,
        line=dict(color=colors[i % len(colors)], width=2),
    ))
fig.update_layout(
    height=400, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(tickprefix="$", gridcolor="#2d3748"),
    xaxis=dict(gridcolor="#2d3748"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Trade log ----
st.subheader("📋 Trade Log")
sym_select = st.selectbox("Select symbol", list(trades_map.keys()))
if sym_select and sym_select in trades_map:
    df_t = trades_map[sym_select]
    if not df_t.empty:
        df_t["pnl"] = df_t["pnl"].astype(float)
        st.dataframe(
            df_t.style.applymap(
                lambda v: "color: #34d399" if isinstance(v, float) and v > 0
                else "color: #f87171" if isinstance(v, float) and v < 0 else "",
                subset=["pnl"],
            ),
            use_container_width=True, height=300,
        )
    else:
        st.info("No trades generated for this symbol.")
