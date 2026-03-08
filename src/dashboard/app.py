"""
QuantEdge — Streamlit Live Dashboard.

Reads live state from TimescaleDB and Redis.
Runs as a SEPARATE process — not inside the main uvloop.
Auto-refreshes every REFRESH_SECONDS using st.rerun().

Launch with:
    streamlit run src/dashboard/app.py

Architecture note:
  Streamlit has its own event loop and cannot share the main
  asyncio loop. All DB/Redis reads use synchronous drivers here:
    - psycopg2 for PostgreSQL reads
    - redis-py (sync) for Redis reads
  This is intentional — the dashboard is read-only and isolated.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import psycopg2.extras
import redis
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)


def _build_db_url() -> str:
    if url := os.environ.get("DASHBOARD_DB_URL"):
        return url
    raw = os.environ["DATABASE_URL"]
    return (
        raw
        .replace("postgresql://", "postgres://", 1)
        .replace("@timescaledb:", "@127.0.0.1:", 1)
        .replace("@timescaledb/", "@127.0.0.1/", 1)
    )


DB_URL        = _build_db_url()
REDIS_URL     = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REFRESH_SECONDS = 10

# ---- Page config ----
st.set_page_config(
    page_title="QuantEdge Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- CSS ----
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #7c3aed;
    }
    .metric-label { color: #9ca3af; font-size: 12px; font-weight: 600; }
    .metric-value { color: #f9fafb; font-size: 28px; font-weight: 700; }
    .positive { color: #34d399 !important; }
    .negative { color: #f87171 !important; }
</style>
""", unsafe_allow_html=True)


# ---- DB connection (singleton with auto-reconnect) ----

_db_conn: psycopg2.extensions.connection | None = None


def get_db_conn() -> psycopg2.extensions.connection:
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            _db_conn = psycopg2.connect(DB_URL)
        _db_conn.cursor().execute("SELECT 1")
        return _db_conn
    except Exception:
        _db_conn = psycopg2.connect(DB_URL)
        return _db_conn


@st.cache_resource
def get_redis() -> redis.Redis:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    return r


# ---- Data loaders (synchronous) ----

def load_equity_curve(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    time_bucket('1 hour', time) AS bucket,
                    SUM(pnl) OVER (ORDER BY time_bucket('1 hour', time)) AS cumulative_pnl
                FROM trades
                WHERE status = 'closed'
                  AND time >= NOW() - INTERVAL '7 days'
                ORDER BY bucket
            """)
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["bucket", "cumulative_pnl"])
        return pd.DataFrame(rows)
    except Exception:
        conn.rollback()
        return pd.DataFrame(columns=["bucket", "cumulative_pnl"])


def load_open_positions(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT symbol, side, qty, fill_price, status, time
                FROM trades
                WHERE status = 'filled'
                ORDER BY time DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        conn.rollback()
        return pd.DataFrame()


def load_recent_trades(conn: psycopg2.extensions.connection, limit: int = 20) -> pd.DataFrame:
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT symbol, side, qty, fill_price, pnl, pnl_pct, status, time
                FROM trades
                WHERE status = 'closed'
                ORDER BY time DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        conn.rollback()
        return pd.DataFrame()


def load_daily_stats(conn: psycopg2.extensions.connection) -> dict:
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                      AS total_trades,
                    SUM(pnl)                                      AS total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)     AS wins,
                    MAX(pnl)                                      AS best_trade,
                    MIN(pnl)                                      AS worst_trade
                FROM trades
                WHERE status = 'closed'
                  AND time >= NOW() - INTERVAL '1 day'
            """)
            row = cur.fetchone()
        return dict(row) if row else {}
    except Exception:
        conn.rollback()
        return {}


def load_signal_scores(r: redis.Redis, symbols: list[str]) -> dict[str, float]:
    scores = {}
    for sym in symbols:
        val = r.hget(f"signal:{sym}", "composite_score")
        if val:
            try:
                scores[sym] = float(val)
            except ValueError:
                pass
    return scores


# ---- Layout ----

def render_dashboard() -> None:
    conn = get_db_conn()
    r    = get_redis()

    st.title("📈 QuantEdge Live Dashboard")
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    stats = load_daily_stats(conn)

    # ---- Top metrics row ----
    c1, c2, c3, c4, c5 = st.columns(5)
    total_pnl    = float(stats.get("total_pnl") or 0)
    total_trades = int(stats.get("total_trades") or 0)
    wins         = int(stats.get("wins") or 0)
    win_rate     = wins / total_trades if total_trades else 0
    best         = float(stats.get("best_trade") or 0)
    worst        = float(stats.get("worst_trade") or 0)

    with c1:
        st.metric("Today's PnL", f"${total_pnl:+,.2f}")
    with c2:
        st.metric("Total Trades", total_trades)
    with c3:
        st.metric("Win Rate", f"{win_rate:.1%}")
    with c4:
        st.metric("Best Trade", f"${best:+.2f}")
    with c5:
        st.metric("Worst Trade", f"${worst:+.2f}")

    st.divider()

    # ---- Row 2: Equity curve + Signal scores ----
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Cumulative PnL (7 days)")
        df_equity = load_equity_curve(conn)
        if not df_equity.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_equity["bucket"],
                y=df_equity["cumulative_pnl"].astype(float),
                mode="lines",
                line=dict(color="#7c3aed", width=2),
                fill="tozeroy",
                fillcolor="rgba(124,58,237,0.1)",
                name="Cumulative PnL",
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(tickprefix="$", gridcolor="#2d3748"),
                xaxis=dict(gridcolor="#2d3748"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet — equity curve will appear after first fills.")

    with col_right:
        st.subheader("Signal Scores")
        watchlist = os.environ.get("WATCHLIST", "AAPL,MSFT,GOOGL,NVDA,TSLA,SPY").split(",")
        scores = load_signal_scores(r, watchlist[:10])
        if scores:
            score_df = pd.DataFrame(
                [{"Symbol": sym, "Score": score} for sym, score in scores.items()]
            ).sort_values("Score", ascending=False)
            score_df["Bar"] = score_df["Score"].apply(
                lambda s: "🟢" * int(abs(s) * 5) if s > 0 else "🔴" * int(abs(s) * 5)
            )
            st.dataframe(
                score_df.style.background_gradient(subset=["Score"], cmap="RdYlGn"),
                use_container_width=True,
                height=280,
            )
        else:
            st.info("Signal scores will appear once the watcher is running.")

    st.divider()

    # ---- Row 3: Open positions + Recent trades ----
    col_pos, col_trades = st.columns(2)

    with col_pos:
        st.subheader("Open Positions")
        df_pos = load_open_positions(conn)
        if not df_pos.empty:
            st.dataframe(df_pos, use_container_width=True, height=250)
        else:
            st.info("No open positions.")

    with col_trades:
        st.subheader("Recent Closed Trades")
        df_trades = load_recent_trades(conn, limit=15)
        if not df_trades.empty:
            df_trades["pnl"] = df_trades["pnl"].astype(float)
            st.dataframe(
                df_trades.style.applymap(
                    lambda v: "color: #34d399" if isinstance(v, float) and v > 0
                    else "color: #f87171" if isinstance(v, float) and v < 0 else "",
                    subset=["pnl"],
                ),
                use_container_width=True,
                height=250,
            )
        else:
            st.info("No closed trades today.")


# ---- Main ----

render_dashboard()

time.sleep(REFRESH_SECONDS)
st.rerun()
