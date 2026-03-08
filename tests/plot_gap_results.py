"""
Run AFTER backtest_gap_and_go.py to generate charts.
Reads gap_and_go_trades.csv and gap_and_go_equity.csv.

Usage:
    python -m tests.plot_gap_results
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

trades_df = pd.read_csv("gap_and_go_trades.csv")
equity    = pd.read_csv("gap_and_go_equity.csv", index_col=0, squeeze=True)
equity.index = pd.to_datetime(equity.index)

fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

COLOR_GREEN = "#00ff88"
COLOR_RED   = "#ff4444"
COLOR_BLUE  = "#4488ff"
COLOR_GOLD  = "#ffd700"
BG          = "#0d1117"
PANEL       = "#161b22"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#30363d")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")

# 1. Equity curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(equity.index, equity.values, color=COLOR_BLUE, linewidth=1.5)
ax1.fill_between(equity.index, equity.values, equity.values[0],
                 where=(equity.values >= equity.values[0]),
                 alpha=0.15, color=COLOR_GREEN)
ax1.fill_between(equity.index, equity.values, equity.values[0],
                 where=(equity.values < equity.values[0]),
                 alpha=0.15, color=COLOR_RED)
ax1.axhline(equity.values[0], color="#555", linestyle="--", linewidth=0.8)
style_ax(ax1, "Equity Curve — Gap-and-Go Strategy")
ax1.set_ylabel("Portfolio Value ($)")

# 2. Daily PnL
ax2 = fig.add_subplot(gs[1, 0])
daily_pnl = trades_df.groupby("date")["pnl"].sum()
colors    = [COLOR_GREEN if p > 0 else COLOR_RED for p in daily_pnl.values]
ax2.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, width=0.7)
ax2.axhline(0, color="#555", linewidth=0.8)
style_ax(ax2, "Daily PnL")
ax2.set_ylabel("PnL ($)")
ax2.set_xlabel("Trading Day")

# 3. PnL by symbol
ax3 = fig.add_subplot(gs[1, 1])
sym_pnl = trades_df.groupby("symbol")["pnl"].sum().sort_values()
cols    = [COLOR_GREEN if p > 0 else COLOR_RED for p in sym_pnl.values]
ax3.barh(sym_pnl.index, sym_pnl.values, color=cols)
ax3.axvline(0, color="#555", linewidth=0.8)
style_ax(ax3, "PnL by Symbol")
ax3.set_xlabel("Total PnL ($)")

# 4. Exit reason breakdown
ax4 = fig.add_subplot(gs[2, 0])
exit_counts = trades_df["exit_reason"].value_counts()
wedge_colors = [COLOR_GREEN, COLOR_BLUE, COLOR_GOLD, COLOR_RED][:len(exit_counts)]
ax4.pie(exit_counts.values, labels=exit_counts.index, colors=wedge_colors,
        autopct="%1.0f%%", textprops={"color": "white", "fontsize": 9},
        pctdistance=0.75)
style_ax(ax4, "Exit Reasons")

# 5. PnL distribution
ax5 = fig.add_subplot(gs[2, 1])
bins = np.linspace(trades_df["pnl"].min(), trades_df["pnl"].max(), 30)
ax5.hist(trades_df[trades_df["pnl"] > 0]["pnl"], bins=bins,
         color=COLOR_GREEN, alpha=0.7, label="Wins")
ax5.hist(trades_df[trades_df["pnl"] < 0]["pnl"], bins=bins,
         color=COLOR_RED, alpha=0.7, label="Losses")
ax5.axvline(0, color="white", linewidth=0.8)
ax5.legend(labelcolor="white", facecolor=PANEL, fontsize=8)
style_ax(ax5, "Trade PnL Distribution")
ax5.set_xlabel("PnL ($)")

fig.suptitle("QuantEdge — Gap-and-Go Backtest", color="white",
             fontsize=14, fontweight="bold", y=0.98)

plt.savefig("gap_and_go_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved: gap_and_go_results.png")
