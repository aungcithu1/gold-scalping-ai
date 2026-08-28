from __future__ import annotations
import io
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from trading_core import (
    Backtester,
    CsvFeed,
    NewsBlackout,
    RiskConfig,
    StrategyConfig,
    SignalEngine,
    ai_commentary,
    features,
    resample_m5,
)

load_dotenv()
st.set_page_config(page_title="Gold Scalping AI Lab", page_icon="🥇", layout="wide")
st.title("🥇 Gold Scalping AI Lab")
st.caption("FxPro MT5 + GOLD M1/M5 research dashboard. Supports REAL and DEMO account data through the MT5 bridge.")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    return str(value if value not in (None, "") else os.getenv(name, default))


def load_bridge(base_url: str, account_id: str, limit: int = 1500) -> pd.DataFrame:
    r = requests.get(f"{base_url.rstrip('/')}/bars/{account_id}", params={"limit": limit}, timeout=8)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError("MT5 bridge has no bars for this account yet")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "spread"]].copy()


def bridge_accounts(base_url: str):
    r = requests.get(f"{base_url.rstrip('/')}/accounts", timeout=5)
    r.raise_for_status()
    return r.json()


def read_uploaded_csv(uploaded, fallback: str) -> pd.DataFrame:
    if uploaded is None:
        return CsvFeed(fallback).load()
    tmp = Path(".uploaded_market.csv")
    tmp.write_bytes(uploaded.getvalue())
    return CsvFeed(str(tmp)).load()


def gold_chart(frame: pd.DataFrame):
    view = frame.tail(240).copy()
    view["direction"] = view["close"] >= view["open"]

    price_min = float(view[["low", "ema9", "ema21", "ema50"]].min().min())
    price_max = float(view[["high", "ema9", "ema21", "ema50"]].max().max())
    span = max(price_max - price_min, 0.5)
    pad = span * 0.08
    domain = [price_min - pad, price_max + pad]

    x = alt.X("timestamp:T", title="Time", axis=alt.Axis(format="%H:%M"))
    y_scale = alt.Scale(domain=domain, zero=False)

    wick = alt.Chart(view).mark_rule().encode(
        x=x,
        y=alt.Y("low:Q", scale=y_scale, title="GOLD price"),
        y2="high:Q",
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time"),
            alt.Tooltip("open:Q", format=".2f"),
            alt.Tooltip("high:Q", format=".2f"),
            alt.Tooltip("low:Q", format=".2f"),
            alt.Tooltip("close:Q", format=".2f"),
        ],
    )

    body = alt.Chart(view).mark_bar(size=3).encode(
        x=x,
        y=alt.Y("open:Q", scale=y_scale, title="GOLD price"),
        y2="close:Q",
        color=alt.condition("datum.direction", alt.value("#21c55d"), alt.value("#ef4444")),
    )

    ema_long = view[["timestamp", "ema9", "ema21", "ema50"]].melt(
        id_vars="timestamp", var_name="series", value_name="price"
    )
    emas = alt.Chart(ema_long).mark_line(strokeWidth=1.6).encode(
        x=x,
        y=alt.Y("price:Q", scale=y_scale, title="GOLD price"),
        color=alt.Color("series:N", title="EMA"),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time"),
            alt.Tooltip("series:N", title="Line"),
            alt.Tooltip("price:Q", format=".2f"),
        ],
    )

    return (wick + body + emas).properties(height=460).interactive(bind_y=False)


with st.sidebar:
    st.header("Data source")
    source = st.radio("Market data", ["Sample / Backtest", "Upload CSV", "FxPro MT5 Bridge"], index=0)
    bridge_url = secret("MT5_BRIDGE_URL", "http://127.0.0.1:8765")
    selected_account = None
    accounts = []
    market_upload = None

    if source == "Upload CSV":
        market_upload = st.file_uploader("Upload M1 market CSV", type=["csv"])
    elif source == "FxPro MT5 Bridge":
        st.caption(f"Bridge: {bridge_url}")
        try:
            accounts = bridge_accounts(bridge_url)
            labels = {
                f"{a['account_mode']} • {a['account_id']} • {a.get('symbol','GOLD')}": a["account_id"]
                for a in accounts
            }
            if labels:
                label = st.selectbox("MT5 account", list(labels.keys()))
                selected_account = labels[label]
                selected_meta = next((a for a in accounts if a["account_id"] == selected_account), None)
                if selected_meta:
                    st.caption(f"Bridge bars: {selected_meta.get('bars', 0)}")
            else:
                st.warning("Bridge connected, but no MT5 accounts are sending data yet.")
        except Exception as exc:
            st.warning(f"MT5 bridge not reachable: {exc}")

    st.header("Strategy")
    max_spread = st.number_input("Max spread ($)", min_value=0.01, max_value=10.0, value=0.60, step=0.05)
    vol_ratio = st.number_input("Volatility spike limit", min_value=1.0, max_value=5.0, value=1.80, step=0.05)
    sl_atr = st.number_input("Stop ATR", min_value=0.2, max_value=5.0, value=0.90, step=0.05)
    tp_atr = st.number_input("Target ATR", min_value=0.2, max_value=8.0, value=1.35, step=0.05)
    be_r = st.number_input("Breakeven trigger (R)", min_value=0.1, max_value=5.0, value=0.75, step=0.05)
    trail_start_r = st.number_input("Trailing starts (R)", min_value=0.1, max_value=5.0, value=1.00, step=0.05)
    trail_atr = st.number_input("Trailing distance (ATR)", min_value=0.1, max_value=5.0, value=0.60, step=0.05)

    st.header("Risk")
    starting_equity = st.number_input("Starting equity ($)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("Risk / trade (%)", min_value=0.01, max_value=5.0, value=0.25, step=0.05)
    max_daily_loss_pct = st.number_input("Daily loss kill-switch (%)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
    max_trades = st.number_input("Max trades / day", min_value=1, max_value=100, value=8, step=1)
    news_upload = st.file_uploader("News calendar CSV", type=["csv"])

scfg = StrategyConfig(max_spread=max_spread, vol_spike_ratio=vol_ratio, sl_atr=sl_atr, tp_atr=tp_atr,
                      breakeven_r=be_r, trailing_start_r=trail_start_r, trailing_atr=trail_atr)
rcfg = RiskConfig(starting_equity=starting_equity, risk_per_trade=risk_pct/100.0,
                  max_daily_loss=max_daily_loss_pct/100.0, max_trades_per_day=int(max_trades))

try:
    if source == "FxPro MT5 Bridge" and selected_account:
        raw = load_bridge(bridge_url, selected_account)
    elif source == "Upload CSV":
        raw = read_uploaded_csv(market_upload, "sample_xauusd.csv")
    else:
        raw = CsvFeed("sample_xauusd.csv").load()
except Exception as exc:
    st.error(f"Market data error: {exc}")
    st.stop()

if news_upload is None:
    news = NewsBlackout.from_csv("sample_news.csv", scfg.news_blackout_before_min, scfg.news_blackout_after_min)
else:
    news = NewsBlackout(pd.read_csv(io.BytesIO(news_upload.getvalue())), scfg.news_blackout_before_min, scfg.news_blackout_after_min)

try:
    trades, equity, metrics = Backtester(scfg, rcfg, news).run(raw)
    m1 = features(raw, scfg).dropna().reset_index(drop=True)
    m5 = features(resample_m5(raw), scfg).dropna().reset_index(drop=True)
    latest = SignalEngine(scfg, news).decide_at(m1, m5, len(m1)-1)
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Trades", metrics["trades"])
c2.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
pf = metrics["profit_factor"]
c3.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
c4.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
c5.metric("Net P/L", f"${metrics['net_pnl']:.2f}")

sig_col, risk_col = st.columns([2,1])
with sig_col:
    st.subheader("Latest paper signal")
    st.write({
        "side": latest.side,
        "score": round(latest.score,2),
        "M1": latest.m1_side,
        "M5": latest.m5_side,
        "entry": round(latest.entry,2),
        "stop": round(latest.stop,2),
        "target": round(latest.target,2),
        "reason": latest.reason,
        "filter": latest.filter_reason,
    })
with risk_col:
    st.subheader("Safety state")
    st.success("PAPER / DEMO research mode")
    st.info(f"Daily kill-switch: {max_daily_loss_pct:.2f}%\n\nMax trades/day: {int(max_trades)}")

chart_tab, trades_tab, equity_tab, ai_tab, format_tab = st.tabs(["📈 Trend chart", "📒 Trade journal", "💰 Equity", "🤖 AI context", "🧾 CSV format"])
with chart_tab:
    st.altair_chart(gold_chart(m1), use_container_width=True)
    last = m1.iloc[-1]
    st.caption(
        f"GOLD M1 candlesticks + EMA 9/21/50 • Last {last['close']:.2f} • "
        f"EMA9 {last['ema9']:.2f} • EMA21 {last['ema21']:.2f} • EMA50 {last['ema50']:.2f}. "
        "M5 confirmation uses completed 5-minute bars."
    )
with trades_tab:
    if trades.empty:
        st.warning("No trades in this sample/configuration.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button("Download journal CSV", trades.to_csv(index=False), "trade_journal.csv", "text/csv")
with equity_tab:
    if not equity.empty:
        st.line_chart(equity.set_index("timestamp")[["equity"]])
with ai_tab:
    if st.button("Generate AI risk commentary"):
        text = ai_commentary(m1, latest)
        if text:
            st.write(text)
        else:
            st.warning("OPENAI_API_KEY is not configured. The quantitative engine still works without it.")
with format_tab:
    st.code("timestamp,open,high,low,close,spread\n2026-08-27T08:00:00Z,4500.1,4501.2,4499.7,4500.8,0.25", language="text")
    st.code("timestamp,title,currency,impact\n2026-08-27T12:30:00Z,US GDP,USD,high", language="text")

st.divider()
st.caption("Backtests are estimates, not guarantees. This build uses MT5 data in analysis mode and does not place autonomous live-money orders.")