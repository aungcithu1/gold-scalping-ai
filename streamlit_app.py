from __future__ import annotations
import io
import os
from pathlib import Path

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
st.caption("FxPro MT5 + XAUUSD M1/M5 research dashboard. Supports REAL and DEMO account data through the MT5 bridge.")


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
                f"{a['account_mode']} • {a['account_id']} • {a.get('symbol','XAUUSD')}": a["account_id"]
                for a in accounts
            }
            if labels:
                label = st.selectbox("MT5 account", list(labels.keys()))
                selected_account = labels[label]
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

sig_col, state_col = st.columns([2,1])
with sig_col:
    st.subheader("Latest signal")
    st.write({"side": latest.side, "score": round(latest.score,2), "M1": latest.m1_side, "M5": latest.m5_side,
              "entry": round(latest.entry,2), "stop": round(latest.stop,2), "target": round(latest.target,2),
              "reason": latest.reason, "filter": latest.filter_reason})
with state_col:
    st.subheader("Data state")
    st.success(source)
    if source == "FxPro MT5 Bridge" and selected_account:
        acct = next((a for a in accounts if str(a["account_id"]) == str(selected_account)), None)
        if acct:
            st.write({"mode": acct.get("account_mode"), "account": acct.get("account_id"), "last_seen": acct.get("last_seen")})

chart_tab, trades_tab, equity_tab, mt5_tab, ai_tab = st.tabs(["📈 Trend", "📒 Journal", "💰 Equity", "🔌 FxPro MT5", "🤖 AI"])
with chart_tab:
    st.line_chart(m1.tail(400).set_index("timestamp")[["close","ema9","ema21","ema50"]])
with trades_tab:
    st.dataframe(trades, use_container_width=True, hide_index=True) if not trades.empty else st.info("No trades in this window/configuration.")
with equity_tab:
    if not equity.empty: st.line_chart(equity.set_index("timestamp")[["equity"]])
with mt5_tab:
    st.subheader("FxPro MT5 Bridge")
    st.markdown("This build supports both **REAL** and **DEMO** MT5 accounts. The EA only sends market/account telemetry; it does not place orders.")
    st.code("python3 bridge_server.py", language="bash")
    st.markdown("In MT5: open MetaEditor, compile `mt5/GoldBridgeEA.mq5`, attach it to an XAUUSD chart, and allow the bridge URL under Expert Advisors → WebRequest allowed URLs.")
    st.code("http://127.0.0.1:8765", language="text")
    st.info("For Streamlit Cloud, set MT5_BRIDGE_URL to a publicly reachable HTTPS bridge endpoint. localhost works only when Streamlit runs on the same Mac.")
with ai_tab:
    if st.button("Generate AI risk commentary"):
        text = ai_commentary(m1, latest)
        st.write(text or "OPENAI_API_KEY is not configured.")

st.divider()
st.caption("Research / analysis build. REAL and DEMO data are kept separate by MT5 account ID and account mode.")
