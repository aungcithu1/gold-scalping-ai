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
st.caption("FxPro MT5 + GOLD M1/M5 live signal dashboard. REAL and DEMO accounts are supported through the local MT5 bridge.")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    return str(value if value not in (None, "") else os.getenv(name, default))


def get_json(url: str, timeout: int = 5):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


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
    return get_json(f"{base_url.rstrip('/')}/accounts")


def bridge_zones(base_url: str, account_id: str) -> pd.DataFrame:
    try:
        rows = get_json(f"{base_url.rstrip('/')}/signal_zones/{account_id}")
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    z = pd.DataFrame(rows)
    z["timestamp"] = pd.to_datetime(z["timestamp"], utc=True)
    z["zone_end"] = z["timestamp"] + pd.Timedelta(minutes=15)
    return z


def save_signal_zone(base_url: str, account_id: str, sig) -> None:
    if sig.side not in ("BUY", "SELL"):
        return
    half = max(float(sig.atr) * 0.25, 0.10)
    payload = {
        "account_id": str(account_id),
        "side": sig.side,
        "timestamp": pd.Timestamp(sig.timestamp).isoformat(),
        "zone_low": float(sig.entry - half),
        "zone_high": float(sig.entry + half),
        "entry": float(sig.entry),
        "stop": float(sig.stop),
        "target": float(sig.target),
        "score": float(sig.score),
        "reason": str(sig.reason),
    }
    try:
        requests.post(f"{base_url.rstrip('/')}/signal_zone", json=payload, timeout=3).raise_for_status()
    except Exception:
        pass


def queue_manual_order(base_url: str, account_id: str, side: str, symbol: str, volume: float, sl: float, tp: float):
    payload = {
        "account_id": str(account_id),
        "side": side,
        "symbol": symbol,
        "volume": float(volume),
        "stop_loss": float(sl),
        "take_profit": float(tp),
        "note": "Human-confirmed Streamlit order",
    }
    r = requests.post(f"{base_url.rstrip('/')}/manual_order", json=payload, timeout=5)
    r.raise_for_status()
    return r.json()


def read_uploaded_csv(uploaded, fallback: str) -> pd.DataFrame:
    if uploaded is None:
        return CsvFeed(fallback).load()
    tmp = Path(".uploaded_market.csv")
    tmp.write_bytes(uploaded.getvalue())
    return CsvFeed(str(tmp)).load()


def confidence_pct(score: float) -> int:
    return int(max(0, min(100, round(abs(float(score)) / 2.5 * 100))))


def gold_chart(frame: pd.DataFrame, zones: pd.DataFrame):
    view = frame.tail(240).copy()
    view["direction"] = view["close"] >= view["open"]
    price_cols = ["low", "high", "ema9", "ema21", "ema50"]
    price_min = float(view[price_cols].min().min())
    price_max = float(view[price_cols].max().max())
    if not zones.empty:
        recent_cut = view["timestamp"].min()
        zv = zones[zones["timestamp"] >= recent_cut]
        if not zv.empty:
            price_min = min(price_min, float(zv["zone_low"].min()))
            price_max = max(price_max, float(zv["zone_high"].max()))
    span = max(price_max - price_min, 0.5)
    domain = [price_min - span * 0.08, price_max + span * 0.08]
    x = alt.X("timestamp:T", title="Time", axis=alt.Axis(format="%H:%M"))
    y_scale = alt.Scale(domain=domain, zero=False)

    layers = []
    if not zones.empty:
        zv = zones[zones["zone_end"] >= view["timestamp"].min()].copy()
        if not zv.empty:
            bands = alt.Chart(zv).mark_rect(opacity=0.14).encode(
                x=alt.X("timestamp:T", title="Time"), x2="zone_end:T",
                y=alt.Y("zone_low:Q", scale=y_scale, title="GOLD price"), y2="zone_high:Q",
                color=alt.Color("side:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["#22c55e", "#ef4444"]), legend=alt.Legend(title="24h AI zones")),
                tooltip=["side:N", alt.Tooltip("timestamp:T"), alt.Tooltip("zone_low:Q", format=".2f"), alt.Tooltip("zone_high:Q", format=".2f"), alt.Tooltip("score:Q", format=".2f")],
            )
            markers = alt.Chart(zv).mark_point(filled=True, size=80).encode(
                x="timestamp:T", y=alt.Y("entry:Q", scale=y_scale),
                shape=alt.Shape("side:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["triangle-up", "triangle-down"])),
                color=alt.Color("side:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["#22c55e", "#ef4444"]), legend=None),
                tooltip=["side:N", alt.Tooltip("entry:Q", format=".2f"), alt.Tooltip("stop:Q", format=".2f"), alt.Tooltip("target:Q", format=".2f")],
            )
            layers.extend([bands, markers])

    wick = alt.Chart(view).mark_rule().encode(
        x=x, y=alt.Y("low:Q", scale=y_scale, title="GOLD price"), y2="high:Q",
        tooltip=[alt.Tooltip("timestamp:T"), alt.Tooltip("open:Q", format=".2f"), alt.Tooltip("high:Q", format=".2f"), alt.Tooltip("low:Q", format=".2f"), alt.Tooltip("close:Q", format=".2f")],
    )
    body = alt.Chart(view).mark_bar(size=3).encode(
        x=x, y=alt.Y("open:Q", scale=y_scale, title="GOLD price"), y2="close:Q",
        color=alt.condition("datum.direction", alt.value("#21c55d"), alt.value("#ef4444")),
    )
    ema_long = view[["timestamp", "ema9", "ema21", "ema50"]].melt(id_vars="timestamp", var_name="series", value_name="price")
    emas = alt.Chart(ema_long).mark_line(strokeWidth=1.5).encode(
        x=x, y=alt.Y("price:Q", scale=y_scale, title="GOLD price"), color=alt.Color("series:N", title="EMA"),
        tooltip=[alt.Tooltip("timestamp:T"), "series:N", alt.Tooltip("price:Q", format=".2f")],
    )
    layers.extend([wick, body, emas])
    chart = layers[0]
    for layer in layers[1:]:
        chart = chart + layer
    return chart.properties(height=500).interactive(bind_y=False)


with st.sidebar:
    st.header("Data source")
    source = st.radio("Market data", ["Sample / Backtest", "Upload CSV", "FxPro MT5 Bridge"], index=2)
    bridge_url = secret("MT5_BRIDGE_URL", "http://127.0.0.1:8765")
    selected_account = None
    selected_meta = None
    market_upload = None
    if source == "Upload CSV":
        market_upload = st.file_uploader("Upload M1 market CSV", type=["csv"])
    elif source == "FxPro MT5 Bridge":
        st.caption(f"Bridge: {bridge_url}")
        try:
            accounts = bridge_accounts(bridge_url)
            labels = {f"{a['account_mode']} • {a['account_id']} • {a.get('symbol','GOLD')}": a["account_id"] for a in accounts}
            if labels:
                label = st.selectbox("MT5 account", list(labels.keys()))
                selected_account = labels[label]
                selected_meta = next((a for a in accounts if a["account_id"] == selected_account), None)
                if selected_meta:
                    st.caption(f"Bridge bars: {selected_meta.get('bars', 0)}")
            else:
                st.warning("Bridge connected, but no MT5 account is sending data.")
        except Exception as exc:
            st.warning(f"MT5 bridge not reachable: {exc}")

    st.header("Strategy")
    max_spread = st.number_input("Max spread ($)", 0.01, 10.0, 0.60, 0.05)
    vol_ratio = st.number_input("Volatility spike limit", 1.0, 5.0, 1.80, 0.05)
    sl_atr = st.number_input("Stop ATR", 0.2, 5.0, 0.90, 0.05)
    tp_atr = st.number_input("Target ATR", 0.2, 8.0, 1.35, 0.05)
    be_r = st.number_input("Breakeven trigger (R)", 0.1, 5.0, 0.75, 0.05)
    trail_start_r = st.number_input("Trailing starts (R)", 0.1, 5.0, 1.00, 0.05)
    trail_atr = st.number_input("Trailing distance (ATR)", 0.1, 5.0, 0.60, 0.05)
    st.header("Risk / backtest")
    starting_equity = st.number_input("Starting equity ($)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("Risk / trade (%)", 0.01, 5.0, 0.25, 0.05)
    max_daily_loss_pct = st.number_input("Daily loss kill-switch (%)", 0.1, 20.0, 1.0, 0.1)
    max_trades = st.number_input("Max trades / day", 1, 100, 8, 1)
    news_upload = st.file_uploader("News calendar CSV", type=["csv"])

scfg = StrategyConfig(max_spread=max_spread, vol_spike_ratio=vol_ratio, sl_atr=sl_atr, tp_atr=tp_atr, breakeven_r=be_r, trailing_start_r=trail_start_r, trailing_atr=trail_atr)
rcfg = RiskConfig(starting_equity=starting_equity, risk_per_trade=risk_pct/100.0, max_daily_loss=max_daily_loss_pct/100.0, max_trades_per_day=int(max_trades))
news = NewsBlackout.from_csv("sample_news.csv", scfg.news_blackout_before_min, scfg.news_blackout_after_min) if news_upload is None else NewsBlackout(pd.read_csv(io.BytesIO(news_upload.getvalue())), scfg.news_blackout_before_min, scfg.news_blackout_after_min)

try:
    if source == "FxPro MT5 Bridge" and selected_account:
        raw = load_bridge(bridge_url, selected_account)
    elif source == "Upload CSV":
        raw = read_uploaded_csv(market_upload, "sample_xauusd.csv")
    else:
        raw = CsvFeed("sample_xauusd.csv").load()
    trades, equity, metrics = Backtester(scfg, rcfg, news).run(raw)
    m1 = features(raw, scfg).dropna().reset_index(drop=True)
    m5 = features(resample_m5(raw), scfg).dropna().reset_index(drop=True)
    latest = SignalEngine(scfg, news).decide_at(m1, m5, len(m1)-1)
except Exception as exc:
    st.error(str(exc))
    st.stop()


@st.fragment(run_every="2s")
def live_panel():
    if source != "FxPro MT5 Bridge" or not selected_account:
        live_raw = raw
    else:
        try:
            live_raw = load_bridge(bridge_url, selected_account)
        except Exception as exc:
            st.warning(f"Live refresh error: {exc}")
            return

    try:
        live_m1 = features(live_raw, scfg).dropna().reset_index(drop=True)
        live_m5 = features(resample_m5(live_raw), scfg).dropna().reset_index(drop=True)
        live_sig = SignalEngine(scfg, news).decide_at(live_m1, live_m5, len(live_m1)-1)
    except Exception as exc:
        st.warning(f"Live signal error: {exc}")
        return

    if source == "FxPro MT5 Bridge" and selected_account:
        save_signal_zone(bridge_url, selected_account, live_sig)
        live_zones = bridge_zones(bridge_url, selected_account)
    else:
        live_zones = pd.DataFrame()

    conf = confidence_pct(live_sig.score)
    icon = "🟢" if live_sig.side == "BUY" else "🔴" if live_sig.side == "SELL" else "🟡"
    label = live_sig.side if live_sig.side in ("BUY", "SELL") else "WAIT"

    st.subheader("Live AI signal")
    a, b, c, d = st.columns([1.2, 1, 1, 1])
    a.metric("Signal", f"{icon} {label}")
    b.metric("Setup confidence", f"{conf}%")
    c.metric("M1 / M5", f"{live_sig.m1_side} / {live_sig.m5_side}")
    d.metric("Last GOLD", f"{float(live_m1.iloc[-1].close):.2f}")
    if live_sig.side in ("BUY", "SELL"):
        st.success(f"{live_sig.side} setup • Entry {live_sig.entry:.2f} • SL {live_sig.stop:.2f} • TP {live_sig.target:.2f} • {live_sig.reason}")
    else:
        st.info(f"WAIT • {live_sig.reason}" + (f" • {live_sig.filter_reason}" if live_sig.filter_reason else ""))

    st.altair_chart(gold_chart(live_m1, live_zones), use_container_width=True)
    last = live_m1.iloc[-1]
    st.caption(f"GOLD M1 + EMA 9/21/50 • Last {last['close']:.2f} • EMA9 {last['ema9']:.2f} • EMA21 {last['ema21']:.2f} • EMA50 {last['ema50']:.2f}. Only this live panel refreshes every 2 seconds.")


live_panel()

if source == "FxPro MT5 Bridge" and selected_account and selected_meta:
    st.subheader("Manual trade control")
    st.caption("BUY/SELL is sent only when you press the button. No automatic order is sent by the signal engine.")
    symbol = str(selected_meta.get("symbol", "GOLD"))
    with st.form("manual_trade_form", clear_on_submit=False):
        f1, f2, f3 = st.columns(3)
        volume = f1.number_input("Volume (lots)", min_value=0.01, value=0.01, step=0.01, format="%.2f")
        default_sl = float(latest.stop) if latest.side in ("BUY", "SELL") else 0.0
        default_tp = float(latest.target) if latest.side in ("BUY", "SELL") else 0.0
        manual_sl = f2.number_input("Stop Loss price (0 = none)", min_value=0.0, value=max(0.0, default_sl), step=0.10, format="%.2f")
        manual_tp = f3.number_input("Take Profit price (0 = none)", min_value=0.0, value=max(0.0, default_tp), step=0.10, format="%.2f")
        confirmed = st.checkbox(f"I confirm this manual order for {selected_meta.get('account_mode')} account {selected_account}")
        buy_col, sell_col = st.columns(2)
        buy_pressed = buy_col.form_submit_button("🟢 BUY MARKET", use_container_width=True, type="primary")
        sell_pressed = sell_col.form_submit_button("🔴 SELL MARKET", use_container_width=True)
        if buy_pressed or sell_pressed:
            if not confirmed:
                st.error("Confirm the account checkbox first.")
            else:
                side = "BUY" if buy_pressed else "SELL"
                try:
                    result = queue_manual_order(bridge_url, selected_account, side, symbol, volume, manual_sl, manual_tp)
                    st.success(f"{side} command queued. ID: {result.get('command_id')}. MT5 EA will pick it up on its next poll.")
                except Exception as exc:
                    st.error(f"Could not queue order: {exc}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Backtest trades", metrics["trades"])
c2.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
pf = metrics["profit_factor"]
c3.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
c4.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
c5.metric("Net P/L", f"${metrics['net_pnl']:.2f}")

zones_tab, orders_tab, trades_tab, equity_tab, ai_tab = st.tabs(["🟢🔴 24h zones", "🖱️ Manual orders", "📒 Backtest journal", "💰 Equity", "🤖 AI context"])
with zones_tab:
    zones = bridge_zones(bridge_url, selected_account) if source == "FxPro MT5 Bridge" and selected_account else pd.DataFrame()
    if zones.empty:
        st.info("No BUY/SELL zone has qualified yet in the current 24-hour record.")
    else:
        show = zones.sort_values("timestamp", ascending=False)[["timestamp", "side", "zone_low", "zone_high", "entry", "stop", "target", "score", "reason"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
with orders_tab:
    if source == "FxPro MT5 Bridge" and selected_account:
        try:
            history = get_json(f"{bridge_url.rstrip('/')}/orders/{selected_account}")
            st.write("Recent manual commands")
            st.dataframe(pd.DataFrame(history.get("commands", [])), use_container_width=True, hide_index=True)
            st.write("MT5 execution results")
            st.dataframe(pd.DataFrame(history.get("results", [])), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Order history unavailable: {exc}")
    else:
        st.info("Select FxPro MT5 Bridge to use manual orders.")
with trades_tab:
    if trades.empty:
        st.warning("No backtest trades in this configuration.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)
with equity_tab:
    if not equity.empty:
        st.line_chart(equity.set_index("timestamp")[["equity"]])
with ai_tab:
    if st.button("Generate AI market commentary"):
        text = ai_commentary(m1, latest)
        st.write(text if text else "OPENAI_API_KEY is not configured. The quantitative signal engine still runs without it.")

st.divider()
st.caption("Signals are model/rule outputs, not guarantees. Only the live signal/chart fragment refreshes every 2 seconds; the page and controls stay stable.")