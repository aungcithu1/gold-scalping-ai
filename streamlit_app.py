from __future__ import annotations
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ctrader_oauth import CTraderConfig, authorization_url, exchange_code, refresh_access_token
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
st.caption("XAUUSD M1/M5 research, paper/demo analysis, risk controls and FxPro cTrader connection setup. Autonomous live-money execution is disabled.")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    return str(value if value not in (None, "") else os.getenv(name, default))


def read_uploaded_csv(uploaded, fallback: str) -> pd.DataFrame:
    if uploaded is None:
        return CsvFeed(fallback).load()
    data = uploaded.getvalue()
    tmp = Path(".uploaded_market.csv")
    tmp.write_bytes(data)
    return CsvFeed(str(tmp)).load()


with st.sidebar:
    st.header("Strategy")
    max_spread = st.number_input("Max spread ($)", min_value=0.01, max_value=10.0, value=0.60, step=0.05)
    vol_ratio = st.number_input("Volatility spike limit (ATR x median)", min_value=1.0, max_value=5.0, value=1.80, step=0.05)
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

    st.header("Data")
    market_upload = st.file_uploader("Upload M1 market CSV", type=["csv"])
    news_upload = st.file_uploader("Upload news calendar CSV", type=["csv"])

scfg = StrategyConfig(
    max_spread=max_spread,
    vol_spike_ratio=vol_ratio,
    sl_atr=sl_atr,
    tp_atr=tp_atr,
    breakeven_r=be_r,
    trailing_start_r=trail_start_r,
    trailing_atr=trail_atr,
)
rcfg = RiskConfig(
    starting_equity=starting_equity,
    risk_per_trade=risk_pct / 100.0,
    max_daily_loss=max_daily_loss_pct / 100.0,
    max_trades_per_day=int(max_trades),
)
raw = read_uploaded_csv(market_upload, "sample_xauusd.csv")
if news_upload is None:
    news = NewsBlackout.from_csv("sample_news.csv", scfg.news_blackout_before_min, scfg.news_blackout_after_min)
else:
    news_df = pd.read_csv(io.BytesIO(news_upload.getvalue()))
    news = NewsBlackout(news_df, scfg.news_blackout_before_min, scfg.news_blackout_after_min)

try:
    trades, equity, metrics = Backtester(scfg, rcfg, news).run(raw)
except Exception as exc:
    st.error(str(exc))
    st.stop()

m1 = features(raw, scfg).dropna().reset_index(drop=True)
m5 = features(resample_m5(raw), scfg).dropna().reset_index(drop=True)
latest = SignalEngine(scfg, news).decide_at(m1, m5, len(m1) - 1)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trades", metrics["trades"])
c2.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
pf = metrics["profit_factor"]
c3.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
c4.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
c5.metric("Net P/L", f"${metrics['net_pnl']:.2f}")

sig_col, risk_col = st.columns([2, 1])
with sig_col:
    st.subheader("Latest paper signal")
    st.write({
        "side": latest.side,
        "score": round(latest.score, 2),
        "M1": latest.m1_side,
        "M5": latest.m5_side,
        "entry": round(latest.entry, 2),
        "stop": round(latest.stop, 2),
        "target": round(latest.target, 2),
        "reason": latest.reason,
        "filter": latest.filter_reason,
    })
with risk_col:
    st.subheader("Safety state")
    st.success("PAPER / DEMO research mode")
    st.info(f"Daily kill-switch: {max_daily_loss_pct:.2f}%\n\nMax trades/day: {int(max_trades)}")

chart_tab, trades_tab, equity_tab, fxpro_tab, ai_tab, format_tab = st.tabs([
    "📈 Trend chart", "📒 Trade journal", "💰 Equity", "🔌 FxPro cTrader", "🤖 AI context", "🧾 CSV format"
])

with chart_tab:
    view = m1.tail(400).set_index("timestamp")[["close", "ema9", "ema21", "ema50"]]
    st.line_chart(view)
    st.caption("M1 close + EMA 9/21/50. M5 confirmation is calculated from completed 5-minute bars.")

with trades_tab:
    if trades.empty:
        st.warning("No trades in this sample/configuration.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button("Download journal CSV", trades.to_csv(index=False), "trade_journal.csv", "text/csv")

with equity_tab:
    if not equity.empty:
        st.line_chart(equity.set_index("timestamp")[["equity"]])

with fxpro_tab:
    st.subheader("FxPro cTrader connection")
    st.write("Use a FxPro **cTrader** trading account. FxPro Direct wallet credentials are not entered into this app.")

    cfg = CTraderConfig.from_values(
        secret("CTRADER_CLIENT_ID"),
        secret("CTRADER_CLIENT_SECRET"),
        secret("CTRADER_REDIRECT_URI", "https://act-gold-scalping-ai.streamlit.app/"),
    )
    env = secret("CTRADER_ENV", "demo").lower()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("OAuth config", "Ready" if cfg.ready else "Missing")
    col_b.metric("Environment", "DEMO" if env != "live" else "LIVE")
    col_c.metric("Permission", "READ ONLY")

    if not cfg.ready:
        st.warning("Add CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET and CTRADER_REDIRECT_URI in Streamlit App settings → Secrets.")
        st.code(
            'CTRADER_CLIENT_ID="..."\n'
            'CTRADER_CLIENT_SECRET="..."\n'
            'CTRADER_REDIRECT_URI="https://act-gold-scalping-ai.streamlit.app/"\n'
            'CTRADER_ENV="demo"',
            language="toml",
        )
        st.info("Register and approve an application in the cTrader Open API portal first, then add the exact Streamlit URL as its redirect URI.")
    else:
        auth_url = authorization_url(cfg, scope="accounts")
        st.link_button("Authorize FxPro cTrader (read-only)", auth_url, type="primary")
        st.caption("The app requests the cTrader 'accounts' scope only. It cannot place trades with this permission.")

        code = st.query_params.get("code")
        if code and "ctrader_tokens" not in st.session_state:
            try:
                with st.spinner("Exchanging cTrader authorization code..."):
                    st.session_state["ctrader_tokens"] = exchange_code(cfg, str(code))
                st.query_params.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"cTrader authorization failed: {exc}")

        tokens = st.session_state.get("ctrader_tokens")
        if tokens:
            st.success("cTrader OAuth connected for this browser session.")
            st.write({"token_type": tokens.get("tokenType", "bearer"), "expires_in_seconds": tokens.get("expiresIn")})
            if st.button("Refresh cTrader access token") and tokens.get("refreshToken"):
                try:
                    st.session_state["ctrader_tokens"] = refresh_access_token(cfg, tokens["refreshToken"])
                    st.success("Token refreshed.")
                except Exception as exc:
                    st.error(f"Refresh failed: {exc}")
            if st.button("Disconnect cTrader session"):
                st.session_state.pop("ctrader_tokens", None)
                st.rerun()
            st.info("Next connector stage: authenticate the cTrader trading account and subscribe to XAUUSD M1/live spot data. Keep CTRADER_ENV=demo while testing.")
        else:
            st.info("After authorization, cTrader redirects back here and the app exchanges the one-time code automatically.")

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
st.caption("Backtests are estimates, not guarantees. This build blocks autonomous live-money execution and uses read-only cTrader OAuth permission.")
