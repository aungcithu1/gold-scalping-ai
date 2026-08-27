# Gold Scalping AI Lab

A Streamlit research dashboard for XAUUSD M1/M5 scalping analysis, backtesting, paper/demo workflows, and risk controls.

## Included

- M1 trigger + M5 confirmation
- EMA 9/21/50, RSI, ATR, momentum
- London / New York session filter
- spread filter
- ATR volatility-spike filter
- high-impact USD/XAU news blackout
- breakeven logic
- ATR trailing stop
- max daily-loss kill switch
- max trades/day
- trade journal
- win rate, profit factor, max drawdown, net P/L
- equity and trend charts
- optional OpenAI risk commentary

Autonomous real-money execution is intentionally disabled. Use this project for research, backtesting, paper/demo trading, or a separate human-approved execution workflow.

## Run on macOS

```bash
git clone https://github.com/aungcithu1/gold-scalping-ai.git
cd gold-scalping-ai
chmod +x run_web.sh
./run_web.sh
```

Then open `http://localhost:8501`.

## Streamlit Community Cloud

Deploy this repository with:

- Repository: `aungcithu1/gold-scalping-ai`
- Branch: `main`
- Main file: `streamlit_app.py`

The app generates synthetic M1 data automatically when no CSV is uploaded, so the cloud deployment works without a large sample market file.

## CSV format

Market data:

```text
timestamp,open,high,low,close,spread
2026-08-27T08:00:00Z,4500.10,4501.20,4499.70,4500.80,0.25
```

News calendar:

```text
timestamp,title,currency,impact
2026-08-27T12:30:00Z,US GDP,USD,high
```

## Secrets

Never commit `.env`, API keys, broker passwords, cTrader client secrets, or access tokens.

For local use, copy `.env.example` to `.env`. For Streamlit Community Cloud, put secrets in the app's Secrets settings instead of GitHub.

## FxPro / cTrader next stage

The intended integration path is cTrader Open API for authorized market data and demo/paper workflows. Keep broker credentials outside the repository and use OAuth/token-based authorization.

Backtest results are estimates, not guarantees of future performance.
