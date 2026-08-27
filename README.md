# Gold Scalping AI Lab

A Streamlit research dashboard for XAUUSD M1/M5 scalping analysis, backtesting, paper/demo workflows, risk controls, and FxPro cTrader OAuth setup.

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
- FxPro cTrader read-only OAuth connection tab

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

## FxPro cTrader connection

FxPro Direct is the broker account-management portal. Do not enter your FxPro Direct password into this app.

The API path uses a FxPro **cTrader** trading account and cTrader ID (cTID). Register your own application in the cTrader Open API portal, wait for approval, and add this exact redirect URI to the application:

```text
https://act-gold-scalping-ai.streamlit.app/
```

Then add these values in Streamlit Community Cloud → App settings → Secrets:

```toml
CTRADER_CLIENT_ID="your-client-id"
CTRADER_CLIENT_SECRET="your-client-secret"
CTRADER_REDIRECT_URI="https://act-gold-scalping-ai.streamlit.app/"
CTRADER_ENV="demo"
```

The current app requests cTrader `accounts` scope only, so it is read-only. OAuth tokens are kept only in the active Streamlit browser session and are not committed to GitHub.

After secrets are configured, open the `FxPro cTrader` tab and choose `Authorize FxPro cTrader (read-only)`.

The next connector stage is authenticated cTrader account discovery plus XAUUSD spot/M1 data subscription. Keep `CTRADER_ENV="demo"` during development/testing.

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

Never commit `.env`, API keys, broker passwords, cTrader client secrets, access tokens, or refresh tokens.

For local use, copy `.env.example` to `.env`. For Streamlit Community Cloud, put secrets in the app's Secrets settings instead of GitHub.

Backtest results are estimates, not guarantees of future performance.
