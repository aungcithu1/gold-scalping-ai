from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    min_score: float = 1.8
    watch_score: float = 0.8
    sl_atr: float = 0.9
    tp_atr: float = 1.35
    max_spread: float = 0.60
    vol_spike_ratio: float = 1.80
    vol_median_window: int = 60
    london_start: int = 8
    london_end: int = 17
    ny_start: int = 8
    ny_end: int = 17
    news_blackout_before_min: int = 20
    news_blackout_after_min: int = 20
    breakeven_r: float = 0.75
    trailing_start_r: float = 1.0
    trailing_atr: float = 0.60


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.0025
    max_daily_loss: float = 0.01
    max_trades_per_day: int = 8
    starting_equity: float = 10000.0


@dataclass
class Signal:
    side: str
    score: float
    entry: float
    stop: float
    target: float
    atr: float
    timestamp: pd.Timestamp
    reason: str
    m1_side: str = "FLAT"
    m5_side: str = "FLAT"
    filters_passed: bool = False
    filter_reason: str = ""
    session: str = "OFF-SESSION"


@dataclass
class Trade:
    trade_id: int
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    initial_stop: float
    final_stop: float
    target: float
    size_units: float
    pnl: float
    r_multiple: float
    exit_reason: str
    score: float
    equity_after: float
    session: str = "UNKNOWN"


class CsvFeed:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        p = Path(self.path)
        if not p.exists():
            return synthetic_m1()
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        for c in ("open", "high", "low", "close"):
            if c not in df.columns:
                raise ValueError("CSV must contain timestamp, open, high, low, close and optional spread")
        if "timestamp" not in df.columns:
            end = pd.Timestamp.now(tz="UTC").floor("min")
            df["timestamp"] = pd.date_range(end=end, periods=len(df), freq="1min", tz="UTC")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "spread" not in df.columns:
            df["spread"] = 0.25
        return df.sort_values("timestamp").reset_index(drop=True)


def synthetic_m1(rows=1800, seed=7):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").floor("min")
    ts = pd.date_range(end=end, periods=rows, freq="1min", tz="UTC")
    noise = rng.normal(0, .42, rows)
    cycle = np.sin(np.linspace(0, 18 * np.pi, rows)) * .10
    close = 4500 + np.cumsum(noise + cycle)
    open_ = np.r_[close[0], close[:-1]]
    wick = rng.uniform(.08, .55, rows)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "spread": np.clip(rng.normal(.28, .09, rows), .08, .85),
    })


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n):
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low).abs(), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def features(df, cfg):
    x = df.copy()
    x["ema9"] = ema(x.close, cfg.ema_fast)
    x["ema21"] = ema(x.close, cfg.ema_mid)
    x["ema50"] = ema(x.close, cfg.ema_slow)
    x["rsi14"] = rsi(x.close, cfg.rsi_period)
    x["atr14"] = atr(x, cfg.atr_period)
    x["mom3"] = x.close.pct_change(3)
    x["atr_med"] = x.atr14.rolling(cfg.vol_median_window, min_periods=20).median()
    x["vol_ratio"] = x.atr14 / x.atr_med.replace(0, np.nan)
    rng = (x.high - x.low).replace(0, np.nan)
    x["body_ratio"] = (x.close - x.open).abs() / rng
    x["close_pos"] = (x.close - x.low) / rng
    x["prev_high20"] = x.high.shift(1).rolling(20, min_periods=10).max()
    x["prev_low20"] = x.low.shift(1).rolling(20, min_periods=10).min()
    x["breakout_up"] = x.close > x.prev_high20
    x["breakout_down"] = x.close < x.prev_low20
    x["ema_slope"] = x.ema21.diff(3)
    return x


def resample_m5(m1):
    return (m1.set_index("timestamp").resample("5min", label="right", closed="right")
            .agg({"open":"first", "high":"max", "low":"min", "close":"last", "spread":"mean"})
            .dropna().reset_index())


def _score(r):
    bull = 0.0
    bear = 0.0

    if r.ema9 > r.ema21 > r.ema50:
        bull += 1.15
    elif r.ema9 > r.ema21:
        bull += 0.45
    if r.ema9 < r.ema21 < r.ema50:
        bear += 1.15
    elif r.ema9 < r.ema21:
        bear += 0.45

    if 52 <= r.rsi14 <= 72:
        bull += 0.60
    elif 50 < r.rsi14 < 52:
        bull += 0.20
    if 28 <= r.rsi14 <= 48:
        bear += 0.60
    elif 48 < r.rsi14 < 50:
        bear += 0.20

    if r.mom3 > 0:
        bull += 0.45
    elif r.mom3 < 0:
        bear += 0.45

    if np.isfinite(r.get("ema_slope", np.nan)):
        if r.ema_slope > 0:
            bull += 0.20
        elif r.ema_slope < 0:
            bear += 0.20

    if bool(r.get("breakout_up", False)):
        bull += 0.35
    if bool(r.get("breakout_down", False)):
        bear += 0.35

    body = float(r.get("body_ratio", 0.0)) if np.isfinite(r.get("body_ratio", np.nan)) else 0.0
    close_pos = float(r.get("close_pos", 0.5)) if np.isfinite(r.get("close_pos", np.nan)) else 0.5
    if body >= 0.55 and close_pos >= 0.70 and r.close > r.open:
        bull += 0.25
    if body >= 0.55 and close_pos <= 0.30 and r.close < r.open:
        bear += 0.25

    return bull - bear


def _raw_side(r, min_score):
    score = _score(r)
    if score >= min_score:
        return "BUY", score, "EMA + RSI + momentum + price-action alignment"
    if score <= -min_score:
        return "SELL", score, "EMA + RSI + momentum + price-action alignment"
    return "FLAT", score, "Directional setup not fully confirmed"


def session_name(ts, cfg):
    lon = ts.tz_convert(ZoneInfo("Europe/London"))
    ny = ts.tz_convert(ZoneInfo("America/New_York"))
    l = cfg.london_start <= lon.hour < cfg.london_end
    n = cfg.ny_start <= ny.hour < cfg.ny_end
    if l and n:
        return "LONDON-NY OVERLAP"
    if l:
        return "LONDON"
    if n:
        return "NEW YORK"
    return "OFF-SESSION"


def session_allowed(ts, cfg):
    name = session_name(ts, cfg)
    return name != "OFF-SESSION", name


class NewsBlackout:
    def __init__(self, events=None, before_min=20, after_min=20):
        self.before = pd.Timedelta(minutes=before_min)
        self.after = pd.Timedelta(minutes=after_min)
        if events is None or events.empty:
            self.events = pd.DataFrame(columns=["timestamp", "title", "currency", "impact"])
        else:
            e = events.copy()
            e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
            e["impact"] = e.get("impact", "high").astype(str).str.lower()
            e["currency"] = e.get("currency", "USD").astype(str).str.upper()
            self.events = e

    @classmethod
    def from_csv(cls, path, before_min=20, after_min=20):
        return cls(pd.read_csv(path), before_min, after_min) if path and Path(path).exists() else cls(None, before_min, after_min)

    def blocked(self, ts):
        eligible = self.events[(self.events.impact == "high") & self.events.currency.isin(["USD", "XAU", "ALL"])] if not self.events.empty else self.events
        for _, e in eligible.iterrows():
            if e.timestamp - self.before <= ts <= e.timestamp + self.after:
                return True, f"News blackout: {e.get('title', 'high-impact event')}"
        return False, "News clear"


class SignalEngine:
    def __init__(self, cfg, news=None):
        self.cfg = cfg
        self.news = news or NewsBlackout()

    def decide_at(self, m1, m5, idx):
        r1 = m1.iloc[idx]
        ts = pd.Timestamp(r1.timestamp)
        px = float(r1.close)
        a = float(r1.atr14)
        score = float(_score(r1))
        sess = session_name(ts, self.cfg)
        prior = m5[m5.timestamp <= ts]

        if prior.empty:
            return Signal("WAIT", score, px, px, px, a, ts, "No completed M5 bar", session=sess)

        r5 = prior.iloc[-1]
        score5 = float(_score(r5))
        side1, _, reason = _raw_side(r1, self.cfg.min_score)
        side5, _, _ = _raw_side(r5, self.cfg.min_score)

        m1_bias = "BUY" if score >= self.cfg.watch_score else "SELL" if score <= -self.cfg.watch_score else "FLAT"
        m5_bias = "BUY" if score5 >= self.cfg.watch_score else "SELL" if score5 <= -self.cfg.watch_score else "FLAT"
        bias = m1_bias

        if bias == "BUY":
            stop = px - self.cfg.sl_atr * a
            target = px + self.cfg.tp_atr * a
        elif bias == "SELL":
            stop = px + self.cfg.sl_atr * a
            target = px - self.cfg.tp_atr * a
        else:
            stop = target = px

        if bias == "FLAT":
            return Signal("WAIT", score, px, stop, target, a, ts, "No useful directional edge", "FLAT", m5_bias, False, "Score below WATCH threshold", sess)

        watch_side = "WATCH " + bias

        if side1 == "FLAT":
            return Signal(watch_side, score, px, stop, target, a, ts, "M1 setup forming", bias, m5_bias, False, "M1 confirmation pending", sess)

        # Full M5 confirmation is ideal. A same-direction M5 WATCH bias is enough to keep a strong M1 setup visible,
        # but it remains WATCH until M5 reaches full confirmation.
        if side5 != side1:
            reason2 = "M5 bias aligned but confirmation pending" if m5_bias == side1 else "M5 direction not aligned"
            return Signal("WATCH " + side1, score, px, stop, target, a, ts, "Directional setup forming", side1, m5_bias, False, reason2, sess)

        if float(r1.spread) > self.cfg.max_spread:
            return Signal("WATCH " + side1, score, px, stop, target, a, ts, reason, side1, side5, False, "Spread too high", sess)
        if np.isfinite(r1.vol_ratio) and float(r1.vol_ratio) > self.cfg.vol_spike_ratio:
            return Signal("WATCH " + side1, score, px, stop, target, a, ts, reason, side1, side5, False, "Volatility spike", sess)
        blocked, why = self.news.blocked(ts)
        if blocked:
            return Signal("WATCH " + side1, score, px, stop, target, a, ts, reason, side1, side5, False, why, sess)

        ok, _ = session_allowed(ts, self.cfg)
        session_note = sess if ok else "OFF-SESSION: lower-quality context"
        return Signal(side1, score, px, stop, target, a, ts, f"{reason}; M5 confirmed", side1, side5, True, session_note, sess)


class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def size(self, equity, sig):
        d = abs(sig.entry - sig.stop)
        return 0.0 if d <= 0 else (equity * self.cfg.risk_per_trade) / d

    def day_locked(self, realized, day_start, count):
        if realized <= -(day_start * self.cfg.max_daily_loss):
            return True, "Daily loss kill-switch"
        if count >= self.cfg.max_trades_per_day:
            return True, "Max trades/day"
        return False, "OK"


def _max_drawdown_pct(eq):
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min() * -100)


def performance_metrics(trades, starting_equity):
    if trades.empty:
        return {"trades":0,"wins":0,"losses":0,"win_rate_pct":0.0,"net_pnl":0.0,"profit_factor":0.0,"max_drawdown_pct":0.0,"ending_equity":starting_equity,"avg_r":0.0}
    pnl = trades.pnl.astype(float)
    gp = float(pnl[pnl > 0].sum())
    gl = abs(float(pnl[pnl < 0].sum()))
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    eq = trades.equity_after.astype(float)
    return {
        "trades": len(trades),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "net_pnl": float(pnl.sum()),
        "profit_factor": pf,
        "max_drawdown_pct": _max_drawdown_pct(pd.concat([pd.Series([starting_equity]), eq], ignore_index=True)),
        "ending_equity": float(eq.iloc[-1]),
        "avg_r": float(trades.r_multiple.mean()),
    }


def session_performance(trades):
    cols = ["session", "trades", "win_rate_pct", "profit_factor", "avg_r", "net_pnl"]
    if trades is None or trades.empty or "session" not in trades.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for name, g in trades.groupby("session"):
        pnl = g.pnl.astype(float)
        gp = float(pnl[pnl > 0].sum())
        gl = abs(float(pnl[pnl < 0].sum()))
        pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        rows.append({
            "session": name,
            "trades": len(g),
            "win_rate_pct": float((pnl > 0).mean() * 100),
            "profit_factor": pf,
            "avg_r": float(g.r_multiple.mean()),
            "net_pnl": float(pnl.sum()),
        })
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False)


class Backtester:
    def __init__(self, scfg, rcfg, news=None):
        self.scfg = scfg
        self.rcfg = rcfg
        self.engine = SignalEngine(scfg, news)
        self.risk = RiskEngine(rcfg)

    def run(self, raw):
        m1 = features(raw, self.scfg).dropna().reset_index(drop=True)
        m5 = features(resample_m5(raw), self.scfg).dropna().reset_index(drop=True)
        if len(m1) < 80 or len(m5) < 20:
            raise ValueError("Not enough M1 data; provide at least about 300 bars")
        equity = self.rcfg.starting_equity
        open_trade = None
        trades = []
        rows = []
        current_day = None
        day_start = equity
        day_realized = 0.0
        day_count = 0
        tid = 0

        for i in range(1, len(m1)):
            b = m1.iloc[i]
            ts = pd.Timestamp(b.timestamp)
            if current_day != ts.date():
                current_day = ts.date()
                day_start = equity
                day_realized = 0.0
                day_count = 0

            if open_trade:
                side = open_trade["side"]
                entry = open_trade["entry"]
                initial_stop = open_trade["initial_stop"]
                initial_risk = abs(entry - initial_stop)
                favorable = float(b.high) - entry if side == "BUY" else entry - float(b.low)
                if favorable >= self.scfg.breakeven_r * initial_risk:
                    open_trade["stop"] = max(open_trade["stop"], entry) if side == "BUY" else min(open_trade["stop"], entry)
                if favorable >= self.scfg.trailing_start_r * initial_risk:
                    dist = self.scfg.trailing_atr * float(b.atr14)
                    cand = float(b.close) - dist if side == "BUY" else float(b.close) + dist
                    open_trade["stop"] = max(open_trade["stop"], cand) if side == "BUY" else min(open_trade["stop"], cand)
                stop = open_trade["stop"]
                target = open_trade["target"]
                exit_px = None
                exit_reason = None
                if side == "BUY":
                    if float(b.low) <= stop:
                        exit_px, exit_reason = stop, "STOP/TRAIL"
                    elif float(b.high) >= target:
                        exit_px, exit_reason = target, "TARGET"
                else:
                    if float(b.high) >= stop:
                        exit_px, exit_reason = stop, "STOP/TRAIL"
                    elif float(b.low) <= target:
                        exit_px, exit_reason = target, "TARGET"
                if exit_px is not None:
                    direction = 1 if side == "BUY" else -1
                    pnl = direction * (float(exit_px) - entry) * open_trade["size_units"]
                    equity += pnl
                    day_realized += pnl
                    day_count += 1
                    tid += 1
                    risk_cash = initial_risk * open_trade["size_units"]
                    trades.append(Trade(
                        tid, side, open_trade["entry_time"].isoformat(), ts.isoformat(), entry, float(exit_px),
                        initial_stop, float(stop), target, open_trade["size_units"], float(pnl),
                        float(pnl / risk_cash if risk_cash else 0), exit_reason, open_trade["score"], float(equity),
                        open_trade["session"],
                    ))
                    open_trade = None

            if open_trade is None and not self.risk.day_locked(day_realized, day_start, day_count)[0]:
                sig = self.engine.decide_at(m1, m5, i)
                if sig.side in ("BUY", "SELL") and sig.filters_passed:
                    size = self.risk.size(equity, sig)
                    if size > 0:
                        open_trade = {
                            "side": sig.side,
                            "entry": sig.entry,
                            "entry_time": sig.timestamp,
                            "initial_stop": sig.stop,
                            "stop": sig.stop,
                            "target": sig.target,
                            "size_units": size,
                            "score": sig.score,
                            "session": sig.session,
                        }
            rows.append({"timestamp": ts, "equity": equity, "day_realized_pnl": day_realized})

        tdf = pd.DataFrame([asdict(t) for t in trades])
        edf = pd.DataFrame(rows)
        return tdf, edf, performance_metrics(tdf, self.rcfg.starting_equity)


def ai_commentary(df, signal):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        r = df.iloc[-1]
        prompt = (
            "Summarize this XAUUSD manual-trading setup in <=120 words. Discuss uncertainty, invalidation, spread/news/volatility risk. Do not execute trades. "
            f"close={r.close:.2f}, ema9={r.ema9:.2f}, ema21={r.ema21:.2f}, ema50={r.ema50:.2f}, "
            f"rsi={r.rsi14:.1f}, atr={r.atr14:.2f}, side={signal.side}, m1={signal.m1_side}, m5={signal.m5_side}, session={signal.session}."
        )
        resp = OpenAI(api_key=key).responses.create(model=os.getenv("OPENAI_MODEL", "gpt-5-mini"), input=prompt)
        return resp.output_text
    except Exception as e:
        return f"AI commentary unavailable: {type(e).__name__}"
