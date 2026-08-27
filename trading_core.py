from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from zoneinfo import ZoneInfo
import math
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
    sl_atr: float = 0.9
    tp_atr: float = 1.35
    max_spread: float = 0.60
    vol_spike_ratio: float = 1.80
    vol_median_window: int = 60
    london_start: int = 8
    london_end: int = 12
    ny_start: int = 8
    ny_end: int = 12
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


def synthetic_m1(rows: int = 1800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").floor("min")
    ts = pd.date_range(end=end, periods=rows, freq="1min", tz="UTC")
    noise = rng.normal(0, 0.42, rows)
    cycle = np.sin(np.linspace(0, 18 * np.pi, rows)) * 0.10
    close = 4500 + np.cumsum(noise + cycle)
    open_ = np.r_[close[0], close[:-1]]
    wick = rng.uniform(0.08, 0.55, rows)
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    spread = np.clip(rng.normal(0.28, 0.09, rows), 0.08, 0.85)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "spread": spread})


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([(df.high - df.low).abs(), (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    x = df.copy()
    x["ema9"] = ema(x.close, cfg.ema_fast)
    x["ema21"] = ema(x.close, cfg.ema_mid)
    x["ema50"] = ema(x.close, cfg.ema_slow)
    x["rsi14"] = rsi(x.close, cfg.rsi_period)
    x["atr14"] = atr(x, cfg.atr_period)
    x["mom3"] = x.close.pct_change(3)
    x["atr_med"] = x.atr14.rolling(cfg.vol_median_window, min_periods=20).median()
    x["vol_ratio"] = x.atr14 / x.atr_med.replace(0, np.nan)
    return x


def resample_m5(m1: pd.DataFrame) -> pd.DataFrame:
    return (m1.set_index("timestamp").resample("5min", label="right", closed="right")
            .agg({"open":"first","high":"max","low":"min","close":"last","spread":"mean"})
            .dropna().reset_index())


def _raw_side(r: pd.Series, min_score: float) -> tuple[str, float, str]:
    bull = (1.2 if r.ema9 > r.ema21 > r.ema50 else 0) + (0.7 if 52 <= r.rsi14 <= 70 else 0) + (0.6 if r.mom3 > 0 else 0)
    bear = (1.2 if r.ema9 < r.ema21 < r.ema50 else 0) + (0.7 if 30 <= r.rsi14 <= 48 else 0) + (0.6 if r.mom3 < 0 else 0)
    score = bull - bear
    if score >= min_score:
        return "BUY", score, "EMA trend + RSI/momentum alignment"
    if score <= -min_score:
        return "SELL", score, "EMA trend + RSI/momentum alignment"
    return "FLAT", score, "No high-conviction alignment"


def session_allowed(ts: pd.Timestamp, cfg: StrategyConfig) -> tuple[bool, str]:
    lon = ts.tz_convert(ZoneInfo("Europe/London")).hour
    ny = ts.tz_convert(ZoneInfo("America/New_York")).hour
    hit = []
    if cfg.london_start <= lon <= cfg.london_end:
        hit.append("London")
    if cfg.ny_start <= ny <= cfg.ny_end:
        hit.append("New York")
    return (True, "+".join(hit)) if hit else (False, "Outside London/New York windows")


class NewsBlackout:
    def __init__(self, events: pd.DataFrame | None = None, before_min: int = 20, after_min: int = 20):
        self.before = pd.Timedelta(minutes=before_min)
        self.after = pd.Timedelta(minutes=after_min)
        if events is None or events.empty:
            self.events = pd.DataFrame(columns=["timestamp","title","currency","impact"])
        else:
            e = events.copy()
            e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
            e["impact"] = e.get("impact", "high").astype(str).str.lower()
            e["currency"] = e.get("currency", "USD").astype(str).str.upper()
            self.events = e

    @classmethod
    def from_csv(cls, path: str | None, before_min: int = 20, after_min: int = 20):
        return cls(pd.read_csv(path), before_min, after_min) if path and Path(path).exists() else cls(None, before_min, after_min)

    def blocked(self, ts: pd.Timestamp) -> tuple[bool, str]:
        eligible = self.events[(self.events.impact == "high") & self.events.currency.isin(["USD","XAU","ALL"])] if not self.events.empty else self.events
        for _, e in eligible.iterrows():
            if e.timestamp - self.before <= ts <= e.timestamp + self.after:
                return True, f"News blackout: {e.get('title','high-impact event')}"
        return False, "No high-impact event in blackout window"


class SignalEngine:
    def __init__(self, cfg: StrategyConfig, news: NewsBlackout | None = None):
        self.cfg = cfg
        self.news = news or NewsBlackout()

    def decide_at(self, m1: pd.DataFrame, m5: pd.DataFrame, idx: int) -> Signal:
        r1 = m1.iloc[idx]
        ts, px, a = pd.Timestamp(r1.timestamp), float(r1.close), float(r1.atr14)
        side1, score, reason = _raw_side(r1, self.cfg.min_score)
        prior = m5[m5.timestamp <= ts]
        if prior.empty:
            return Signal("FLAT", score, px, px, px, a, ts, "No completed M5 bar")
        side5, _, _ = _raw_side(prior.iloc[-1], self.cfg.min_score)
        if side1 == "FLAT" or side5 != side1:
            return Signal("FLAT", score, px, px, px, a, ts, f"M1/M5 not aligned ({side1}/{side5})", side1, side5)
        ok, why = session_allowed(ts, self.cfg)
        if not ok:
            return Signal("FLAT", score, px, px, px, a, ts, reason, side1, side5, False, why)
        if float(r1.spread) > self.cfg.max_spread:
            return Signal("FLAT", score, px, px, px, a, ts, reason, side1, side5, False, "Spread filter")
        if np.isfinite(r1.vol_ratio) and float(r1.vol_ratio) > self.cfg.vol_spike_ratio:
            return Signal("FLAT", score, px, px, px, a, ts, reason, side1, side5, False, "Volatility-spike filter")
        blocked, why = self.news.blocked(ts)
        if blocked:
            return Signal("FLAT", score, px, px, px, a, ts, reason, side1, side5, False, why)
        stop = px - self.cfg.sl_atr * a if side1 == "BUY" else px + self.cfg.sl_atr * a
        target = px + self.cfg.tp_atr * a if side1 == "BUY" else px - self.cfg.tp_atr * a
        return Signal(side1, score, px, stop, target, a, ts, f"{reason}; M5 confirmed; {why}", side1, side5, True, "All filters passed")


class RiskEngine:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size(self, equity: float, sig: Signal) -> float:
        distance = abs(sig.entry - sig.stop)
        return 0.0 if distance <= 0 else (equity * self.cfg.risk_per_trade) / distance

    def day_locked(self, realized: float, day_start: float, count: int) -> tuple[bool, str]:
        if realized <= -(day_start * self.cfg.max_daily_loss):
            return True, "Daily loss kill-switch"
        if count >= self.cfg.max_trades_per_day:
            return True, "Max trades/day"
        return False, "OK"


def _max_drawdown_pct(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min() * -100)


def performance_metrics(trades: pd.DataFrame, starting_equity: float) -> dict:
    if trades.empty:
        return {"trades":0,"wins":0,"losses":0,"win_rate_pct":0.0,"net_pnl":0.0,"profit_factor":0.0,"max_drawdown_pct":0.0,"ending_equity":starting_equity,"avg_r":0.0}
    pnl = trades.pnl.astype(float)
    gp = float(pnl[pnl > 0].sum())
    gl = abs(float(pnl[pnl < 0].sum()))
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    eq = trades.equity_after.astype(float)
    return {"trades":len(trades),"wins":int((pnl>0).sum()),"losses":int((pnl<0).sum()),"win_rate_pct":float((pnl>0).mean()*100),"net_pnl":float(pnl.sum()),"profit_factor":pf,"max_drawdown_pct":_max_drawdown_pct(pd.concat([pd.Series([starting_equity]),eq],ignore_index=True)),"ending_equity":float(eq.iloc[-1]),"avg_r":float(trades.r_multiple.mean())}


class Backtester:
    def __init__(self, scfg: StrategyConfig, rcfg: RiskConfig, news: NewsBlackout | None = None):
        self.scfg, self.rcfg = scfg, rcfg
        self.engine = SignalEngine(scfg, news)
        self.risk = RiskEngine(rcfg)

    def run(self, raw: pd.DataFrame):
        m1 = features(raw, self.scfg).dropna().reset_index(drop=True)
        m5 = features(resample_m5(raw), self.scfg).dropna().reset_index(drop=True)
        if len(m1) < 80 or len(m5) < 20:
            raise ValueError("Not enough M1 data; provide at least about 300 bars")
        equity, open_trade, trades, rows = self.rcfg.starting_equity, None, [], []
        current_day, day_start, day_realized, day_count, tid = None, equity, 0.0, 0, 0
        for i in range(1, len(m1)):
            b, ts = m1.iloc[i], pd.Timestamp(m1.iloc[i].timestamp)
            if current_day != ts.date():
                current_day, day_start, day_realized, day_count = ts.date(), equity, 0.0, 0
            if open_trade:
                side, entry, initial_stop = open_trade["side"], open_trade["entry"], open_trade["initial_stop"]
                initial_risk = abs(entry - initial_stop)
                favorable = float(b.high) - entry if side == "BUY" else entry - float(b.low)
                if favorable >= self.scfg.breakeven_r * initial_risk:
                    open_trade["stop"] = max(open_trade["stop"], entry) if side == "BUY" else min(open_trade["stop"], entry)
                if favorable >= self.scfg.trailing_start_r * initial_risk:
                    dist = self.scfg.trailing_atr * float(b.atr14)
                    cand = float(b.close) - dist if side == "BUY" else float(b.close) + dist
                    open_trade["stop"] = max(open_trade["stop"], cand) if side == "BUY" else min(open_trade["stop"], cand)
                stop, target, exit_px, exit_reason = open_trade["stop"], open_trade["target"], None, None
                if side == "BUY":
                    if float(b.low) <= stop: exit_px, exit_reason = stop, "STOP/TRAIL"
                    elif float(b.high) >= target: exit_px, exit_reason = target, "TARGET"
                else:
                    if float(b.high) >= stop: exit_px, exit_reason = stop, "STOP/TRAIL"
                    elif float(b.low) <= target: exit_px, exit_reason = target, "TARGET"
                if exit_px is not None:
                    direction = 1 if side == "BUY" else -1
                    pnl = direction * (float(exit_px) - entry) * open_trade["size_units"]
                    equity += pnl; day_realized += pnl; day_count += 1; tid += 1
                    risk_cash = initial_risk * open_trade["size_units"]
                    trades.append(Trade(tid, side, open_trade["entry_time"].isoformat(), ts.isoformat(), entry, float(exit_px), initial_stop, float(stop), target, open_trade["size_units"], float(pnl), float(pnl/risk_cash if risk_cash else 0), exit_reason, open_trade["score"], float(equity)))
                    open_trade = None
            if open_trade is None and not self.risk.day_locked(day_realized, day_start, day_count)[0]:
                sig = self.engine.decide_at(m1, m5, i)
                if sig.side in ("BUY","SELL") and sig.filters_passed:
                    size = self.risk.size(equity, sig)
                    if size > 0:
                        open_trade = {"side":sig.side,"entry":sig.entry,"entry_time":sig.timestamp,"initial_stop":sig.stop,"stop":sig.stop,"target":sig.target,"size_units":size,"score":sig.score}
            rows.append({"timestamp":ts,"equity":equity,"day_realized_pnl":day_realized})
        tdf = pd.DataFrame([asdict(t) for t in trades])
        edf = pd.DataFrame(rows)
        return tdf, edf, performance_metrics(tdf, self.rcfg.starting_equity)


def ai_commentary(df: pd.DataFrame, signal: Signal) -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        r = df.iloc[-1]
        prompt = ("Summarize this XAUUSD paper-trading setup in <=120 words. Discuss uncertainty, invalidation, spread/news/volatility risk. Do not execute or recommend autonomous live-money trading. "
                  f"close={r.close:.2f}, ema9={r.ema9:.2f}, ema21={r.ema21:.2f}, ema50={r.ema50:.2f}, rsi={r.rsi14:.1f}, atr={r.atr14:.2f}, side={signal.side}, m1={signal.m1_side}, m5={signal.m5_side}.")
        resp = OpenAI(api_key=key).responses.create(model=os.getenv("OPENAI_MODEL","gpt-5-mini"), input=prompt)
        return resp.output_text
    except Exception as e:
        return f"AI commentary unavailable: {type(e).__name__}"
