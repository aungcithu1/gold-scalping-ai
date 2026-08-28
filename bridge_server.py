from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Gold Scalping MT5 Bridge", version="0.5")
_lock = Lock()
_bars: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
_accounts: Dict[str, Dict[str, Any]] = {}
_signal_zones: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_commands: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_command_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
MAX_BARS_PER_ACCOUNT = 5000
ZONE_RETENTION_HOURS = 24
ZONE_FILE = Path("signal_zones_24h.json")


class MT5Packet(BaseModel):
    account_id: str
    account_mode: str = Field(pattern="^(REAL|DEMO|CONTEST|UNKNOWN)$")
    broker: str = "FxPro"
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    bid: float
    ask: float
    spread: float
    balance: Optional[float] = None
    equity: Optional[float] = None


class SignalZone(BaseModel):
    account_id: str
    side: str = Field(pattern="^(BUY|SELL)$")
    timestamp: datetime
    zone_low: float
    zone_high: float
    entry: float
    stop: float
    target: float
    score: float = 0.0
    reason: str = ""


class ManualOrder(BaseModel):
    account_id: str
    side: str = Field(pattern="^(BUY|SELL)$")
    symbol: str = "GOLD"
    volume: float = Field(gt=0)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    note: str = "Manual dashboard order"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_zones() -> None:
    if not ZONE_FILE.exists():
        return
    try:
        data = json.loads(ZONE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for account_id, rows in data.items():
                if isinstance(rows, list):
                    _signal_zones[str(account_id)] = rows
    except Exception:
        pass


def _save_zones() -> None:
    try:
        ZONE_FILE.write_text(json.dumps(dict(_signal_zones), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _store(packet: MT5Packet) -> None:
    row = packet.model_dump(mode="json")
    row["received_at"] = _now().isoformat()
    ts_key = row["timestamp"]
    account = _bars[packet.account_id]
    account[ts_key] = row
    if len(account) > MAX_BARS_PER_ACCOUNT:
        oldest = sorted(account.keys())[: len(account) - MAX_BARS_PER_ACCOUNT]
        for key in oldest:
            account.pop(key, None)
    _accounts[packet.account_id] = {
        "account_id": packet.account_id,
        "account_mode": packet.account_mode,
        "broker": packet.broker,
        "symbol": packet.symbol,
        "balance": packet.balance,
        "equity": packet.equity,
        "bars": len(account),
        "last_seen": row["received_at"],
    }


def _prune_zones(account_id: str) -> None:
    cutoff = _now() - timedelta(hours=ZONE_RETENTION_HOURS)
    kept = []
    for item in _signal_zones.get(account_id, []):
        try:
            ts = datetime.fromisoformat(str(item["timestamp"]).replace("Z", "+00:00"))
            if ts >= cutoff:
                kept.append(item)
        except Exception:
            pass
    if len(kept) != len(_signal_zones.get(account_id, [])):
        _signal_zones[account_id] = kept
        _save_zones()


_load_zones()
for _aid in list(_signal_zones.keys()):
    _prune_zones(_aid)


@app.get("/")
def root():
    return {"service": "Gold Scalping MT5 Bridge", "ok": True, "accounts": len(_accounts)}


@app.get("/health")
def health():
    with _lock:
        bars = sum(len(v) for v in _bars.values())
        zones = sum(len(v) for v in _signal_zones.values())
        return {"ok": True, "accounts": len(_accounts), "bars": bars, "zones_24h": zones}


@app.post("/ingest")
def ingest(packet: MT5Packet):
    with _lock:
        _store(packet)
    return {"ok": True, "stored": 1}


@app.post("/ingest_batch")
def ingest_batch(packets: List[MT5Packet]):
    if not packets:
        raise HTTPException(400, "empty batch")
    if len(packets) > 500:
        raise HTTPException(400, "batch limit is 500 packets")
    with _lock:
        for packet in packets:
            _store(packet)
    return {"ok": True, "stored": len(packets)}


@app.get("/accounts")
def accounts():
    with _lock:
        return list(_accounts.values())


@app.get("/bars/{account_id}")
def bars(account_id: str, limit: int = 1000):
    if limit < 1 or limit > MAX_BARS_PER_ACCOUNT:
        raise HTTPException(400, "limit must be 1..5000")
    with _lock:
        account = _bars.get(account_id, {})
        rows = [account[k] for k in sorted(account.keys())][-limit:]
    return rows


@app.get("/latest/{account_id}")
def latest(account_id: str):
    with _lock:
        account = _bars.get(account_id, {})
        if not account:
            raise HTTPException(404, "no data")
        return account[max(account.keys())]


@app.post("/signal_zone")
def signal_zone(zone: SignalZone):
    row = zone.model_dump(mode="json")
    with _lock:
        _prune_zones(zone.account_id)
        existing = _signal_zones[zone.account_id]
        minute_key = str(row["timestamp"])[:16]
        replaced = False
        for idx in range(len(existing) - 1, -1, -1):
            item = existing[idx]
            if item.get("side") == zone.side and str(item.get("timestamp", ""))[:16] == minute_key:
                existing[idx] = row
                replaced = True
                break
        if not replaced:
            existing.append(row)
        _save_zones()
    return {"ok": True}


@app.get("/signal_zones/{account_id}")
def signal_zones(account_id: str):
    with _lock:
        _prune_zones(account_id)
        return list(_signal_zones.get(account_id, []))


@app.post("/manual_order")
def manual_order(order: ManualOrder):
    if order.account_id not in _accounts:
        raise HTTPException(404, "account is not connected to bridge")
    cmd = order.model_dump()
    cmd["id"] = uuid4().hex[:12]
    cmd["created_at"] = _now().isoformat()
    cmd["status"] = "PENDING"
    with _lock:
        _commands[order.account_id].append(cmd)
    return {"ok": True, "command_id": cmd["id"], "status": "PENDING"}


@app.get("/command_text/{account_id}", response_class=PlainTextResponse)
def command_text(account_id: str):
    with _lock:
        pending = next((x for x in _commands.get(account_id, []) if x.get("status") == "PENDING"), None)
        if not pending:
            return "NONE"
        pending["status"] = "DELIVERED"
        return "{id}|{side}|{symbol}|{volume:.4f}|{sl:.8f}|{tp:.8f}".format(
            id=pending["id"], side=pending["side"], symbol=pending["symbol"], volume=pending["volume"],
            sl=pending["stop_loss"], tp=pending["take_profit"]
        )


@app.get("/command_result/{account_id}/{command_id}")
def command_result(account_id: str, command_id: str, ok: int, detail: str = ""):
    result = {"command_id": command_id, "account_id": account_id, "ok": bool(ok), "detail": detail[:300], "timestamp": _now().isoformat()}
    with _lock:
        for cmd in _commands.get(account_id, []):
            if cmd.get("id") == command_id:
                cmd["status"] = "EXECUTED" if ok else "FAILED"
                break
        _command_results[account_id].append(result)
        _command_results[account_id] = _command_results[account_id][-100:]
    return {"ok": True}


@app.get("/orders/{account_id}")
def orders(account_id: str):
    with _lock:
        return {"commands": list(_commands.get(account_id, []))[-50:], "results": list(_command_results.get(account_id, []))[-50:]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=8765, reload=False)
