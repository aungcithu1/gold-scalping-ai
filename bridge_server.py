from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Gold Scalping MT5 Bridge", version="0.2")
_lock = Lock()
# account_id -> timestamp -> packet. Timestamp keys make live updates idempotent.
_bars: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
_accounts: Dict[str, Dict[str, Any]] = {}
MAX_BARS_PER_ACCOUNT = 5000


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


def _store(packet: MT5Packet) -> None:
    row = packet.model_dump(mode="json")
    row["received_at"] = datetime.now(timezone.utc).isoformat()
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


@app.get("/")
def root():
    return {"service": "Gold Scalping MT5 Bridge", "ok": True, "accounts": len(_accounts)}


@app.get("/health")
def health():
    with _lock:
        bars = sum(len(v) for v in _bars.values())
        return {"ok": True, "accounts": len(_accounts), "bars": bars}


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
        key = max(account.keys())
        return account[key]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=8765, reload=False)
