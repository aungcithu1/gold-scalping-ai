from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Gold Scalping MT5 Bridge", version="0.1")
_lock = Lock()
_bars: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=5000))
_accounts: dict[str, dict[str, Any]] = {}


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
    balance: float | None = None
    equity: float | None = None


@app.get("/health")
def health():
    return {"ok": True, "accounts": len(_accounts)}


@app.post("/ingest")
def ingest(packet: MT5Packet):
    row = packet.model_dump(mode="json")
    row["received_at"] = datetime.now(timezone.utc).isoformat()
    with _lock:
        _bars[packet.account_id].append(row)
        _accounts[packet.account_id] = {
            "account_id": packet.account_id,
            "account_mode": packet.account_mode,
            "broker": packet.broker,
            "symbol": packet.symbol,
            "balance": packet.balance,
            "equity": packet.equity,
            "last_seen": row["received_at"],
        }
    return {"ok": True}


@app.get("/accounts")
def accounts():
    with _lock:
        return list(_accounts.values())


@app.get("/bars/{account_id}")
def bars(account_id: str, limit: int = 1000):
    if limit < 1 or limit > 5000:
        raise HTTPException(400, "limit must be 1..5000")
    with _lock:
        rows = list(_bars.get(account_id, []))[-limit:]
    return rows


@app.get("/latest/{account_id}")
def latest(account_id: str):
    with _lock:
        rows = _bars.get(account_id)
        if not rows:
            raise HTTPException(404, "no data")
        return rows[-1]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=8765, reload=False)
