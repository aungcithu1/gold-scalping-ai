from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

AUTH_BASE = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
TOKEN_URL = "https://openapi.ctrader.com/apps/token"


@dataclass
class CTraderConfig:
    client_id: str
    client_secret: str
    redirect_uri: str

    @classmethod
    def from_values(cls, client_id: str | None, client_secret: str | None, redirect_uri: str | None):
        return cls((client_id or "").strip(), (client_secret or "").strip(), (redirect_uri or "").strip())

    @property
    def ready(self) -> bool:
        return bool(self.client_id and self.client_secret and self.redirect_uri)


def authorization_url(cfg: CTraderConfig, scope: str = "accounts") -> str:
    if not cfg.client_id or not cfg.redirect_uri:
        raise ValueError("CTRADER_CLIENT_ID and CTRADER_REDIRECT_URI are required")
    if scope not in {"accounts", "trading"}:
        raise ValueError("scope must be 'accounts' or 'trading'")
    params = {
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "scope": scope,
        "product": "web",
    }
    return f"{AUTH_BASE}?{urlencode(params)}"


def exchange_code(cfg: CTraderConfig, code: str, timeout: int = 15) -> dict:
    if not cfg.ready:
        raise ValueError("cTrader OAuth settings are incomplete")
    params = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": cfg.redirect_uri,
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
    }
    response = requests.get(TOKEN_URL, params=params, timeout=timeout, headers={"Accept": "application/json"})
    response.raise_for_status()
    payload = response.json()
    if payload.get("errorCode"):
        raise RuntimeError(payload.get("description") or payload["errorCode"])
    return payload


def refresh_access_token(cfg: CTraderConfig, refresh_token: str, timeout: int = 15) -> dict:
    if not cfg.client_id or not cfg.client_secret:
        raise ValueError("cTrader client credentials are incomplete")
    params = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
    }
    response = requests.post(TOKEN_URL, params=params, timeout=timeout, headers={"Accept": "application/json"})
    response.raise_for_status()
    payload = response.json()
    if payload.get("errorCode"):
        raise RuntimeError(payload.get("description") or payload["errorCode"])
    return payload
