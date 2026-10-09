"""API 입출력 스키마 (pydantic)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HeartbeatIn(BaseModel):
    device_ts: datetime
    status: str = "ok"
    fps: float = 0.0
    queue_len: int = 0
    temp_c: float | None = None
    model_version: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class CountsIn(BaseModel):
    date: str                      # YYYY-MM-DD
    processed: int
    cracked: int
    severity: dict[str, int] = Field(default_factory=dict)  # {"1": n, "2": n, "3": n}
    hourly: list[dict[str, Any]] = Field(default_factory=list)  # [{"hour_ts": "YYYY-MM-DDTHH", "processed": n, "cracked": n}]


class EventIn(BaseModel):
    device_ts: datetime
    type: str
    code: str = ""
    message: str = ""
    dedup_key: str | None = None


class EggLogIn(BaseModel):
    egg_uid: str
    device_ts: datetime
    cam: int = 0
    roi: int = 0
    scores: list[float] = Field(default_factory=list)
    fused_score: float = 0.0
    reject: bool = False
    severity: int = 0


class DeviceCreate(BaseModel):
    device_id: int
    name: str = ""
    site: str = ""


class ConfigUpdate(BaseModel):
    config: dict[str, Any]
