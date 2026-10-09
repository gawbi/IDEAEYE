"""장비 쪽 업로더.

- Aggregator: 계란 판정(EggDecision 또는 dict)을 받아 일별/시간별 집계를 유지
- LocalQueue: SQLite 기반 전송 큐 (오프라인 시 보관, 멱등 키, 재시도 횟수)
- Uploader: 큐를 순서대로 서버에 전송(실패 시 지수 백오프), 하트비트, 설정 폴링
- Transport: HTTP(기본). MQTT 는 같은 인터페이스로 교체 (LTE Cat.M1 모듈 채택 시)
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------- 집계


@dataclass
class DayStats:
    date: str
    processed: int = 0
    cracked: int = 0
    severity: dict[str, int] = field(default_factory=lambda: {"1": 0, "2": 0, "3": 0})
    hourly: dict[str, dict[str, int]] = field(default_factory=dict)  # hour_ts → {processed, cracked}

    def payload(self) -> dict[str, Any]:
        return {"date": self.date, "processed": self.processed, "cracked": self.cracked, "severity": dict(self.severity),
                "hourly": [{"hour_ts": k, **v} for k, v in sorted(self.hourly.items())]}


class Aggregator:
    def __init__(self):
        self.days: dict[str, DayStats] = {}

    def add(self, reject: bool, severity: int = 0, ts: datetime | None = None) -> DayStats:
        ts = ts or datetime.now()
        d = ts.strftime("%Y-%m-%d")
        h = ts.strftime("%Y-%m-%dT%H")
        st = self.days.setdefault(d, DayStats(d))
        st.processed += 1
        hr = st.hourly.setdefault(h, {"processed": 0, "cracked": 0})
        hr["processed"] += 1
        if reject:
            st.cracked += 1
            hr["cracked"] += 1
            if str(severity) in st.severity:
                st.severity[str(severity)] += 1
        return st


# ---------------------------------------------------------------- 로컬 큐


class LocalQueue:
    """SQLite 전송 큐. kind: counts | events | eggs | heartbeat. key 가 같으면 덮어씀(집계 upsert)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, key TEXT, payload TEXT, created REAL, attempts INTEGER DEFAULT 0)")
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_kind_key ON queue(kind, key)")
        self.conn.commit()

    def put(self, kind: str, key: str, payload: Any) -> None:
        self.conn.execute("INSERT INTO queue(kind, key, payload, created, attempts) VALUES (?,?,?,?,0) ON CONFLICT(kind, key) DO UPDATE SET payload=excluded.payload, created=excluded.created",
                          (kind, key, json.dumps(payload, ensure_ascii=False, default=str), time.time()))
        self.conn.commit()

    def peek(self, limit: int = 20) -> list[tuple[int, str, str, Any, int]]:
        rows = self.conn.execute("SELECT id, kind, key, payload, attempts FROM queue ORDER BY id LIMIT ?", (limit,)).fetchall()
        return [(r[0], r[1], r[2], json.loads(r[3]), r[4]) for r in rows]

    def ack(self, row_id: int) -> None:
        self.conn.execute("DELETE FROM queue WHERE id=?", (row_id,))
        self.conn.commit()

    def fail(self, row_id: int) -> None:
        self.conn.execute("UPDATE queue SET attempts=attempts+1 WHERE id=?", (row_id,))
        self.conn.commit()

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0])


# ---------------------------------------------------------------- 전송


class Transport(Protocol):
    def post(self, path: str, payload: Any) -> bool: ...
    def get(self, path: str) -> Any | None: ...


class HttpTransport:
    def __init__(self, base_url: str, device_id: int, api_key: str, timeout: float = 10.0):
        import httpx

        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self.device_id = device_id

    def post(self, path: str, payload: Any) -> bool:
        try:
            r = self.client.post(f"/api/v1/devices/{self.device_id}/{path}", json=payload)
            return r.status_code == 200
        except Exception:
            return False

    def get(self, path: str) -> Any | None:
        try:
            r = self.client.get(f"/api/v1/devices/{self.device_id}/{path}")
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None


class Uploader:
    """큐 전송 루프. send_pending() 을 주기적으로 호출. 연속 실패 시 백오프(최대 max_backoff 초)."""

    def __init__(self, transport: Transport, queue: LocalQueue, max_backoff: float = 300.0, on_config: Callable[[dict[str, Any]], None] | None = None):
        self.t = transport
        self.q = queue
        self.max_backoff = max_backoff
        self.backoff = 1.0
        self.next_try = 0.0
        self.on_config = on_config
        self.config_version: int | None = None
        self.sent = 0

    def enqueue_counts(self, stats: DayStats) -> None:
        self.q.put("counts", stats.date, stats.payload())

    def enqueue_event(self, type_: str, code: str, message: str, ts: datetime | None = None) -> None:
        ts = ts or datetime.now()
        key = f"{ts.isoformat()}|{code}|{message}"[:200]
        self.q.put("events", key, [{"device_ts": ts.isoformat(), "type": type_, "code": code, "message": message, "dedup_key": key}])

    def enqueue_egg(self, egg_uid: str, scores: list[float], fused: float, reject: bool, severity: int, cam: int = 0, roi: int = 0, ts: datetime | None = None) -> None:
        ts = ts or datetime.now()
        self.q.put("eggs", egg_uid, [{"egg_uid": egg_uid, "device_ts": ts.isoformat(), "cam": cam, "roi": roi, "scores": scores, "fused_score": fused, "reject": reject, "severity": severity}])

    def heartbeat(self, status: str = "ok", fps: float = 0.0, model_version: str = "", temp_c: float | None = None) -> bool:
        ok = self.t.post("heartbeat", {"device_ts": datetime.now().isoformat(), "status": status, "fps": fps, "queue_len": len(self.q), "model_version": model_version, "temp_c": temp_c})
        return ok

    def send_pending(self, now: float | None = None, limit: int = 20) -> int:
        now = now if now is not None else time.time()
        if now < self.next_try:
            return 0
        n_ok = 0
        for row_id, kind, key, payload, attempts in self.q.peek(limit):
            if self.t.post(kind, payload):
                self.q.ack(row_id)
                n_ok += 1
            else:
                self.q.fail(row_id)
                self.backoff = min(self.max_backoff, self.backoff * 2)
                self.next_try = now + self.backoff
                return n_ok
        self.backoff = 1.0
        self.sent += n_ok
        return n_ok

    def poll_config(self) -> dict[str, Any] | None:
        c = self.t.get("config")
        if not c:
            return None
        if c.get("config_version") != self.config_version:
            self.config_version = c.get("config_version")
            if self.on_config:
                self.on_config(c.get("config") or {})
        return c
