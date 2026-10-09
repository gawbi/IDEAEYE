"""FastAPI 서버: 장비 API(하트비트·집계·이벤트·계란 로그·설정 폴링) + 대시보드 + KS 송출 내보내기.

실행: uvicorn src.webmon.app:app --host 0.0.0.0 --port 8000
관리자 토큰: 환경변수 EGGMON_ADMIN_TOKEN (기본 'admin-dev'); 장비 등록 시 발급되는 api_key 를 장비가 Bearer 로 사용.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from .db import DailyCount, Device, EggLog, Event, Heartbeat, HourlyCount, make_engine, make_session_factory
from .schemas import ConfigUpdate, CountsIn, DeviceCreate, EggLogIn, EventIn, HeartbeatIn

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
ADMIN_TOKEN = os.environ.get("EGGMON_ADMIN_TOKEN", "admin-dev")
OFFLINE_AFTER = timedelta(minutes=5)


def create_app(database_url: str | None = None) -> FastAPI:
    app = FastAPI(title="파각란검출기 모니터링", version="0.1")
    engine = make_engine(database_url)
    SessionLocal = make_session_factory(engine)
    app.state.engine = engine

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def hash_key(k: str) -> str:
        return hashlib.sha256(k.encode()).hexdigest()

    def auth_device(device_id: int, authorization: str = Header(default=""), db: Session = Depends(get_db)) -> Device:
        dev = db.get(Device, device_id)
        if dev is None:
            raise HTTPException(404, "unknown device")
        token = authorization.removeprefix("Bearer ").strip()
        if not token or hash_key(token) != dev.api_key_hash:
            raise HTTPException(401, "bad device token")
        return dev

    def auth_admin(x_admin_token: str = Header(default="")) -> None:
        if x_admin_token != ADMIN_TOKEN:
            raise HTTPException(401, "admin token required")

    # ---------------- 관리자: 장비 등록/설정
    @app.post("/api/v1/admin/devices")
    def create_device(body: DeviceCreate, db: Session = Depends(get_db), _=Depends(auth_admin)):
        if db.get(Device, body.device_id):
            raise HTTPException(409, "device exists")
        key = secrets.token_urlsafe(24)
        dev = Device(device_id=body.device_id, name=body.name, site=body.site, api_key_hash=hash_key(key), config={"decision": {"crack_threshold": 0.5}}, config_version=1)
        db.add(dev)
        db.commit()
        return {"device_id": dev.device_id, "api_key": key, "config_version": dev.config_version}

    @app.put("/api/v1/admin/devices/{device_id}/config")
    def set_config(device_id: int, body: ConfigUpdate, db: Session = Depends(get_db), _=Depends(auth_admin)):
        dev = db.get(Device, device_id)
        if dev is None:
            raise HTTPException(404)
        dev.config = body.config
        dev.config_version = (dev.config_version or 0) + 1
        dev.config_set_ts = datetime.utcnow()
        db.add(Event(device_id=device_id, device_ts=datetime.utcnow(), type="config", code="SET", message=f"config v{dev.config_version}"))
        db.commit()
        return {"config_version": dev.config_version, "set_ts": dev.config_set_ts.strftime("%Y%m%d%H%M%S")}

    # ---------------- 장비 → 서버
    @app.post("/api/v1/devices/{device_id}/heartbeat")
    def heartbeat(device_id: int, body: HeartbeatIn, dev: Device = Depends(auth_device), db: Session = Depends(get_db)):
        db.add(Heartbeat(device_id=device_id, device_ts=body.device_ts, status=body.status, fps=body.fps, queue_len=body.queue_len, temp_c=body.temp_c, extra=body.extra))
        dev.last_seen_at = datetime.utcnow()
        if body.model_version:
            dev.model_version = body.model_version
        db.commit()
        return {"ok": True, "config_version": dev.config_version}

    @app.post("/api/v1/devices/{device_id}/counts")
    def counts(device_id: int, body: CountsIn, dev: Device = Depends(auth_device), db: Session = Depends(get_db)):
        row = db.execute(select(DailyCount).where(DailyCount.device_id == device_id, DailyCount.date == body.date)).scalar_one_or_none()
        if row is None:
            row = DailyCount(device_id=device_id, date=body.date)
            db.add(row)
        row.processed, row.cracked = body.processed, body.cracked  # upsert (장비가 누적값을 보냄 → 멱등)
        row.severity_1, row.severity_2, row.severity_3 = body.severity.get("1", 0), body.severity.get("2", 0), body.severity.get("3", 0)
        row.updated_ts = datetime.utcnow()
        for h in body.hourly:
            hr = db.execute(select(HourlyCount).where(HourlyCount.device_id == device_id, HourlyCount.hour_ts == h["hour_ts"])).scalar_one_or_none()
            if hr is None:
                hr = HourlyCount(device_id=device_id, hour_ts=h["hour_ts"])
                db.add(hr)
            hr.processed, hr.cracked = int(h.get("processed", 0)), int(h.get("cracked", 0))
        dev.last_seen_at = datetime.utcnow()
        db.commit()
        return {"ok": True}

    @app.post("/api/v1/devices/{device_id}/events")
    def events(device_id: int, body: list[EventIn], dev: Device = Depends(auth_device), db: Session = Depends(get_db)):
        n = 0
        for e in body:
            if e.dedup_key and db.execute(select(Event).where(Event.device_id == device_id, Event.dedup_key == e.dedup_key)).first():
                continue
            db.add(Event(device_id=device_id, device_ts=e.device_ts, type=e.type, code=e.code, message=e.message, dedup_key=e.dedup_key))
            n += 1
        db.commit()
        return {"ok": True, "stored": n}

    @app.post("/api/v1/devices/{device_id}/eggs")
    def eggs(device_id: int, body: list[EggLogIn], dev: Device = Depends(auth_device), db: Session = Depends(get_db)):
        n = 0
        for g in body:
            if db.execute(select(EggLog).where(EggLog.device_id == device_id, EggLog.egg_uid == g.egg_uid)).first():
                continue
            db.add(EggLog(device_id=device_id, egg_uid=g.egg_uid, device_ts=g.device_ts, cam=g.cam, roi=g.roi, scores=g.scores, fused_score=g.fused_score, reject=g.reject, severity=g.severity))
            n += 1
        db.commit()
        return {"ok": True, "stored": n}

    @app.get("/api/v1/devices/{device_id}/config")
    def get_config(device_id: int, dev: Device = Depends(auth_device)):
        return {"config_version": dev.config_version, "config": dev.config, "set_ts": dev.config_set_ts.strftime("%Y%m%d%H%M%S") if dev.config_set_ts else None}

    # ---------------- 대시보드/조회 (관리자 토큰 또는 로그인 생략: 1차는 내부망 가정)
    def device_status(dev: Device) -> str:
        if dev.last_seen_at is None:
            return "never"
        return "online" if datetime.utcnow() - dev.last_seen_at < OFFLINE_AFTER else "offline"

    @app.get("/api/v1/devices")
    def list_devices(db: Session = Depends(get_db)):
        out = []
        today = datetime.utcnow().strftime("%Y-%m-%d")
        for dev in db.execute(select(Device)).scalars():
            dc = db.execute(select(DailyCount).where(DailyCount.device_id == dev.device_id, DailyCount.date == today)).scalar_one_or_none()
            out.append({"device_id": dev.device_id, "name": dev.name, "site": dev.site, "status": device_status(dev), "last_seen_at": dev.last_seen_at,
                        "today_processed": dc.processed if dc else 0, "today_cracked": dc.cracked if dc else 0, "config_version": dev.config_version, "model_version": dev.model_version})
        return out

    @app.get("/api/v1/devices/{device_id}/summary")
    def summary(device_id: int, days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)):
        dev = db.get(Device, device_id)
        if dev is None:
            raise HTTPException(404)
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        daily = db.execute(select(DailyCount).where(DailyCount.device_id == device_id, DailyCount.date >= since).order_by(DailyCount.date)).scalars().all()
        hourly = db.execute(select(HourlyCount).where(HourlyCount.device_id == device_id, HourlyCount.hour_ts >= since).order_by(HourlyCount.hour_ts)).scalars().all()
        ev = db.execute(select(Event).where(Event.device_id == device_id).order_by(desc(Event.id)).limit(50)).scalars().all()
        hb = db.execute(select(Heartbeat).where(Heartbeat.device_id == device_id).order_by(desc(Heartbeat.id)).limit(1)).scalar_one_or_none()
        return {"device": {"device_id": dev.device_id, "name": dev.name, "site": dev.site, "status": device_status(dev), "config": dev.config, "config_version": dev.config_version, "model_version": dev.model_version},
                "daily": [{"date": d.date, "processed": d.processed, "cracked": d.cracked, "rate": (d.cracked / d.processed) if d.processed else 0.0, "severity": [d.severity_1, d.severity_2, d.severity_3]} for d in daily],
                "hourly": [{"hour_ts": h.hour_ts, "processed": h.processed, "cracked": h.cracked} for h in hourly],
                "events": [{"ts": e.device_ts, "type": e.type, "code": e.code, "message": e.message} for e in ev],
                "last_heartbeat": {"device_ts": hb.device_ts, "status": hb.status, "fps": hb.fps, "queue_len": hb.queue_len, "temp_c": hb.temp_c} if hb else None}

    @app.get("/api/v1/export/ks")
    def export_ks(device_id: int | None = None, date_from: str | None = None, date_to: str | None = None, db: Session = Depends(get_db)):
        """KS B 7956-4 송출 데이터: 검출기 번호, 처리란 개수(개/일), 파각란 개수(개/일), 설정 시간(YYYYMMDDhhmmss)."""
        q = select(DailyCount)
        if device_id is not None:
            q = q.where(DailyCount.device_id == device_id)
        if date_from:
            q = q.where(DailyCount.date >= date_from)
        if date_to:
            q = q.where(DailyCount.date <= date_to)
        rows = db.execute(q.order_by(DailyCount.device_id, DailyCount.date)).scalars().all()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["검출기 번호", "날짜", "처리란 개수(개/일)", "파각란 개수(개/일)", "설정 시간(YYYYMMDDhhmmss)"])
        for r in rows:
            dev = db.get(Device, r.device_id)
            set_ts = dev.config_set_ts.strftime("%Y%m%d%H%M%S") if dev and dev.config_set_ts else ""
            w.writerow([r.device_id, r.date, r.processed, r.cracked, set_ts])
        buf.seek(0)
        return StreamingResponse(iter([("﻿" + buf.getvalue()).encode("utf-8")]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=ks_export.csv"})

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, db: Session = Depends(get_db)):
        return TEMPLATES.TemplateResponse(request, "devices.html", {"devices": list_devices(db)})

    @app.get("/devices/{device_id}", response_class=HTMLResponse)
    def device_page(request: Request, device_id: int, days: int = 7, db: Session = Depends(get_db)):
        return TEMPLATES.TemplateResponse(request, "device.html", {"s": summary(device_id, days, db), "days": days})

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz():
        return "ok"

    return app


app = create_app()
