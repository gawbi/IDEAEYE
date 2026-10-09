"""SQLAlchemy 모델 (SQLite 기본, DATABASE_URL 로 교체 가능)."""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Device(Base):
    __tablename__ = "devices"
    device_id = Column(Integer, primary_key=True)          # KS 검출기 번호 (농장 단위 고유 정수)
    name = Column(String, default="")
    site = Column(String, default="")
    api_key_hash = Column(String, nullable=False)
    model_version = Column(String, default="")
    config = Column(JSON, default=dict)
    config_version = Column(Integer, default=1)
    config_set_ts = Column(DateTime, default=datetime.utcnow)  # KS 설정 시각
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=True)


class Heartbeat(Base):
    __tablename__ = "heartbeats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, index=True)
    device_ts = Column(DateTime)
    server_ts = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="ok")
    fps = Column(Float, default=0.0)
    queue_len = Column(Integer, default=0)
    temp_c = Column(Float, nullable=True)
    extra = Column(JSON, default=dict)


class DailyCount(Base):
    __tablename__ = "daily_counts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, index=True)
    date = Column(String, index=True)  # YYYY-MM-DD (장비 로컬)
    processed = Column(Integer, default=0)
    cracked = Column(Integer, default=0)
    severity_1 = Column(Integer, default=0)
    severity_2 = Column(Integer, default=0)
    severity_3 = Column(Integer, default=0)
    updated_ts = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("device_id", "date", name="uq_daily"),)


class HourlyCount(Base):
    __tablename__ = "hourly_counts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, index=True)
    hour_ts = Column(String, index=True)  # YYYY-MM-DDTHH
    processed = Column(Integer, default=0)
    cracked = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint("device_id", "hour_ts", name="uq_hourly"),)


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, index=True)
    device_ts = Column(DateTime)
    server_ts = Column(DateTime, default=datetime.utcnow)
    type = Column(String)      # alarm | config | start | stop | info
    code = Column(String, default="")
    message = Column(String, default="")
    dedup_key = Column(String, index=True, nullable=True)  # 멱등 전송용


class EggLog(Base):
    __tablename__ = "egg_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, index=True)
    egg_uid = Column(String, index=True)
    device_ts = Column(DateTime)
    cam = Column(Integer, default=0)
    roi = Column(Integer, default=0)
    scores = Column(JSON, default=list)
    fused_score = Column(Float, default=0.0)
    reject = Column(Boolean, default=False)
    severity = Column(Integer, default=0)
    image_key = Column(String, nullable=True)
    __table_args__ = (UniqueConstraint("device_id", "egg_uid", name="uq_egg"),)


def make_engine(url: str | None = None):
    url = url or os.environ.get("EGGMON_DATABASE_URL", "sqlite:///data/webmon.sqlite3")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
