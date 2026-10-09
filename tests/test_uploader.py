from datetime import datetime

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.telemetry.uploader import Aggregator, LocalQueue, Uploader  # noqa: E402
from src.webmon.app import ADMIN_TOKEN, create_app  # noqa: E402


class FlakyTransport:
    """처음 fail_n 번은 실패(오프라인), 그 뒤 성공. 전송된 payload 기록."""

    def __init__(self, fail_n=0):
        self.fail_n = fail_n
        self.posts = []
        self.config = {"config_version": 1, "config": {"decision": {"crack_threshold": 0.5}}}

    def post(self, path, payload):
        if self.fail_n > 0:
            self.fail_n -= 1
            return False
        self.posts.append((path, payload))
        return True

    def get(self, path):
        return self.config


def test_aggregator_counts():
    ag = Aggregator()
    ts = datetime(2026, 10, 9, 9, 5)
    for i in range(10):
        ag.add(reject=(i % 5 == 0), severity=2, ts=ts)
    ag.add(reject=True, severity=3, ts=datetime(2026, 10, 9, 10, 1))
    st = ag.days["2026-10-09"]
    assert st.processed == 11 and st.cracked == 3 and st.severity == {"1": 0, "2": 2, "3": 1}
    p = st.payload()
    assert p["hourly"][0] == {"hour_ts": "2026-10-09T09", "processed": 10, "cracked": 2}


def test_queue_offline_then_resend(tmp_path):
    q = LocalQueue(tmp_path / "q.sqlite3")
    t = FlakyTransport(fail_n=2)
    up = Uploader(t, q, max_backoff=8)
    ag = Aggregator()
    st = ag.add(True, 1, datetime(2026, 10, 9, 9))
    up.enqueue_counts(st)
    up.enqueue_event("alarm", "E200", "camera lost", datetime(2026, 10, 9, 9, 1))
    assert len(q) == 2
    assert up.send_pending(now=0) == 0 and len(q) == 2 and up.backoff == 2  # 실패 → 백오프
    assert up.send_pending(now=1) == 0                                         # 아직 대기
    assert up.send_pending(now=3) == 0 and up.backoff == 4                     # 두 번째 실패
    assert up.send_pending(now=10) == 2 and len(q) == 0                        # 복구 후 전부 전송
    kinds = [p[0] for p in t.posts]
    assert kinds == ["counts", "events"]
    # 같은 날짜 집계는 키가 같아 덮어쓰기 (중복 전송 방지)
    st = ag.add(False, 0, datetime(2026, 10, 9, 9, 30))
    up.enqueue_counts(st)
    up.enqueue_counts(st)
    assert len(q) == 1


def test_config_poll_callback(tmp_path):
    t = FlakyTransport()
    applied = []
    up = Uploader(t, LocalQueue(tmp_path / "q.sqlite3"), on_config=applied.append)
    up.poll_config()
    up.poll_config()
    assert applied == [{"decision": {"crack_threshold": 0.5}}]  # 버전이 같으면 한 번만
    t.config = {"config_version": 2, "config": {"decision": {"crack_threshold": 0.4}}}
    up.poll_config()
    assert applied[-1]["decision"]["crack_threshold"] == 0.4


def test_end_to_end_with_server(tmp_path):
    """실제 FastAPI 앱 + httpx 전송기로 장비 → 서버 왕복."""
    import httpx

    app = create_app(f"sqlite:///{tmp_path / 'srv.sqlite3'}")
    admin = TestClient(app)
    key = admin.post("/api/v1/admin/devices", json={"device_id": 7, "name": "sim"}, headers={"X-Admin-Token": ADMIN_TOKEN}).json()["api_key"]

    class AppTransport:  # HttpTransport 와 같은 인터페이스를 TestClient 로 구현
        def __init__(self):
            self.c = TestClient(app)
            self.h = {"Authorization": f"Bearer {key}"}

        def post(self, path, payload):
            return self.c.post(f"/api/v1/devices/7/{path}", json=payload, headers=self.h).status_code == 200

        def get(self, path):
            r = self.c.get(f"/api/v1/devices/7/{path}", headers=self.h)
            return r.json() if r.status_code == 200 else None

    up = Uploader(AppTransport(), LocalQueue(tmp_path / "dev.sqlite3"))
    ag = Aggregator()
    for i in range(50):
        st = ag.add(reject=(i % 10 == 0), severity=1 + i % 3, ts=datetime(2026, 10, 9, 9 + i // 25, i % 25))
    up.enqueue_counts(st)
    up.enqueue_egg("c0r0_0001", [0.2, 0.9, 0.95], 0.95, True, 2, ts=datetime(2026, 10, 9, 9))
    assert up.heartbeat(fps=11.0, model_version="v1")
    assert up.send_pending() == 2
    s = admin.get("/api/v1/devices/7/summary?days=30").json()
    assert s["daily"][0]["processed"] == 50 and s["daily"][0]["cracked"] == 5 and s["device"]["status"] == "online"
    assert up.poll_config()["config_version"] == 1
