from datetime import datetime

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.webmon.app import ADMIN_TOKEN, create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    return TestClient(app)


def _register(client, device_id=101):
    r = client.post("/api/v1/admin/devices", json={"device_id": device_id, "name": "검출기A", "site": "1농장"}, headers={"X-Admin-Token": ADMIN_TOKEN})
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


def test_register_requires_admin(client):
    r = client.post("/api/v1/admin/devices", json={"device_id": 1})
    assert r.status_code == 401


def test_device_flow_and_ks_export(client):
    key = _register(client)
    h = {"Authorization": f"Bearer {key}"}
    now = datetime(2026, 10, 9, 9, 0, 0).isoformat()
    assert client.post("/api/v1/devices/101/heartbeat", json={"device_ts": now, "status": "ok", "fps": 12.5, "queue_len": 0, "model_version": "crackbin_b0_v1"}, headers=h).status_code == 200
    assert client.post("/api/v1/devices/101/heartbeat", json={"device_ts": now}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    body = {"date": "2026-10-09", "processed": 1200, "cracked": 36, "severity": {"1": 20, "2": 12, "3": 4}, "hourly": [{"hour_ts": "2026-10-09T09", "processed": 600, "cracked": 18}]}
    assert client.post("/api/v1/devices/101/counts", json=body, headers=h).status_code == 200
    body["processed"], body["cracked"] = 1500, 40  # 누적값 재전송 → upsert
    assert client.post("/api/v1/devices/101/counts", json=body, headers=h).status_code == 200
    ev = [{"device_ts": now, "type": "alarm", "code": "E200", "message": "camera lost", "dedup_key": "e1"}]
    assert client.post("/api/v1/devices/101/events", json=ev, headers=h).json()["stored"] == 1
    assert client.post("/api/v1/devices/101/events", json=ev, headers=h).json()["stored"] == 0  # 멱등
    eggs = [{"egg_uid": "c0r1_000123", "device_ts": now, "cam": 0, "roi": 1, "scores": [0.1, 0.8, 0.9], "fused_score": 0.9, "reject": True, "severity": 2}]
    assert client.post("/api/v1/devices/101/eggs", json=eggs, headers=h).json()["stored"] == 1
    assert client.post("/api/v1/devices/101/eggs", json=eggs, headers=h).json()["stored"] == 0

    s = client.get("/api/v1/devices/101/summary?days=30").json()
    assert s["daily"][0]["processed"] == 1500 and s["daily"][0]["cracked"] == 40 and s["daily"][0]["severity"] == [20, 12, 4]
    assert s["hourly"][0]["processed"] == 600 and s["events"][0]["code"] == "E200" and s["device"]["model_version"] == "crackbin_b0_v1"
    lst = client.get("/api/v1/devices").json()
    assert lst[0]["status"] == "online" and lst[0]["device_id"] == 101

    r = client.put("/api/v1/admin/devices/101/config", json={"config": {"decision": {"crack_threshold": 0.42}}}, headers={"X-Admin-Token": ADMIN_TOKEN})
    assert r.json()["config_version"] == 2 and len(r.json()["set_ts"]) == 14
    c = client.get("/api/v1/devices/101/config", headers=h).json()
    assert c["config_version"] == 2 and c["config"]["decision"]["crack_threshold"] == 0.42

    csv_text = client.get("/api/v1/export/ks?device_id=101").text
    lines = csv_text.strip().splitlines()
    assert "검출기 번호" in lines[0] and lines[1].startswith("101,2026-10-09,1500,40,") and len(lines[1].split(",")[-1]) == 14

    assert client.get("/").status_code == 200 and "검출기A" in client.get("/").text
    assert client.get("/devices/101").status_code == 200 and "E200" in client.get("/devices/101").text
    assert client.get("/healthz").text == "ok"
