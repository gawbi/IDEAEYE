"""가상 검출기 시뮬레이터: 서버에 장비를 등록하고 집계·하트비트·이벤트·계란 로그를 보낸다 (대시보드 데모·통신 테스트용).

1) 서버 실행:  uvicorn src.webmon.app:app --port 8000
2) 시뮬레이터: python scripts/simulate_device.py --server http://127.0.0.1:8000 --device-id 101 --days 3 [--live 60]
   --live N : N초 동안 10초마다 실시간 집계·하트비트 전송 (Ctrl+C 로 중단)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.telemetry.uploader import Aggregator, HttpTransport, LocalQueue, Uploader  # noqa: E402
from src.webmon.app import ADMIN_TOKEN  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--device-id", type=int, default=101)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--live", type=int, default=0)
    ap.add_argument("--queue", default="data/device_queue_sim.sqlite3")
    args = ap.parse_args()

    r = httpx.post(f"{args.server}/api/v1/admin/devices", json={"device_id": args.device_id, "name": f"시뮬레이터 {args.device_id}", "site": "시험장"}, headers={"X-Admin-Token": ADMIN_TOKEN})
    if r.status_code == 409:
        print("이미 등록된 장비입니다. 기존 api_key 가 필요하므로 새 device-id 로 실행하세요.")
        return 1
    r.raise_for_status()
    key = r.json()["api_key"]
    print("registered, api_key:", key)

    up = Uploader(HttpTransport(args.server, args.device_id, key), LocalQueue(args.queue), on_config=lambda c: print("config applied:", c))
    rng = random.Random(args.device_id)
    # 과거 N일 집계 생성
    for d in range(args.days, 0, -1):
        ag = Aggregator()
        day = datetime.now() - timedelta(days=d)
        for h in range(8, 18):
            for _ in range(rng.randint(300, 500)):
                reject = rng.random() < 0.04
                ag.add(reject, rng.choice([1, 1, 2, 3]) if reject else 0, day.replace(hour=h, minute=rng.randint(0, 59)))
        for st in ag.days.values():
            up.enqueue_counts(st)
        if rng.random() < 0.5:
            up.enqueue_event("alarm", "W400", "latency exceeded", day.replace(hour=12))
    print("queued", len(up.q), "→ sending")
    while len(up.q):
        n = up.send_pending()
        if n == 0:
            time.sleep(1)
    up.heartbeat(fps=12.0, model_version="crackbin_b0_sim")
    up.poll_config()
    print("history sent. dashboard:", f"{args.server}/devices/{args.device_id}")

    if args.live:
        ag = Aggregator()
        t_end = time.time() + args.live
        while time.time() < t_end:
            for _ in range(rng.randint(20, 40)):
                reject = rng.random() < 0.04
                st = ag.add(reject, rng.choice([1, 2, 3]) if reject else 0)
                if reject:
                    up.enqueue_egg(f"sim_{int(time.time() * 1000)}", [rng.random(), rng.random(), 0.9], 0.9, True, 2)
            up.enqueue_counts(st)
            up.send_pending()
            up.heartbeat(fps=12.0 + rng.random(), model_version="crackbin_b0_sim", temp_c=45 + rng.random() * 5)
            up.poll_config()
            print(datetime.now().strftime("%H:%M:%S"), "processed", st.processed, "cracked", st.cracked, "queue", len(up.q))
            time.sleep(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
