"""큐 상태 감시: 완료된 작업 수가 늘어나거나 큐가 끝나면 종료한다 (백그라운드 완료 알림으로 점검을 트리거하기 위한 용도).

python scripts/watch_queue.py --status outputs/logs/queue_status_queue_all.json --done 2 [--poll 30] [--max-hours 4]
  --done N : 현재 완료(returncode 기록)된 작업 수. 이보다 많아지면 즉시 종료(rc 0). 큐 자체가 끝나면 rc 0. 시간 초과 rc 3.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def snapshot(path: Path) -> tuple[int, bool, list[dict]]:
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, False, []
    jobs = s.get("jobs", [])
    done = sum(1 for j in jobs if "returncode" in j)
    return done, "finished" in s, jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="outputs/logs/queue_status_queue_all.json")
    ap.add_argument("--done", type=int, required=True)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--max-hours", type=float, default=4)
    args = ap.parse_args()
    path = Path(args.status)
    t0 = time.time()
    while True:
        done, finished, jobs = snapshot(path)
        if done > args.done or finished:
            last = jobs[done - 1] if done else {}
            print(json.dumps({"done": done, "finished": finished, "last_job": {k: last.get(k) for k in ("name", "returncode", "seconds")}, "running": jobs[done]["name"] if done < len(jobs) else None}, ensure_ascii=False))
            return 0
        if time.time() - t0 > args.max_hours * 3600:
            print("watch timeout")
            return 3
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
