"""다른 프로세스가 끝날 때까지 대기 (GPU 큐가 앞선 백그라운드 학습과 겹치지 않게).

python scripts/wait_for.py train_det.py [--poll 30] [--max-hours 6]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def running(pattern: str) -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {{ $_.CommandLine -match '{pattern}' -and $_.CommandLine -notmatch 'wait_for' }} | Measure-Object).Count"],
        capture_output=True, text=True,
    )
    try:
        return int(out.stdout.strip() or 0) > 0
    except ValueError:
        return False


def queue_finished(status_file: str) -> bool:
    """run_queue.py 상태 파일에 finished 키가 생겼는지 (프로세스 조회보다 훨씬 빠르고 확실)."""
    import json
    from pathlib import Path

    p = Path(status_file)
    if not p.exists():
        return False
    try:
        return "finished" in json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--max-hours", type=float, default=6)
    ap.add_argument("--status-file", default=None, help="지정 시 프로세스 조회 대신 큐 상태 파일의 finished 로 판단 (Get-CimInstance 가 느린 환경용)")
    args = ap.parse_args()
    t0 = time.time()
    if args.status_file:
        while not queue_finished(args.status_file):
            if time.time() - t0 > args.max_hours * 3600:
                print("timeout waiting for", args.status_file)
                return 1
            time.sleep(args.poll)
        print(f"queue finished per {args.status_file} (waited {time.time() - t0:.0f}s)")
        return 0
    while running(args.pattern):
        if time.time() - t0 > args.max_hours * 3600:
            print(f"timeout waiting for {args.pattern}")
            return 1
        time.sleep(args.poll)
    print(f"{args.pattern} finished (waited {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
