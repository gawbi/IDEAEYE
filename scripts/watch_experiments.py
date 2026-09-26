"""experiments/ 에 eval.json 이 있는 실험 폴더 수가 늘어나면 종료 (auto_next 체인의 작업 완료를 백그라운드 알림으로 받기 위한 감시).

python scripts/watch_experiments.py --count <현재 수> [--poll 60] [--max-hours 4]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def count_done() -> tuple[int, list[str]]:
    dirs = sorted(d for d in ROOT.iterdir() if d.is_dir() and ((d / "eval.json").exists() or (d / "summary.json").exists() or (d / "benchmark.json").exists()))
    return len(dirs), [d.name for d in dirs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=None, help="기준 개수 (생략 시 현재 개수)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=4)
    args = ap.parse_args()
    base = args.count if args.count is not None else count_done()[0]
    t0 = time.time()
    while True:
        n, names = count_done()
        if n > base:
            print(f"new experiments: {names[base:]} (total {n})")
            return 0
        if time.time() - t0 > args.max_hours * 3600:
            print("watch timeout")
            return 3
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
