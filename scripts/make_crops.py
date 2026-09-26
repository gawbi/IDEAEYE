"""Phase 2: 계란 크롭 생성. python scripts/make_crops.py [--limit 200] [--workers 16] [--lights MONO COLOR]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.datasets.audit import load_records  # noqa: E402
from src.datasets.crops import run_crops  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="이미지 수 제한 (스모크 테스트)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lights", nargs="*", default=None)
    args = ap.parse_args()

    data_cfg, crops_cfg = load_config("data"), load_config("crops")
    if args.lights:
        crops_cfg["lights"] = args.lights
    df = load_records(resolve_path(crops_cfg["records_cache"]))
    split_path = resolve_path(crops_cfg["split_csv"])
    split_df = pd.read_csv(split_path, dtype=str, encoding="utf-8-sig") if split_path.exists() else None
    t0 = time.time()
    run_crops(data_cfg, crops_cfg, df, split_df, workers=args.workers, limit=args.limit)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
