"""Phase 3-A: YOLO 검출 데이터셋 추출. python scripts/export_det.py [--lights MONO COLOR] [--workers 8]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.datasets.audit import load_records  # noqa: E402
from src.datasets.yolo_export import export_yolo  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lights", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    data_cfg, det_cfg, crops_cfg = load_config("data"), load_config("det_export"), load_config("crops")
    df = load_records(resolve_path(crops_cfg["records_cache"]))
    split_df = pd.read_csv(resolve_path(crops_cfg["split_csv"]), dtype=str, encoding="utf-8-sig")
    lights = args.lights or list(det_cfg["per_light"])
    t0 = time.time()
    for light in lights:
        p = export_yolo(data_cfg, det_cfg, df, split_df, light, workers=args.workers)
        print(f"→ {p}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
