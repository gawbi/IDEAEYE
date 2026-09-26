"""COLOR 세척여부 비교 전용 대체 분할 (판 단위, 세척 코드별 층화). 촬영일이 분할 간에 겹치므로 KPI 평가에는 쓰지 않는다.

배경: COLOR 는 세척 코드가 촬영일·지역과 완전히 겹쳐(세척1 = 10월/37380, 세척2 = 9월/34310) 촬영일 기준 분할에서는 test 에 한 세척 코드만 남는다.
python scripts/make_alt_split_color.py → data/crops/index_color_wash.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import resolve_path  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    idx = pd.read_csv(resolve_path("data/crops/index.csv"), dtype={"cls": str, "state": str, "state_detail": str, "washed": str, "region_code": str}, encoding="utf-8-sig")
    color = idx[idx["light"] == "COLOR"].copy()
    trays = color.drop_duplicates("tray_id")[["tray_id", "washed", "date"]]
    rng = np.random.RandomState(42)
    assign = {}
    for w, g in trays.groupby("washed"):  # 세척 코드별로 판을 섞어 8:1:1
        ids = g["tray_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        for i, t in enumerate(ids):
            assign[t] = "train" if i < 0.8 * n else ("val" if i < 0.9 * n else "test")
    color["split"] = color["tray_id"].map(assign)
    out = resolve_path("data/crops/index_color_wash.csv")
    color.to_csv(out, index=False, encoding="utf-8-sig")
    print(pd.crosstab([color["split"]], color["washed"]).to_markdown())
    print(pd.crosstab(color["split"], color["state"]).to_markdown())
    print(f"→ {out}  (주의: 촬영일이 분할 간에 겹침 — 세척 비교 전용)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
