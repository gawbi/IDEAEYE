"""라벨 품질 점검 (CPU): bbox 크기 이상치, 가장자리 걸침, 판 내 중복 박스, object_count 불일치, 메타 결측.

python scripts/label_quality.py  → outputs/audit/label_quality.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import resolve_path  # noqa: E402
from src.datasets.audit import load_records  # noqa: E402


def pairwise_dup(g: pd.DataFrame, iou_thr: float = 0.85) -> int:
    """판 안에서 IoU > thr 인 박스 쌍 수 (판당 계란 ≤ 17 이라 O(n²) 무방)."""
    b = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    n = len(b)
    if n < 2:
        return 0
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
    return int((np.triu(iou, 1) > iou_thr).sum())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = load_records(resolve_path("outputs/audit/records.pkl"))
    df["w"], df["h"] = df["x2"] - df["x1"], df["y2"] - df["y1"]
    lines = ["# 라벨 품질 점검 (scripts/label_quality.py)", "", f"객체 {len(df):,} / 판 {df.drop_duplicates(['light', 'tray_id']).shape[0]:,}", ""]

    lines += ["## 1. bbox 크기 이상치 (광원별 중앙값의 0.6배 미만 또는 1.5배 초과)", "", "| light | median w×h | 너무 작음 | 너무 큼 | 가로세로비 이상(w/h<0.55 or >1.1) |", "|---|---|---:|---:|---:|"]
    flags = {}
    for light, g in df.groupby("light"):
        mw, mh = g["w"].median(), g["h"].median()
        small = (g["w"] < 0.6 * mw) | (g["h"] < 0.6 * mh)
        big = (g["w"] > 1.5 * mw) | (g["h"] > 1.5 * mh)
        ar = g["w"] / g["h"].clip(lower=1)
        odd = (ar < 0.55) | (ar > 1.1)
        flags[light] = {"small": small, "big": big, "odd": odd}
        lines.append(f"| {light} | {mw:.0f}×{mh:.0f} | {int(small.sum()):,} ({small.mean():.2%}) | {int(big.sum()):,} ({big.mean():.2%}) | {int(odd.sum()):,} ({odd.mean():.2%}) |")

    lines += ["", "## 2. 이미지 가장자리에 걸친 박스 (경계 2px 이내 또는 범위 밖)", "", "| light | 걸침 | 범위 밖 |", "|---|---:|---:|"]
    for light, g in df.groupby("light"):
        edge = (g["x1"] <= 2) | (g["y1"] <= 2) | (g["x2"] >= g["width"] - 2) | (g["y2"] >= g["height"] - 2)
        oob = (g["x1"] < 0) | (g["y1"] < 0) | (g["x2"] > g["width"]) | (g["y2"] > g["height"])
        lines.append(f"| {light} | {int(edge.sum()):,} ({edge.mean():.2%}) | {int(oob.sum()):,} |")

    lines += ["", "## 3. 판 내 중복 박스 (IoU>0.85 쌍)", ""]
    for light, g in df.groupby("light"):
        sample = g.groupby("tray_id")
        dup = sum(pairwise_dup(t) for _, t in sample)
        lines.append(f"- {light}: 중복 쌍 {dup:,} (판 {sample.ngroups:,})")

    lines += ["", "## 4. object_count(XML) vs 실제 bndbox 수", ""]
    if "meta_object_count" in df:
        per = df.groupby(["light", "tray_id"]).agg(n=("obj_index", "size"), declared=("meta_object_count", "first"))
        per["declared"] = pd.to_numeric(per["declared"], errors="coerce")
        mism = per[per["declared"].notna() & (per["declared"] != per["n"])]
        for light, g in per.groupby(level=0):
            m = mism.loc[light] if light in mism.index.get_level_values(0) else pd.DataFrame()
            lines.append(f"- {light}: 불일치 판 {len(m):,} / {len(g):,}" + (f" (예: 선언 {int(m['declared'].iloc[0])} vs 실제 {int(m['n'].iloc[0])})" if len(m) else ""))

    lines += ["", "## 5. 메타 결측 (판 단위)", ""]
    trays = df.drop_duplicates(["light", "tray_id"])
    for col in ["date", "region_code", "washed", "breed", "meta_farming_env", "meta_category", "meta_temperature", "meta_humidity"]:
        if col in trays:
            miss = trays[col].isna() | (trays[col].astype(str).str.strip() == "")
            if miss.any():
                lines.append(f"- {col}: 결측 {int(miss.sum()):,} 판 ({dict(trays.loc[miss, 'light'].value_counts())})")
    lines += ["", "## 6. 소분류가 0 인 객체 (MONO 는 소분류 필수)", ""]
    for light, g in df.groupby("light"):
        z = (g["state_detail"].astype(str) == "0").mean()
        lines.append(f"- {light}: state_detail=0 비율 {z:.2%}")

    out = resolve_path("outputs/audit/label_quality.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
