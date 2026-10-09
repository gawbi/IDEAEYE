"""라벨된 영상별 index.csv 들을 학습용 data/own/index.csv 로 통합.

- 학습 코드 호환 열: light="DEVICE", crop_path, state("1" normal / "2" crack), state_detail(심각도 1~3, normal 은 0), cls, split, exists
- 그룹 분할: 같은 영상·카메라·자리에서 연속된 정지 구간 3개(=한 계란의 3시점)를 egg_id 로 묶고, egg_id 단위로 train/val/test 8:1:1 (영상이 하나면 egg 단위, 여러 개면 영상 단위 우선)
- unsure/empty/미라벨 크롭은 제외. 계란 단위 라벨(egg_label)은 3시점 중 하나라도 crack 이면 crack, 심각도는 최대값.
python scripts/own_build_index.py [--frames data/own/frames] [--out data/own/index.csv] [--views 3] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import resolve_path  # noqa: E402


def assign_egg_ids(df: pd.DataFrame, views: int) -> pd.Series:
    """영상·카메라·자리별로 정지 구간 순서대로 views 개씩 묶어 egg_id 부여 (기기 회전 신호가 없을 때의 근사)."""
    ids = pd.Series(index=df.index, dtype=object)
    for (video, cam, roi), g in df.sort_values("segment").groupby(["video", "cam", "roi"]):
        for k, idx in enumerate(g.index):
            ids[idx] = f"{Path(video).stem}|c{cam}r{roi}|e{k // views:04d}"
    return ids


def split_by_group(groups: pd.Series, ratios=(0.8, 0.1, 0.1), seed: int = 42) -> pd.Series:
    uniq = sorted(groups.unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train, n_val = int(round(n * ratios[0])), int(round(n * ratios[1]))
    mapping = {g: ("train" if i < n_train else "val" if i < n_train + n_val else "test") for i, g in enumerate(uniq)}
    return groups.map(mapping)


def build(frames_root: Path, views: int, seed: int) -> pd.DataFrame:
    parts = []
    for idx in sorted(frames_root.glob("*/index.csv")):
        d = pd.read_csv(idx, dtype=str, encoding="utf-8-sig").fillna("")
        if "label" in d:
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["cam"], df["roi"], df["segment"] = df["cam"].astype(int), df["roi"].astype(int), df["segment"].astype(int)
    df["egg_id"] = assign_egg_ids(df, views)
    lab = df[df["label"].isin(["normal", "crack"])].copy()
    if lab.empty:
        return lab
    lab["state"] = np.where(lab["label"] == "crack", "2", "1")
    sev = pd.to_numeric(lab["severity"], errors="coerce").fillna(0).astype(int)
    lab["state_detail"] = np.where(lab["state"] == "2", sev.clip(0, 3).astype(str), "0")
    lab["cls"] = lab["state"] + "/" + lab["state_detail"]
    lab["light"] = "DEVICE"
    lab["exists"] = [Path(p).exists() for p in lab["crop_path"]]
    # 계란 단위 라벨
    egg = lab.groupby("egg_id").agg(egg_label=("label", lambda s: "crack" if (s == "crack").any() else "normal"), egg_severity=("state_detail", lambda s: max(int(v) for v in s)))
    lab = lab.merge(egg, left_on="egg_id", right_index=True)
    # 분할: 영상이 2개 이상이면 영상 단위, 아니면 계란 단위
    group_key = lab["video"] if lab["video"].nunique() >= 3 else lab["egg_id"]
    lab["split"] = split_by_group(group_key, seed=seed)
    cols = ["light", "video", "cam", "roi", "segment", "egg_id", "crop_path", "label", "severity", "shell_color", "stamp", "state", "state_detail", "cls", "egg_label", "egg_severity", "split", "exists"]
    return lab[[c for c in cols if c in lab]].reset_index(drop=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/own/frames")
    ap.add_argument("--out", default="data/own/index.csv")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = build(resolve_path(args.frames), args.views, args.seed)
    if df.empty:
        print("라벨된 크롭이 없습니다 (label 열에 normal/crack 이 필요).")
        return 1
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(pd.crosstab(df["split"], df["cls"]).to_markdown())
    print(f"eggs: {df['egg_id'].nunique()}, crops: {len(df)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
