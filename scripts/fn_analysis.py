"""Phase 4: 크랙 미검출(FN) 분석. 분류기 test_predictions.csv 에서 파각 Recall 95% 운영점 기준 미검출 크롭을 소분류별로 모은다.

python scripts/fn_analysis.py --exp <experiments/…_cls_mono_effb0 디렉터리> [--per-class 40]
출력: outputs/fn_analysis/<subclass>/*.jpg, outputs/fn_analysis/summary.md (소분류별 FN 수·밝기·크기·촬영일 분포, 가설)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import resolve_path  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402
from src.train.classifier import imread  # noqa: E402


def brightness_stats(path: str) -> dict[str, float]:
    img = imread(path)
    if img is None:
        return {"mean": np.nan, "std": np.nan, "sat_hi": np.nan, "sat_lo": np.nan, "w": 0, "h": 0}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"mean": float(g.mean()), "std": float(g.std()), "sat_hi": float((g >= 250).mean()), "sat_lo": float((g <= 5).mean()), "w": img.shape[1], "h": img.shape[0]}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--out", default="outputs/fn_analysis")
    args = ap.parse_args()

    exp = resolve_experiment(args.exp)
    ev = json.loads((exp / "eval.json").read_text(encoding="utf-8"))
    thr = ev["operating_points"]["recall@0.95"]["threshold"]
    pred = pd.read_csv(exp / "test_predictions.csv", dtype={"cls": str, "y_pred": str, "washed": str, "region_code": str}, encoding="utf-8-sig")
    crack = pred[pred["cls"].str.startswith("2/")].copy()
    crack["detected"] = crack["p_crack"] >= thr
    fn = crack[~crack["detected"]].copy()
    out = resolve_path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stats = pd.DataFrame([brightness_stats(p) for p in crack["crop_path"]])
    crack = pd.concat([crack.reset_index(drop=True), stats], axis=1)
    fn = crack[~crack["detected"]]

    lines = [f"# 크랙 미검출(FN) 분석 — {exp.name}", "", f"운영점: 파각 Recall 95% → 임계값 {thr:.4f}, 전체 크랙 {len(crack)} 중 FN {len(fn)} ({len(fn) / max(1, len(crack)):.1%})", ""]
    lines += ["## 소분류별 FN", "", "| subclass | total | FN | FN율 | FN 밝기평균 | 검출 밝기평균 | FN 과노출비율 | FN 예측 상위 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for cls, g in crack.groupby("cls"):
        f = g[~g["detected"]]
        top = f["y_pred"].value_counts().head(3).to_dict()
        lines.append(f"| {cls} | {len(g)} | {len(f)} | {len(f) / len(g):.1%} | {f['mean'].mean():.1f} | {g[g['detected']]['mean'].mean():.1f} | {f['sat_hi'].mean():.3f} | {top} |")
        d = out / cls.replace("/", "_")
        d.mkdir(exist_ok=True)
        for p in f["crop_path"].head(args.per_class):
            try:
                shutil.copy(p, d / Path(p).name)
            except OSError:
                pass

    lines += ["", "## 가설 점검 지표", ""]
    det, miss = crack[crack["detected"]], fn
    def row(name, a, b):
        lines.append(f"| {name} | {a:.3f} | {b:.3f} |")
    lines += ["| 지표 | 검출 크랙 | 미검출 크랙 |", "|---|---:|---:|"]
    row("밝기 평균 (노출)", det["mean"].mean(), miss["mean"].mean())
    row("밝기 표준편차 (대비)", det["std"].mean(), miss["std"].mean())
    row("과노출 픽셀 비율 (>=250)", det["sat_hi"].mean(), miss["sat_hi"].mean())
    row("암부 픽셀 비율 (<=5)", det["sat_lo"].mean(), miss["sat_lo"].mean())
    row("크롭 긴 변 px", det[["w", "h"]].max(axis=1).mean(), miss[["w", "h"]].max(axis=1).mean())
    if "washed" in crack:
        lines += ["", "세척 코드별 FN율: " + ", ".join(f"{k}: {1 - v:.1%}" for k, v in crack.groupby("washed")["detected"].mean().items())]
    if "region_code" in crack:
        lines += ["지역코드별 FN율: " + ", ".join(f"{k}: {1 - v:.1%}" for k, v in crack.groupby("region_code")["detected"].mean().items())]
    lines += ["", "## 미검출이 '정상'으로 예측된 비율", "", f"{(fn['y_pred'].str.startswith('1/')).mean():.1%} 가 정상 소분류로, {(fn['y_pred'].str.startswith('4/')).mean():.1%} 가 탈색으로 예측됨"]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    crack.to_csv(out / "crack_test_with_stats.csv", index=False, encoding="utf-8-sig")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
