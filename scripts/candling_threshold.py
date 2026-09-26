"""Phase 5-1: 임계값 기반 혈반 검출 베이스라인 평가.

양성 = 혈반육반(3/1), 음성 = 정상(1/*). 추가로 먹통(3/5) 과의 분리도도 보고.
python scripts/candling_threshold.py [--n-neg 6000] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import resolve_path  # noqa: E402
from src.eval.candling_baseline import blood_spot_score  # noqa: E402
from src.eval.metrics import operating_point  # noqa: E402
from src.eval.tracker import ExperimentTracker  # noqa: E402
from src.models.patchcore import auroc  # noqa: E402
from src.train.classifier import imread, load_index  # noqa: E402


def _score(path: str) -> dict[str, float]:
    img = imread(path)
    if img is None:
        return {"max": 0.0, "area": 0.0, "contrast": 0.0, "egg_px": 0.0}
    return blood_spot_score(img)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neg", type=int, default=6000)
    ap.add_argument("--n-dark", type=int, default=3000, help="먹통(3/5) 샘플 수")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_index(resolve_path("data/crops/index.csv"), "MONO")
    test = df[df["split"] == "test"]
    pos = test[test["cls"] == "3/1"]
    neg = test[test["state"] == "1"].sample(n=min(args.n_neg, (test["state"] == "1").sum()), random_state=args.seed)
    dark = test[test["cls"] == "3/5"].sample(n=min(args.n_dark, (test["cls"] == "3/5").sum()), random_state=args.seed)
    sub = pd.concat([pos.assign(group="blood"), neg.assign(group="normal"), dark.assign(group="dark")]).reset_index(drop=True)
    print(f"blood={len(pos)} normal={len(neg)} dark(먹통)={len(dark)}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        feats = list(ex.map(_score, sub["crop_path"].tolist(), chunksize=64))
    F = pd.DataFrame(feats)
    sub = pd.concat([sub, F], axis=1)

    tracker = ExperimentTracker("candling_threshold_baseline", {"n_neg": args.n_neg, "n_dark": args.n_dark, "feature": "blackhat relative"}, seed=args.seed, notes="Phase 5 threshold baseline (no DL)")
    result: dict = {"n": sub["group"].value_counts().to_dict(), "features": {}}
    for feat in ("max", "area", "contrast"):
        s = sub[feat].to_numpy()
        blood_vs_normal = auroc(s[sub["group"] != "dark"], (sub["group"][sub["group"] != "dark"] == "blood").to_numpy())
        blood_vs_dark = auroc(s[sub["group"] != "normal"], (sub["group"][sub["group"] != "normal"] == "blood").to_numpy())
        ops = {}
        m = sub["group"] != "dark"
        for t in (0.80, 0.88, 0.95):
            ops[f"recall@{t:.2f}"] = operating_point(s[m], (sub["group"][m] == "blood").to_numpy(), t)
        result["features"][feat] = {"auroc_blood_vs_normal": blood_vs_normal, "auroc_blood_vs_dark": blood_vs_dark, "operating_points_vs_normal": ops}
    best = max(result["features"], key=lambda f: result["features"][f]["auroc_blood_vs_normal"])
    result["best_feature"] = best
    sub.to_csv(tracker.dir / "scores.csv", index=False, encoding="utf-8-sig")
    (tracker.dir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    b = result["features"][best]
    tracker.log_metrics({"blood_vs_normal": {"precision": 0.0, "recall": b["operating_points_vs_normal"]["recall@0.88"]["recall"], "f1": 0.0, "support": int(len(pos))}},
                        headline={"auroc_blood_vs_normal": b["auroc_blood_vs_normal"], "auroc_blood_vs_dark": b["auroc_blood_vs_dark"], "normal_fpr@blood_recall88": b["operating_points_vs_normal"]["recall@0.88"]["fpr"]})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"→ {tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
