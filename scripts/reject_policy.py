"""배출 정책 시뮬레이션: "정상만 통과, 나머지(크랙·혈반·탈색·이물질·기형) 전부 배출" 정책에서
결함 Recall 목표별 정상란 오배출률과 크랙 Recall 을 계산한다.

배출 점수 = 1 - p(정상). 양성 = 정상이 아닌 모든 크롭, 음성 = 정상(state 1).
python scripts/reject_policy.py --exp latest:cls_mono_3cls_effb0 latest:cls_mono_effb0 latest:abl_mono_combined → docs/reject_policy.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import PROJECT_ROOT  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402
from src.eval.metrics import operating_point  # noqa: E402


def analyze(exp: Path, targets=(0.90, 0.95, 0.98, 0.99)) -> dict:
    ev = json.loads((exp / "eval.json").read_text(encoding="utf-8"))
    classes = ev["classes"]
    probs = np.load(exp / "test_probs.npy").astype(np.float32)
    pred = pd.read_csv(exp / "test_predictions.csv", dtype=str, encoding="utf-8-sig")
    label_col = next(c for c in ("label3", "cls", "state") if c in pred)
    y = pred[label_col].astype(str).to_numpy()
    normal_idx = [i for i, c in enumerate(classes) if c.startswith("1/") or c == "1"]
    if not normal_idx:
        return {"exp": exp.name, "error": "no normal class"}
    reject_score = 1.0 - probs[:, normal_idx].sum(1)
    is_defect = ~np.array([c.startswith("1/") or c == "1" for c in y])
    is_crack = np.array([c.startswith("2/") or c == "2" for c in y])
    out = {"exp": exp.name, "n_normal": int((~is_defect).sum()), "n_defect": int(is_defect.sum()), "n_crack": int(is_crack.sum()), "points": {}}
    for t in targets:
        op = operating_point(reject_score, is_defect, t)
        thr = op["threshold"]
        out["points"][f"defect_recall@{t:.2f}"] = {
            "threshold": thr,
            "normal_reject_rate": op["fpr"],
            "defect_recall": op["recall"],
            "crack_recall": float((reject_score[is_crack] >= thr).mean()) if is_crack.any() else None,
        }
    # 크랙만 놓치지 않는 기준: 크랙 Recall 95% 를 만족하는 임계값에서 정상 오배출률
    opc = operating_point(reject_score, is_crack, 0.95)
    out["crack_recall95"] = {"threshold": opc["threshold"], "normal_reject_rate": float((reject_score[~is_defect] >= opc["threshold"]).mean()), "defect_recall": float((reject_score[is_defect] >= opc["threshold"]).mean())}
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--out", default="docs/reject_policy.md")
    args = ap.parse_args()
    results = []
    for spec in args.exp:
        try:
            results.append(analyze(resolve_experiment(spec)))
        except FileNotFoundError as e:
            print("skip:", e)
    lines = ["# 배출 정책 시뮬레이션 (자동 생성: scripts/reject_policy.py)", "",
             "정책: 정상만 통과, 그 외(크랙·혈반·탈색·이물질·기형) 전부 배출. 배출 점수 = 1 − p(정상). test 분할 기준, 클래스명은 코드 가설.", "",
             "| 모델 | 정상 n | 결함 n | 결함 Recall 90% | 95% | 98% | 99% | 크랙 Recall 95% 기준 |", "|---|---:|---:|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            continue
        cells = []
        for t in ("0.90", "0.95", "0.98", "0.99"):
            p = r["points"][f"defect_recall@{t}"]
            cells.append(f"정상 오배출 {p['normal_reject_rate']:.1%} / 크랙 R {p['crack_recall']:.2f}")
        c = r["crack_recall95"]
        lines.append(f"| {r['exp']} | {r['n_normal']:,} | {r['n_defect']:,} | " + " | ".join(cells) + f" | 정상 오배출 {c['normal_reject_rate']:.1%}, 결함 R {c['defect_recall']:.2f} |")
    lines += ["", "읽는 법: '결함 Recall 95%' 열은 모든 결함의 95% 를 배출할 때 정상란이 잘못 배출되는 비율과 그때 크랙이 잡히는 비율. 마지막 열은 크랙만 95% 잡도록 임계값을 정했을 때의 정상 오배출률."]
    out = PROJECT_ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    for r in results:
        (PROJECT_ROOT / "experiments" / r["exp"] / "reject_policy.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
