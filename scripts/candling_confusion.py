"""Phase 5-3: 검란 결과 정리 — 혈반육반(3/1) vs 먹통(3/5) vs 정상 혼동행렬, 혈반 점수 AUROC, 전처리 ablation 비교.

python scripts/candling_confusion.py --exp latest:cls_candling_none latest:cls_candling_clahe ...
출력: docs/phase5_candling.md (표), 각 실험 폴더에 candling_confusion.json
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
from src.models.patchcore import auroc  # noqa: E402

BLOOD, DARK = "3/1", "3/5"


def group_of(c: str) -> str:
    if c == BLOOD:
        return "혈반육반"
    if c == DARK:
        return "먹통"
    if c.startswith("3/"):
        return "혈반기타(3/2-4)"
    if c.startswith("1/"):
        return "정상"
    return "기타"


def analyze(exp: Path) -> dict:
    ev = json.loads((exp / "eval.json").read_text(encoding="utf-8"))
    classes = ev["classes"]
    pred = pd.read_csv(exp / "test_predictions.csv", dtype={"cls": str, "y_pred": str}, encoding="utf-8-sig")
    probs = np.load(exp / "test_probs.npy").astype(np.float32)
    assert len(pred) == len(probs)
    pred["g_true"], pred["g_pred"] = pred["cls"].map(group_of), pred["y_pred"].map(group_of)
    order = ["정상", "혈반육반", "먹통", "혈반기타(3/2-4)", "기타"]
    cm = pd.crosstab(pred["g_true"], pred["g_pred"]).reindex(index=[o for o in order if o in set(pred["g_true"])], columns=[o for o in order if o in set(pred["g_pred"])], fill_value=0)
    out: dict = {"exp": exp.name, "preprocess": None}
    try:
        import yaml

        out["preprocess"] = yaml.safe_load((exp / "config.yaml").read_text(encoding="utf-8")).get("preprocess")
    except Exception:
        pass
    out["confusion"] = cm.to_dict()
    out["confusion_md"] = cm.to_markdown()
    # 혈반 점수 = p(3/1). 양성 혈반육반, 음성 정상 / 음성 먹통
    if BLOOD in classes:
        s = probs[:, classes.index(BLOOD)]
        is_blood = (pred["cls"] == BLOOD).to_numpy()
        is_norm = pred["cls"].str.startswith("1/").to_numpy()
        is_dark = (pred["cls"] == DARK).to_numpy()
        m1, m2 = is_blood | is_norm, is_blood | is_dark
        out["auroc_blood_vs_normal"] = auroc(s[m1], is_blood[m1])
        out["auroc_blood_vs_dark"] = auroc(s[m2], is_blood[m2]) if is_dark.any() else None
        out["operating_points_vs_normal"] = {f"recall@{t:.2f}": operating_point(s[m1], is_blood[m1], t) for t in (0.80, 0.88, 0.95)}
        out["blood_recall_argmax"] = float((pred.loc[is_blood, "y_pred"] == BLOOD).mean())
        out["blood_as_dark_rate"] = float((pred.loc[is_blood, "y_pred"] == DARK).mean())
        out["dark_as_blood_rate"] = float((pred.loc[is_dark, "y_pred"] == BLOOD).mean()) if is_dark.any() else None
        out["n"] = {"blood": int(is_blood.sum()), "dark": int(is_dark.sum()), "normal": int(is_norm.sum())}
    (exp / "candling_confusion.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--out", default="docs/phase5_candling.md")
    args = ap.parse_args()

    results = []
    for spec in args.exp:
        try:
            results.append(analyze(resolve_experiment(spec)))
        except FileNotFoundError as e:
            print("skip:", e)
    lines = ["# Phase 5 검란 결과 (자동 생성: scripts/candling_confusion.py)", "", "코드 가설: 3/1 혈반육반, 3/5 먹통 (configs/class_codes.yaml). 양성=혈반육반.", ""]
    lines += ["| 실험 | 전처리 | AUROC vs 정상 | AUROC vs 먹통 | 정상 FPR @ 혈반 Recall 80/88/95% | argmax Recall | 혈반→먹통 오분류 | 먹통→혈반 오분류 |", "|---|---|---:|---:|---|---:|---:|---:|"]
    for r in results:
        if "auroc_blood_vs_normal" not in r:
            continue
        ops = r["operating_points_vs_normal"]
        fpr = "/".join(f"{ops[k]['fpr']:.3f}" for k in ("recall@0.80", "recall@0.88", "recall@0.95"))
        d = r["auroc_blood_vs_dark"]
        lines.append(f"| {r['exp']} | {r.get('preprocess')} | {r['auroc_blood_vs_normal']:.3f} | {d:.3f} | {fpr} | {r['blood_recall_argmax']:.3f} | {r['blood_as_dark_rate']:.3f} | {r['dark_as_blood_rate'] if r['dark_as_blood_rate'] is None else round(r['dark_as_blood_rate'], 3)} |" if d is not None else f"| {r['exp']} | {r.get('preprocess')} | {r['auroc_blood_vs_normal']:.3f} | - | {fpr} | {r['blood_recall_argmax']:.3f} | {r['blood_as_dark_rate']:.3f} | - |")
    for r in results:
        lines += ["", f"## 혼동행렬 — {r['exp']}", "", r["confusion_md"]]
    out = PROJECT_ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
