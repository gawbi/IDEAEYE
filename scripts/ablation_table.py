"""Phase 4: ablation 누적 표 — 베이스라인 대비 각 실험의 소분류 Recall·운영점 변화.

python scripts/ablation_table.py [--base cls_mono_effb0] [--exps abl_mono_res320 abl_mono_clahe ...] → docs/phase4_ablation.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import PROJECT_ROOT  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402

TARGETS = [("2/6", "잠재파형크랙"), ("2/8", "미세크랙"), ("2/7", "파형크랙?"), ("2/2", "일자형크랙?"), ("3/1", "혈반육반"), ("3/5", "먹통")]
DEFAULT_EXPS = ["abl_mono_res320", "abl_mono_clahe", "abl_mono_wfocal", "abl_mono_combined", "cls_mono_state_effb0", "cls_mono_crackbin_effb0"]


def load(name: str) -> tuple[dict, dict] | None:
    try:
        d = resolve_experiment(f"latest:{name}")
        ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        return ev, cfg
    except (FileNotFoundError, KeyError):
        return None


def variable(cfg: dict, base_cfg: dict) -> str:
    diffs = [f"{k}={cfg.get(k)}" for k in ("img_size", "preprocess", "loss", "label_col", "class_filter", "epochs", "model") if cfg.get(k) != base_cfg.get(k)]
    return ", ".join(diffs) or "(baseline)"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cls_mono_effb0")
    ap.add_argument("--exps", nargs="*", default=DEFAULT_EXPS)
    ap.add_argument("--out", default="docs/phase4_ablation.md")
    args = ap.parse_args()

    base = load(args.base)
    if base is None:
        print("baseline missing")
        return 1
    bev, bcfg = base
    rows = [(args.base, bev, bcfg)] + [(n, *r) for n in args.exps if (r := load(n)) is not None]

    def fpr(ev, k):
        return ev.get("operating_points", {}).get(k, {}).get("fpr")

    lines = ["# Phase 4 ablation 누적 표 (자동 생성: scripts/ablation_table.py)", "",
             "한 번에 한 변수만 변경. 값은 test 기준, 괄호는 베이스라인 대비 변화(%p). 클래스명은 코드 가설.", "",
             "| 실험 | 변경 변수 | 정상 FPR @ 파각 Recall 90 / 95 / 98% | macro Recall | " + " | ".join(f"R {n}" for _, n in TARGETS) + " |",
             "|---|---|---|---:|" + "---:|" * len(TARGETS)]
    for name, ev, cfg in rows:
        cells = []
        for k in ("recall@0.90", "recall@0.95", "recall@0.98"):
            v, b = fpr(ev, k), fpr(bev, k)
            cells.append("-" if v is None else (f"{v:.3f}" if name == args.base or b is None else f"{v:.3f} ({(v - b) * 100:+.1f})"))
        recs = []
        for code, _ in TARGETS:
            v = ev["overall"].get(code, {}).get("recall")
            b = bev["overall"].get(code, {}).get("recall")
            if v is None:
                recs.append("-")
            elif name == args.base or b is None:
                recs.append(f"{v:.3f}")
            else:
                recs.append(f"{v:.3f} ({(v - b) * 100:+.1f})")
        mr = ev["macro_recall"]
        mr_s = f"{mr:.3f}" if name == args.base else f"{mr:.3f} ({(mr - bev['macro_recall']) * 100:+.1f})"
        lines.append(f"| {name} | {variable(cfg, bcfg)} | {' / '.join(cells)} | {mr_s} | " + " | ".join(recs) + " |")
    lines += ["", "해석 규칙: 운영 KPI 는 'Recall 95% 에서 정상 FPR'. 소분류 Recall(argmax) 은 33클래스 혼동을 포함하므로 보조 지표. 2클래스(크랙 vs 정상) 실험은 소분류 Recall 이 없어 '-' 로 표시."]
    out = PROJECT_ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
