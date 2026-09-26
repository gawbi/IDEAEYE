"""큐가 끝난 뒤 결과를 읽고 후속 실험을 스스로 고른다 (Claude 개입 없이 기계가 이어감).

규칙:
 1. 대분류(6클래스) 분류기와 크랙 vs 정상 2클래스 분류기는 항상 실행 (운영점 KPI 직접 대상)
 2. ablation 중 베이스라인보다 "정상 FPR @ 파각 Recall 95%" 를 낮춘 변수를 모두 조합한 설정을 1회 실행
 3. 끝나면 collect_results
python scripts/auto_next.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import PROJECT_ROOT  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402


def fpr95(name: str) -> float | None:
    try:
        ev = json.loads((resolve_experiment(f"latest:{name}") / "eval.json").read_text(encoding="utf-8"))
        return ev["operating_points"]["recall@0.95"]["fpr"]
    except (FileNotFoundError, KeyError):
        return None


def run(cmd: str, dry: bool) -> int:
    print("===", cmd, flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT).returncode


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = fpr95("cls_mono_effb0")
    abl = {"img_size=320 batch=64": fpr95("abl_mono_res320"), "preprocess=clahe": fpr95("abl_mono_clahe"), "loss=weighted_focal": fpr95("abl_mono_wfocal")}
    print("baseline FPR@95:", base, "| ablations:", abl)
    improved = [k for k, v in abl.items() if v is not None and base is not None and v < base - 0.01]

    cmds = [
        "python scripts/train_cls.py --config exp_cls_mono_state",
        "python scripts/train_cls.py --config exp_cls_mono_crackbin",
        "python scripts/make_alt_split_color.py",
        "python scripts/train_cls.py --config exp_cls_color_wash",
    ]
    if improved:
        cmds.append("python scripts/train_cls.py --config exp_cls_mono --override " + " ".join(improved) + " name=abl_mono_combined")
    else:
        print("no ablation improved FPR@95 by >1%p → combined run skipped")
    cmds.append("python scripts/ablation_table.py")
    cmds.append("python scripts/collect_results.py")

    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status = {"improved": improved, "jobs": []}
    for c in cmds:
        rc = run(c, args.dry_run)
        status["jobs"].append({"cmd": c, "rc": rc})
        (log_dir / "auto_next_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
