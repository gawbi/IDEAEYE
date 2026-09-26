"""Phase 3-A: YOLO11 검출기 학습 + test 평가 (대분류 mAP, 클래스별 P/R, 위치 검출 전용 single-class 평가).

python scripts/train_det.py --config exp_det [--lights MONO] [--epochs 1 --fraction 0.05]  (스모크)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import CONFIG_DIR, load_config  # noqa: E402
from src.eval.tracker import ExperimentTracker  # noqa: E402


def metrics_from_results(res, names: list[str]) -> tuple[dict, dict]:
    box = res.box
    overall = {"mAP50": float(box.map50), "mAP50-95": float(box.map), "precision": float(box.mp), "recall": float(box.mr)}
    per_class = {}
    for i, name in enumerate(names):
        if i in box.ap_class_index:
            j = list(box.ap_class_index).index(i)
            per_class[name] = {"precision": float(box.p[j]), "recall": float(box.r[j]), "f1": float(box.f1[j]), "ap50": float(box.ap50[j]), "support": 0}
        else:
            per_class[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap50": 0.0, "support": 0}
    return overall, per_class


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_det")
    ap.add_argument("--lights", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=1.0, help="학습 데이터 비율 (스모크 테스트용)")
    ap.add_argument("--resume", action="store_true", help="last.pt 가 있으면 이어서 학습, 이미 끝났으면 학습 생략")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    from ultralytics import YOLO

    for light in args.lights or cfg["lights"]:
        data_yaml = Path(cfg["det_root"]) / light / "data.yaml"
        run_name = f"{cfg['name']}_{light}"
        tracker = ExperimentTracker(run_name, {**cfg, "light": light, "data": str(data_yaml), "fraction": args.fraction, "resume": args.resume}, seed=cfg["seed"], notes="Phase 3-A YOLO detector")
        run_dir = Path(cfg["project"]) / run_name
        last, results_csv = run_dir / "weights" / "last.pt", run_dir / "results.csv"
        done_epochs = max(0, len(results_csv.read_text(encoding="utf-8").splitlines()) - 1) if results_csv.exists() else 0
        if args.resume and last.exists() and done_epochs >= cfg["epochs"]:
            print(f"[{light}] already trained ({done_epochs} epochs) → skip training")
        elif args.resume and last.exists():
            print(f"[{light}] resume from epoch {done_epochs}")
            YOLO(str(last)).train(resume=True)
        else:
            model = YOLO(cfg["model"])
            model.train(
                data=str(data_yaml), imgsz=cfg["imgsz"], epochs=cfg["epochs"], patience=cfg["patience"], batch=cfg["batch"],
                workers=cfg["workers"], seed=cfg["seed"], project=cfg["project"], name=run_name, exist_ok=True, amp=cfg.get("amp", True),
                fraction=args.fraction, verbose=False, plots=False,
            )
        best = run_dir / "weights" / "best.pt"
        model = YOLO(str(best))
        names = [model.names[i] for i in range(len(model.names))]
        res = model.val(data=str(data_yaml), split="test", imgsz=cfg["imgsz"], batch=cfg["batch"], project=cfg["project"], name=run_name + "_test", exist_ok=True, plots=False, verbose=False)
        overall, per_class = metrics_from_results(res, names)
        res1 = model.val(data=str(data_yaml), split="test", imgsz=cfg["imgsz"], batch=cfg["batch"], single_cls=True, project=cfg["project"], name=run_name + "_test_single", exist_ok=True, plots=False, verbose=False)
        egg_loc = {"mAP50": float(res1.box.map50), "precision": float(res1.box.mp), "recall": float(res1.box.mr)}
        speed = dict(res.speed)  # ms per image: preprocess/inference/postprocess
        headline = {"mAP50": overall["mAP50"], "egg_loc_recall": egg_loc["recall"], "infer_ms": speed.get("inference", 0.0)}
        tracker.log_metrics(per_class, by={"state(대분류)": per_class}, headline=headline)
        (tracker.dir / "summary.json").write_text(json.dumps({"overall": overall, "per_class": per_class, "egg_localization_single_cls": egg_loc, "speed_ms": speed, "weights": str(best)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{light}] test mAP50={overall['mAP50']:.3f} mAP50-95={overall['mAP50-95']:.3f} egg-loc recall={egg_loc['recall']:.3f} infer={speed.get('inference', 0):.1f}ms → {tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
