"""Phase 6: 자사 라인 데이터 도메인 적응 (데이터가 들어오면 실행).

전제: data/own/index.csv — 열: crop_path, cls(AI Hub 코드 체계 "state/state_detail"), split(train|test), light, [washed, breed, region_code]
  (판 이미지+xml 로 들어오면 먼저 scripts/make_crops.py 와 같은 방식으로 크롭·index 를 만든다. configs/data.yaml own_root 참고)

1. zero-shot: AI Hub 로 학습한 모델을 자사 test 에 그대로 적용
2. 학습 곡선: 자사 train 에서 100/300/1000/… 장으로 미세조정 → 고정된 자사 test 로 측정
3. flat-field 보정 전후 비교 (--flatfield)

python scripts/own_domain_adapt.py --base latest:cls_mono_effb0 --sizes 100 300 1000 [--flatfield]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import PROJECT_ROOT, resolve_path  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402
from src.eval.tracker import ExperimentTracker  # noqa: E402


def apply_flatfield(index: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    from src.datasets.flatfield import flat_field_correct
    from src.train.classifier import imread

    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in index["crop_path"]:
        img = imread(p)
        q = out_dir / Path(p).name
        if img is not None:
            ok, buf = cv2.imencode(".jpg", flat_field_correct(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            buf.tofile(str(q))
        paths.append(str(q))
    out = index.copy()
    out["crop_path"] = paths
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="AI Hub 로 학습한 분류기 실험 (latest:이름 또는 경로)")
    ap.add_argument("--own-index", default="data/own/index.csv")
    ap.add_argument("--sizes", nargs="*", type=int, default=[100, 300, 1000, 3000])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--flatfield", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    own_path = resolve_path(args.own_index)
    if not own_path.exists():
        print(f"자사 데이터 index 없음: {own_path}\n  → data/own/ 에 크롭과 index.csv(crop_path, cls, split, light) 를 준비하면 실행됩니다.")
        return 2

    import torch

    from src.train.classifier import CropDataset, evaluate, load_index, predict, train_classifier
    from torch.utils.data import DataLoader

    base = resolve_experiment(args.base)
    ck = torch.load(base / "model.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    classes = ck["classes"]
    own = pd.read_csv(own_path, dtype={"cls": str, "split": str}, encoding="utf-8-sig")
    own["exists"] = [Path(p).exists() for p in own["crop_path"]]
    own = own[own["exists"] & own["cls"].isin(classes)]
    if args.flatfield:
        own = apply_flatfield(own, resolve_path("data/own/crops_flatfield"))
    test = own[own["split"] == "test"]
    train = own[own["split"] == "train"]
    print(f"own train={len(train)} test={len(test)} classes(in base)={len(classes)}")

    tracker = ExperimentTracker(f"own_adapt_{cfg['light']}{'_ff' if args.flatfield else ''}", {**cfg, "base": str(base), "sizes": args.sizes, "flatfield": args.flatfield}, seed=args.seed, notes="Phase 6 domain adaptation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import timm

    # 1. zero-shot
    model = timm.create_model(cfg["model"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    loader = DataLoader(CropDataset(test, classes, cfg["img_size"], False, cfg.get("preprocess", "none")), batch_size=cfg["batch"], num_workers=4)
    probs, ys = predict(model, loader, device)
    curve = [{"n_train": 0, **{k: v for k, v in evaluate(probs, ys, classes).items() if k in ("accuracy", "macro_recall")}}]
    print("zero-shot:", curve[-1])

    # 2. 학습 곡선: 자사 train 서브셋으로 미세조정 (AI Hub 가중치에서 시작)
    tmp_index = tracker.dir / "own_index_tmp.csv"
    for n in args.sizes:
        sub = train.sample(n=min(n, len(train)), random_state=args.seed)
        idx = pd.concat([sub.assign(split="train"), sub.sample(frac=0.2, random_state=args.seed).assign(split="val"), test.assign(split="test")])
        idx["light"] = cfg["light"]
        idx["exists"] = True
        idx.to_csv(tmp_index, index=False, encoding="utf-8-sig")
        run_cfg = {**cfg, "epochs": args.epochs, "train_cap_per_class": None, "val_max": None, "test_max": None, "lr": cfg["lr"] * 0.3, "init_from": str(base / "model.pt")}
        ev = train_classifier(run_cfg, tmp_index, tracker.dir / f"finetune_{n}", device)
        curve.append({"n_train": len(sub), "accuracy": ev["accuracy"], "macro_recall": ev["macro_recall"]})
        print(f"finetune n={len(sub)}:", curve[-1])
    (tracker.dir / "learning_curve.json").write_text(json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.log_metrics({}, headline={"zero_shot_macro_recall": curve[0]["macro_recall"], "best_macro_recall": max(c["macro_recall"] for c in curve)})
    print(f"→ {tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
