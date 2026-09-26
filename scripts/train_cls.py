"""크롭 분류기 학습 + test 평가 + experiments/ 기록.

python scripts/train_cls.py --config exp_cls_mono
python scripts/train_cls.py --config exp_cls_mono --override loss=focal preprocess=clahe name=cls_mono_focal_clahe   (ablation)
python scripts/train_cls.py --config exp_cls_mono --smoke   (train 2k, 1 epoch)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.eval.tracker import ExperimentTracker  # noqa: E402


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    import yaml

    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=None, help="key=value ...")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    if args.smoke:
        cfg.update({"epochs": 1, "train_cap_per_class": 100, "val_max": 500, "test_max": 1000, "name": cfg["name"] + "_smoke"})

    import torch

    from src.train.classifier import benchmark_latency, train_classifier

    tracker = ExperimentTracker(cfg["name"], cfg, seed=cfg["seed"], notes=f"crop classifier {cfg['light']} loss={cfg.get('loss')} preprocess={cfg.get('preprocess')}")
    ev = train_classifier(cfg, resolve_path(cfg["index_csv"]), tracker.dir)

    ops = ev.get("operating_points", {})
    op95 = ops.get("recall@0.95", {})
    headline = {"acc": ev["accuracy"], "macro_recall": ev["macro_recall"]}
    if op95:
        headline["normal_fpr@crack_recall95"] = op95["fpr"]
    for key in ("2/6", "2/8", "3/1"):
        if key in ev["overall"]:
            headline[f"recall_{key}"] = ev["overall"][key]["recall"]
    tracker.log_metrics(ev["overall"], by=ev.get("by", {}), headline=headline)

    # 지연 측정 (이 GPU 기준, PyTorch FP16, batch 1 / 16)
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(tracker.dir / "model.pt", map_location=device, weights_only=False)
    model = timm.create_model(cfg["model"], pretrained=False, num_classes=len(ck["classes"])).to(device).half() if device.type == "cuda" else timm.create_model(cfg["model"], pretrained=False, num_classes=len(ck["classes"]))
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ck["state_dict"].items()})
    lat = {f"batch{b}": benchmark_latency(model, cfg["img_size"], device, b) for b in (1, 16)} if device.type == "cuda" else {}
    (tracker.dir / "latency.json").write_text(json.dumps({"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", **lat}, indent=2), encoding="utf-8")
    print(json.dumps({"headline": headline, "operating_points": {k: {kk: vv for kk, vv in v.items() if kk != "subclass_recall"} for k, v in ops.items()}, "latency": lat}, ensure_ascii=False, indent=2))
    print(f"→ {tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
