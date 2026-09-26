"""Phase 3-C: PatchCore-lite 정상 vs 크랙. python scripts/train_patchcore.py --config exp_patchcore_mono [--smoke]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.eval.metrics import operating_point  # noqa: E402
from src.eval.tracker import ExperimentTracker  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_patchcore_mono")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg.update({"n_normal_train": 64, "n_normal_test": 128, "n_crack_test": 128, "name": cfg["name"] + "_smoke"})

    import torch
    from torch.utils.data import DataLoader

    from src.models.patchcore import PatchCoreLite, auroc
    from src.train.classifier import CropDataset, load_index

    df = load_index(resolve_path(cfg["index_csv"]), cfg["light"])
    normal = df[df["state"] == "1"]
    crack = df[df["state"] == "2"]
    tr = normal[normal["split"] == "train"].sample(n=min(cfg["n_normal_train"], (normal["split"] == "train").sum()), random_state=cfg["seed"])
    te_n = normal[normal["split"] == "test"].sample(n=min(cfg["n_normal_test"], (normal["split"] == "test").sum()), random_state=cfg["seed"])
    te_c = crack[crack["split"] == "test"]
    if cfg.get("n_crack_test"):
        te_c = te_c.sample(n=min(cfg["n_crack_test"], len(te_c)), random_state=cfg["seed"])
    te = pd.concat([te_n, te_c]).reset_index(drop=True)
    classes = sorted(te["cls"].unique(), key=lambda v: tuple(int(x) for x in v.split("/")))
    print(f"memory bank normal={len(tr)}  test normal={len(te_n)} crack={len(te_c)}")

    tracker = ExperimentTracker(cfg["name"], cfg, seed=cfg["seed"], notes="Phase 3-C PatchCore-lite normal-only")
    mk = lambda d: DataLoader(CropDataset(d, classes if d is te else sorted(d["cls"].unique()), cfg["img_size"], False), batch_size=cfg["batch"], num_workers=cfg["workers"])
    pc = PatchCoreLite(cfg["backbone"], tuple(cfg["layers"]), cfg["proj_dim"], cfg["coreset_ratio"], seed=cfg["seed"])
    pc.fit(mk(tr))
    scores, y_idx = pc.score(mk(te))
    y_cls = np.array([classes[i] for i in y_idx])
    is_crack = np.char.startswith(y_cls.astype(str), "2/")

    result = {"auroc": auroc(scores, is_crack), "n_memory": int(pc.memory.shape[0]), "operating_points": {}}
    for t in (0.90, 0.95, 0.98):
        op = operating_point(scores, is_crack, t)
        thr = op["threshold"]
        op["subclass_recall"] = {c: float((scores[y_cls == c] >= thr).mean()) for c in classes if c.startswith("2/") and (y_cls == c).any()}
        result["operating_points"][f"recall@{t:.2f}"] = op
    pd.DataFrame({"crop_path": te["crop_path"], "cls": y_cls, "score": scores}).to_csv(tracker.dir / "test_scores.csv", index=False, encoding="utf-8-sig")
    (tracker.dir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    op95 = result["operating_points"]["recall@0.95"]
    per_class = {c: {"precision": 0.0, "recall": op95["subclass_recall"].get(c, 0.0), "f1": 0.0, "support": int((y_cls == c).sum())} for c in classes}
    tracker.log_metrics(per_class, headline={"auroc": result["auroc"], "normal_fpr@crack_recall95": op95["fpr"]})
    print(json.dumps({k: v for k, v in result.items() if k != "operating_points"}, indent=2), "\n", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "subclass_recall"} for k, v in result["operating_points"].items()}, indent=2))
    print(f"→ {tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
