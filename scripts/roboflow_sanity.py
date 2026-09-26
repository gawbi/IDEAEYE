"""로보플로우 보조 데이터로 크랙 모델 sanity check (도메인이 전혀 다른 사진에서도 반응하는지). KPI 아님 (CLAUDE.md 3.2).

각 데이터셋의 YOLO 라벨 bbox 로 계란을 크롭 → (A) 투광 2클래스 크랙 모델, (B) 반사광 COLOR 5클래스 모델로 크랙 점수 산출.
python scripts/roboflow_sanity.py [--mono latest:cls_mono_crackbin_effb0] [--color latest:cls_color_effb0] → docs/roboflow_sanity.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import PROJECT_ROOT, resolve_path  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402
from src.datasets.coords import Box  # noqa: E402
from src.datasets.roboflow_yolo import SPLIT_DIRS, long_path, read_yolo_labels  # noqa: E402
from src.eval.metrics import operating_point  # noqa: E402
from src.infer.adapters import TimmClassifier  # noqa: E402
from src.models.patchcore import auroc  # noqa: E402
from src.train.classifier import imread  # noqa: E402

CRACK_WORDS = ("crack",)


def collect_crops(root: Path, max_images: int | None = None) -> pd.DataFrame:
    d = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = list(d.get("names", []))
    rows = []
    n_img = 0
    for split in SPLIT_DIRS.values():
        img_dir = long_path(root / split / "images")
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = long_path(root / split / "labels" / (img.stem + ".txt"))
            if not lbl.exists():
                continue
            n_img += 1
            if max_images and n_img > max_images:
                break
            for lb in read_yolo_labels(lbl):
                name = names[lb.cls] if lb.cls < len(names) else str(lb.cls)
                rows.append({"image": str(img), "split": split, "label": name, "is_crack": any(w in name.lower() for w in CRACK_WORDS), "cx": lb.cx, "cy": lb.cy, "w": lb.w, "h": lb.h})
    return pd.DataFrame(rows)


def score_dataset(df: pd.DataFrame, model: TimmClassifier, crack_key: str, margin: float = 0.10, batch: int = 64) -> np.ndarray:
    """crack_key: 모델 클래스 중 크랙 클래스 이름 ("2" 또는 "2/0")."""
    ci = model.classes.index(crack_key)
    scores = np.zeros(len(df), dtype=np.float32)
    cache: dict[str, np.ndarray | None] = {}
    idx_by_image = df.groupby("image").indices
    for image, idxs in idx_by_image.items():
        img = cache.get(image)
        if img is None:
            img = imread(image)
            cache = {image: img}
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        for i in idxs:
            r = df.iloc[i]
            x1, y1 = (r.cx - r.w / 2) * W, (r.cy - r.h / 2) * H
            x2, y2 = (r.cx + r.w / 2) * W, (r.cy + r.h / 2) * H
            boxes.append(Box(x1, y1, x2, y2).expanded(margin).clipped(W, H))
        probs = model_probs(model, img, boxes)
        scores[idxs] = probs[:, ci]
    return scores


def model_probs(model: TimmClassifier, image: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """classify_boxes 는 argmax 만 돌려주므로 확률 전체가 필요한 경우 직접 forward."""
    th = model.torch
    size = model.cfg["img_size"]
    dt = th.float16 if model.half else th.float32
    img = th.from_numpy(np.ascontiguousarray(image)).to(model.device)
    img = img[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(dt) / 255.0
    H, W = img.shape[-2:]
    tiles = []
    for b in boxes:
        x1, y1, x2, y2 = b.clipped(W, H).to_int()
        tiles.append(model._letterbox_tensor(img[:, :, y1:y2, x1:x2], size) if x2 > x1 and y2 > y1 else th.zeros((1, 3, size, size), device=model.device, dtype=dt))
    t = th.cat(tiles)
    if model._mean is None:
        model._forward(t[:1])
    t = (t - model._mean) / model._std
    with th.no_grad():
        return th.softmax(model.model(t).float(), dim=1).cpu().numpy()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mono", default="latest:cls_mono_crackbin_effb0")
    ap.add_argument("--color", default="latest:cls_color_effb0")
    ap.add_argument("--max-images", type=int, default=1500)
    ap.add_argument("--out", default="docs/roboflow_sanity.md")
    args = ap.parse_args()

    rf_root = resolve_path("data/roboflow")
    models = {}
    for tag, spec, key in (("MONO 2클래스", args.mono, "2"), ("COLOR 5클래스", args.color, "2/0")):
        exp = resolve_experiment(spec)
        m = TimmClassifier(exp / "model.pt")
        ev = json.loads((exp / "eval.json").read_text(encoding="utf-8"))
        thr = ev["operating_points"]["recall@0.95"]["threshold"]
        models[tag] = (m, key, thr, exp.name)

    lines = ["# 로보플로우 보조 데이터 sanity check (자동 생성: scripts/roboflow_sanity.py)", "",
             "우리 모델을 촬영 환경이 전혀 다른 공개 사진에 그대로 적용한 결과. **KPI 가 아니며** 도메인 이동의 크기를 가늠하는 용도. 임계값은 AI Hub test 에서 파각 Recall 95% 인 값을 그대로 사용.", "",
             "| 데이터셋 | 모델 | 크랙 n | 정상 n | AUROC(크랙 vs 정상) | 크랙 Recall@AI Hub 임계값 | 정상 FPR@같은 임계값 | 이 데이터에서 Recall 95% 맞출 때 FPR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    results = {}
    for ds in sorted(p for p in rf_root.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "data.yaml").exists()):
        df = collect_crops(ds, args.max_images)
        if df.empty or not df["is_crack"].any():
            lines.append(f"| {ds.name} | - | 0 | {int((~df['is_crack']).sum()) if not df.empty else 0} | 크랙 라벨 없음 → 건너뜀 | | | |")
            continue
        for tag, (m, key, thr, exp_name) in models.items():
            s = score_dataset(df, m, key)
            pos = df["is_crack"].to_numpy()
            r = {"n_crack": int(pos.sum()), "n_normal": int((~pos).sum()), "recall_at_thr": float((s[pos] >= thr).mean())}
            if (~pos).any():
                r["auroc"] = auroc(s, pos)
                r["fpr_at_thr"] = float((s[~pos] >= thr).mean())
                r["fpr_at_recall95"] = operating_point(s, pos, 0.95)["fpr"]
            results[f"{ds.name}|{tag}"] = r
            lines.append(f"| {ds.name} | {tag} | {r['n_crack']:,} | {r['n_normal']:,} | {r.get('auroc', float('nan')):.3f} | {r['recall_at_thr']:.3f} | {r.get('fpr_at_thr', float('nan')):.3f} | {r.get('fpr_at_recall95', float('nan')):.3f} |")
    lines += ["", "해석: AUROC 가 0.5 근처면 도메인 이동이 너무 커서 전혀 전이되지 않는 것, 0.8 이상이면 특징은 전이되지만 임계값 재조정이 필요한 것. 이 사진들은 스톡 사진·그레이스케일 변환·배경 제거가 섞여 있어 자사 라인보다 훨씬 먼 도메인이다."]
    out = PROJECT_ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    (PROJECT_ROOT / "outputs" / "roboflow_sanity.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
