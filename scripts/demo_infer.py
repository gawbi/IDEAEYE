"""데모 추론: 폴더의 판 이미지 → 검출 → 헤드 A(크랙 vs 정상) + 헤드 B(33클래스 소분류·검란) → 판정 이미지 + CSV.

python scripts/demo_infer.py --images <폴더> --out outputs/demo [--n 20] [--crack-recall 0.95]
  --det     검출기 가중치 (기본 C:/aihub_egg/runs/det_yolo11n_MONO/weights/best.pt)
  --head-a  크랙 2클래스 실험 (기본 latest:cls_mono_crackbin_effb0)
  --head-b  33클래스 실험 (기본 latest:cls_mono_effb0)
판정 규칙(임시): 크랙 = 헤드 A 점수 ≥ AI Hub test 에서 파각 Recall 목표를 만족하는 임계값, 검란 배출 = 헤드 B argmax 가 혈반 계열(3/x).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402
from src.infer.adapters import TimmClassifier, YoloDetector  # noqa: E402
from src.train.classifier import imread  # noqa: E402

HYP = {}
try:
    HYP = load_config("class_codes").get("MONO", {})
except Exception:
    pass


def name_of(code: str) -> str:
    st, det = (code.split("/") + ["0"])[:2]
    s = HYP.get("state", {}).get(st, {})
    d = HYP.get("state_detail", {}).get(code, {}) if isinstance(HYP.get("state_detail"), dict) else {}
    n = s.get("name", st) if isinstance(s, dict) else st
    if isinstance(d, dict) and d.get("name"):
        n = str(d["name"])
    return n


def probs_for(model: TimmClassifier, image: np.ndarray, boxes) -> np.ndarray:
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


def draw(image: np.ndarray, rows: list[dict]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    fp = Path(r"C:\Windows\Fonts\malgun.ttf")
    font = ImageFont.truetype(str(fp), 20) if fp.exists() else None
    for r in rows:
        color = (255, 60, 60) if r["reject"] else (60, 220, 60)
        d.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=color, width=3)
        txt = f"#{r['egg']} {'배출' if r['reject'] else '통과'} | 크랙 {r['p_crack']:.2f} | {r['label_b_name']} {r['p_b']:.2f}"
        d.rectangle([r["x1"], max(0, r["y1"] - 26), r["x1"] + 12 * len(txt), r["y1"]], fill=(0, 0, 0))
        d.text((r["x1"] + 2, max(0, r["y1"] - 25)), txt, fill=color, font=font)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--det", default="C:/aihub_egg/runs/det_yolo11n_MONO/weights/best.pt")
    ap.add_argument("--head-a", default="latest:cls_mono_crackbin_effb0")
    ap.add_argument("--head-b", default="latest:cls_mono_effb0")
    ap.add_argument("--crack-recall", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.10)
    args = ap.parse_args()

    infer_cfg = load_config("infer")
    det = YoloDetector(args.det, imgsz=640, conf=infer_cfg["detector"]["conf_threshold"])
    exp_a, exp_b = resolve_experiment(args.head_a), resolve_experiment(args.head_b)
    A, B = TimmClassifier(exp_a / "model.pt"), TimmClassifier(exp_b / "model.pt")
    ev_a = json.loads((exp_a / "eval.json").read_text(encoding="utf-8"))
    thr = ev_a["operating_points"][f"recall@{args.crack_recall:.2f}"]["threshold"]
    ci_a = A.classes.index("2")
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.n]
    all_rows = []
    for p in images:
        img = imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = [b.expanded(args.margin).clipped(W, H) for b in det.detect(img)]
        if not boxes:
            all_rows.append({"image": p.name, "egg": 0, "note": "no egg detected"})
            continue
        pa, pb = probs_for(A, img, boxes), probs_for(B, img, boxes)
        rows = []
        for i, b in enumerate(boxes):
            p_crack = float(pa[i, ci_a])
            jb = int(pb[i].argmax())
            code_b = B.classes[jb]
            candling = code_b.startswith("3/")
            reject = p_crack >= thr or candling
            x1, y1, x2, y2 = b.to_int()
            rows.append({"image": p.name, "egg": i + 1, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "p_crack": p_crack, "crack": p_crack >= thr, "label_b": code_b, "label_b_name": name_of(code_b), "p_b": float(pb[i, jb]), "candling_reject": candling, "reject": reject})
        all_rows += rows
        ok, buf = cv2.imencode(".jpg", draw(img, rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        buf.tofile(str(out / f"demo_{p.stem}.jpg"))
    with open(out / "demo_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r}))
        w.writeheader()
        w.writerows(all_rows)
    n_eggs = sum(1 for r in all_rows if r.get("egg"))
    n_rej = sum(1 for r in all_rows if r.get("reject"))
    print(f"images {len(images)}, eggs {n_eggs}, rejected {n_rej} (crack thr {thr:.3f} @ recall {args.crack_recall}) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
