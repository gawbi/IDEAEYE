"""노트북에서 실행: 분류기(고정 배치)·검출기 ONNX 내보내기 + 메타 JSON.

python deploy/jetson/export_onnx.py --cls latest:cls_mono_crackbin_effb0 [--det <best.pt>] [--batch 6] [--out deploy/jetson/models]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.experiments import resolve_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True)
    ap.add_argument("--det", default=None)
    ap.add_argument("--batch", type=int, default=6, help="고정 배치 (카메라 2대 × 자리 3개)")
    ap.add_argument("--out", default="deploy/jetson/models")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import timm
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = resolve_experiment(args.cls) if args.cls.startswith("latest:") else Path(args.cls).parent
    ck = torch.load(exp / "model.pt", map_location="cpu", weights_only=False)
    cfg, classes = ck["cfg"], ck["classes"]
    model = timm.create_model(cfg["model"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    size = cfg["img_size"]
    dummy = torch.randn(args.batch, 3, size, size)
    cls_path = out / f"classifier_b{args.batch}_{size}.onnx"
    torch.onnx.export(model, dummy, str(cls_path), input_names=["input"], output_names=["logits"], opset_version=args.opset, dynamic_axes=None)
    meta = {"source_experiment": exp.name, "model": cfg["model"], "img_size": size, "classes": classes, "batch": args.batch, "preprocess": cfg.get("preprocess", "none"),
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "input_layout": "NCHW RGB float32 0-1 then normalize",
            "crack_class": "2" if "2" in classes else None}
    (out / "classifier_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("classifier →", cls_path)

    if args.det:
        from ultralytics import YOLO

        p = YOLO(args.det).export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=args.opset)
        dst = out / "detector_640.onnx"
        Path(p).replace(dst)
        print("detector →", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
