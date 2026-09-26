"""Phase 7: ONNX 변환 + 엔드투엔드 지연 벤치마크 (검출 → 크롭 분류 → 판정 → mock PLC).

python scripts/export_and_benchmark.py --det <best.pt> --cls <experiments/…/model.pt> [--n 100] [--light MONO]
출력: experiments/<stamp>_benchmark_<light>/ (onnx 파일, 단계별 p50/p95, 계란 1개당 지연)
TensorRT 는 SDK 미설치로 생략 (ONNX Runtime CUDA EP 로 측정).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config  # noqa: E402
from src.common.experiments import resolve_experiment  # noqa: E402
from src.eval.tracker import ExperimentTracker  # noqa: E402
from src.infer.adapters import OnnxClassifier, TimmClassifier, YoloDetector  # noqa: E402
from src.infer.pipeline import DecisionRule, InferencePipeline, JudgmentLogger  # noqa: E402
from src.plc.mock import MockPLC  # noqa: E402
from src.train.classifier import imread  # noqa: E402


def pct(v: list[float]) -> dict[str, float]:
    a = np.asarray(v)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "mean": float(a.mean())} if len(a) else {}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--light", default="MONO")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--det-root", default="C:/aihub_egg/det")
    ap.add_argument("--cls-backend", choices=["onnx", "torch"], default="onnx", help="분류기 실행 백엔드 (onnx: ORT CUDA 고정 배치, torch: PyTorch FP16 배치)")
    ap.add_argument("--fixed-batch", type=int, default=16)
    args = ap.parse_args()
    if args.cls.startswith("latest:"):
        args.cls = str(resolve_experiment(args.cls) / "model.pt")

    import torch

    infer_cfg = load_config("infer")
    tracker = ExperimentTracker(f"benchmark_{args.light}_{args.cls_backend}", {**infer_cfg, "det": args.det, "cls": args.cls, "n": args.n, "cls_backend": args.cls_backend, "fixed_batch": args.fixed_batch}, seed=0, notes="Phase 7 latency benchmark")

    # 1. ONNX 변환
    from ultralytics import YOLO

    det_onnx = YOLO(args.det).export(format="onnx", imgsz=infer_cfg["detector"].get("imgsz", 640), half=False, dynamic=False, simplify=True)
    tc = TimmClassifier(args.cls, half=False)
    cls_onnx = tracker.dir / "classifier.onnx"
    dummy = torch.randn(1, 3, tc.cfg["img_size"], tc.cfg["img_size"], device=tc.device)
    torch.onnx.export(tc.model.float(), dummy, str(cls_onnx), input_names=["input"], output_names=["logits"], opset_version=17, dynamic_axes={"input": {0: "batch"}})
    (tracker.dir / "classifier_classes.json").write_text(json.dumps({"classes": tc.classes, "img_size": tc.cfg["img_size"], "preprocess": tc.cfg.get("preprocess", "none")}, ensure_ascii=False), encoding="utf-8")

    # 2. 파이프라인 구성 (검출: Ultralytics PyTorch FP16, 분류: ONNX Runtime CUDA)
    detector = YoloDetector(args.det, imgsz=infer_cfg["detector"].get("imgsz", 640), conf=infer_cfg["detector"]["conf_threshold"])
    if args.cls_backend == "torch":
        classifier = TimmClassifier(args.cls, half=True)
        classifier.providers = ["torch-fp16"]
    else:
        classifier = OnnxClassifier(cls_onnx, tc.classes, tc.cfg["img_size"], tc.cfg.get("preprocess", "none"), fixed_batch=args.fixed_batch)
    reject_classes = [c for c in tc.classes if c.split("/")[0] in {"2", "3", "5", "6"}]  # 크랙·혈반·이물질·기형 코드 (가설)
    rule = DecisionRule(reject_classes, infer_cfg["decision"]["crack_threshold"])
    plc = MockPLC()
    plc.connect()
    pipe = InferencePipeline(detector, classifier, rule, plc, infer_cfg, JudgmentLogger(tracker.dir / "judgments", save_images=False))

    # 3. test 판 이미지로 측정
    images = sorted((Path(args.det_root) / args.light / "images" / "test").glob("*.jpg"))[: args.n]
    for p in images[:3]:  # warm-up
        pipe.process(imread(str(p)))
    stage: dict[str, list[float]] = {"detect": [], "classify": [], "plc": [], "total": []}
    per_egg, n_eggs = [], []
    for p in images:
        r = pipe.process(imread(str(p)))
        for k in stage:
            stage[k].append(r.timings_ms.get(k, 0.0))
        n_eggs.append(len(r.eggs))
        if r.eggs:
            per_egg.append(r.timings_ms["total"] / len(r.eggs))
    result = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "classifier_providers": classifier.providers,
        "n_images": len(images), "eggs_per_image_mean": float(np.mean(n_eggs)) if n_eggs else 0,
        "stage_ms": {k: pct(v) for k, v in stage.items()},
        "per_egg_ms": pct(per_egg),
        "rejects_total": len(plc.rejects()),
        "onnx": {"detector": str(det_onnx), "classifier": str(cls_onnx)},
        "note": "TensorRT 미설치 → ORT CUDA EP. 검출은 Ultralytics PyTorch FP16. 실제 장비에서 재측정 필요",
    }
    (tracker.dir / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.log_metrics({}, headline={"per_egg_ms_p50": result["per_egg_ms"].get("p50", 0), "total_ms_p95": result["stage_ms"]["total"].get("p95", 0)})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"→ {tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
