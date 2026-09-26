"""실제 모델을 InferencePipeline 의 Detector / Classifier 프로토콜에 맞추는 어댑터 (Phase 7).

- YoloDetector: Ultralytics .pt 또는 .onnx (계란 위치, 클래스 무시)
- TimmClassifier: PyTorch 체크포인트 (experiments/…/model.pt)
- OnnxClassifier: onnxruntime (CUDAExecutionProvider 우선)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..datasets.coords import Box
from ..train.classifier import MEAN, STD, letterbox, preprocess_image


class YoloDetector:
    def __init__(self, weights: str | Path, imgsz: int = 640, conf: float = 0.25, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def detect(self, image: np.ndarray) -> list[Box]:
        res = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        return [Box(*map(float, b)) for b in boxes]


def _to_input(crop: np.ndarray, img_size: int, preprocess: str) -> np.ndarray:
    if preprocess != "none":
        crop = preprocess_image(crop, preprocess)
    x = letterbox(crop, img_size)[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


class TimmClassifier:
    def __init__(self, checkpoint: str | Path, device: str | None = None, half: bool = True):
        import timm
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes: list[str] = ck["classes"]
        self.cfg = ck["cfg"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(self.cfg["model"], pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.torch = torch
        self._mean = self._std = None

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        return self.classify_batch([crop])[0]

    def _letterbox_gpu(self, crop: np.ndarray, size: int):
        """uint8 크롭을 GPU 로 올린 뒤 리사이즈·패딩 (CPU 는 전송만). 반환 (3,size,size) float, RGB, 0..1."""
        th = self.torch
        t = th.from_numpy(np.ascontiguousarray(crop)).to(self.device, non_blocking=True)  # (H,W,3) BGR
        t = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(th.float16 if self.half else th.float32) / 255.0
        h, w = t.shape[-2:]
        s = size / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        if (nh, nw) != (h, w):
            t = th.nn.functional.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False, antialias=s < 1)
        out = th.zeros((1, 3, size, size), device=self.device, dtype=t.dtype)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        out[:, :, y0 : y0 + nh, x0 : x0 + nw] = t
        return out

    def _letterbox_tensor(self, t, size: int):
        """(1,3,h,w) float 텐서 → (1,3,size,size) letterbox (GPU)."""
        th = self.torch
        h, w = t.shape[-2:]
        s = size / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        if (nh, nw) != (h, w):
            t = th.nn.functional.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False, antialias=s < 1)
        out = th.zeros((1, 3, size, size), device=self.device, dtype=t.dtype)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        out[:, :, y0 : y0 + nh, x0 : x0 + nw] = t
        return out

    def classify_boxes(self, image: np.ndarray, boxes: list[Box]) -> list[tuple[str, float]]:
        """판 이미지를 GPU 로 한 번만 올리고 bbox 별 크롭·리사이즈·정규화를 모두 GPU 에서 수행 (계란별 전송 제거)."""
        if not boxes:
            return []
        if self.cfg.get("preprocess", "none") != "none":  # CLAHE 등은 CPU 크롭 경로로
            return self.classify_batch([image[int(b.y1) : int(b.y2), int(b.x1) : int(b.x2)] for b in boxes])
        th = self.torch
        size = self.cfg["img_size"]
        dt = th.float16 if self.half else th.float32
        img = th.from_numpy(np.ascontiguousarray(image)).to(self.device, non_blocking=True)  # (H,W,3) BGR uint8, 1회 전송
        img = img[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(dt) / 255.0  # (1,3,H,W) RGB
        H, W = img.shape[-2:]
        tiles = []
        for b in boxes:
            x1, y1, x2, y2 = b.clipped(W, H).to_int()
            if x2 <= x1 or y2 <= y1:
                tiles.append(th.zeros((1, 3, size, size), device=self.device, dtype=dt))
                continue
            tiles.append(self._letterbox_tensor(img[:, :, y1:y2, x1:x2], size))
        t = th.cat(tiles)
        return self._forward(t)

    def _forward(self, t) -> list[tuple[str, float]]:
        if self._mean is None:
            self._mean = self.torch.tensor(MEAN, device=self.device, dtype=t.dtype).view(1, 3, 1, 1)
            self._std = self.torch.tensor(STD, device=self.device, dtype=t.dtype).view(1, 3, 1, 1)
        t = (t - self._mean) / self._std
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(t).float(), dim=1).cpu().numpy()
        idx = p.argmax(1)
        return [(self.classes[int(i)], float(p[r, i])) for r, i in enumerate(idx)]

    def classify_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """리사이즈·정규화 모두 GPU 에서 수행 (판당 분류 지연의 대부분이 CPU 전처리였음)."""
        if not crops:
            return []
        size, pre = self.cfg["img_size"], self.cfg.get("preprocess", "none")
        if pre != "none":  # CLAHE 등은 CPU 에서
            crops = [preprocess_image(c, pre) for c in crops]
        t = self.torch.cat([self._letterbox_gpu(c, size) for c in crops])
        return self._forward(t)


def pad_batches(n: int, fixed: int) -> list[tuple[int, int]]:
    """n 개 크롭을 fixed 크기 청크로 나눈 (start, end) 목록. 마지막 청크는 fixed 로 패딩해 입력 형태를 고정한다."""
    if fixed <= 0:
        return [(0, n)] if n else []
    return [(s, min(n, s + fixed)) for s in range(0, n, fixed)]


class OnnxClassifier:
    """ONNX Runtime 분류기. 입력 형태가 바뀔 때마다 CUDA EP 가 cuDNN 알고리즘을 재탐색해 느려지므로
    배치 크기를 fixed_batch 로 고정(패딩)하고 알고리즘 탐색을 HEURISTIC 으로 둔다."""

    def __init__(self, onnx_path: str | Path, classes: list[str], img_size: int, preprocess: str = "none", providers: list[str] | None = None, fixed_batch: int = 16):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        if providers is None:
            providers = []
            if "CUDAExecutionProvider" in avail:
                providers.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
            providers.append("CPUExecutionProvider")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.classes, self.img_size, self.preprocess, self.fixed_batch = classes, img_size, preprocess, fixed_batch
        self.providers = self.sess.get_providers()

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        return self.classify_batch([crop])[0]

    def _run(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if self.fixed_batch > 0 and n < self.fixed_batch:  # 형태 고정을 위한 패딩
            x = np.concatenate([x, np.zeros((self.fixed_batch - n, *x.shape[1:]), dtype=x.dtype)])
        return self.sess.run(None, {self.input_name: x})[0][:n]

    def classify_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """동적 배치 축으로 export 된 ONNX 전제. 판 하나의 계란을 고정 크기 청크로 추론."""
        if not crops:
            return []
        x = np.concatenate([_to_input(c, self.img_size, self.preprocess) for c in crops])
        logits = np.concatenate([self._run(x[s:e]) for s, e in pad_batches(len(x), self.fixed_batch)])
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        idx = p.argmax(1)
        return [(self.classes[int(i)], float(p[r, i])) for r, i in enumerate(idx)]
