"""엣지 추론 파이프라인 뼈대: 캡처 → 계란 검출 → 크롭 분류 → 판정 → PLC 배출 신호.

Detector / Classifier 는 Protocol 로 두어 실제 TensorRT 구현과 테스트용 stub 을 바꿔 끼운다.
설계 문서: docs/inference_pipeline.md
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..datasets.coords import Box
from ..plc.base import AlarmCode, PLCError, PLCInterface, RejectCommand


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]: ...


class Classifier(Protocol):
    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """(label, score). score 는 label 의 확률/신뢰도."""
        ...


@dataclass
class EggResult:
    index: int
    box: tuple[int, int, int, int]
    label: str
    score: float
    reject: bool


@dataclass
class TrayResult:
    tray_seq: int
    eggs: list[EggResult] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    safety_action: str | None = None

    @property
    def rejected(self) -> list[int]:
        return [e.index for e in self.eggs if e.reject]


class DecisionRule:
    """소분류 라벨과 점수로 배출 여부 결정. 임계값은 configs/infer.yaml decision 절."""

    def __init__(self, reject_classes: list[str], crack_threshold: float, thresholds: dict[str, float] | None = None):
        self.reject_classes = set(reject_classes)
        self.default_threshold = crack_threshold
        self.thresholds = thresholds or {}

    def decide(self, label: str, score: float) -> bool:
        if label not in self.reject_classes:
            return False
        return score >= self.thresholds.get(label, self.default_threshold)


class JudgmentLogger:
    """판정 로그(JSONL + 선택적 이미지 저장). 재학습 데이터로 재사용한다."""

    def __init__(self, root: str | Path, save_images: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.path = self.root / "judgments.jsonl"

    def log(self, result: TrayResult, image: np.ndarray | None = None) -> None:
        rec = {"ts": time.time(), **asdict(result)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.save_images and image is not None:
            import cv2

            ok, buf = cv2.imencode(".jpg", image)
            if ok:
                buf.tofile(str(self.root / f"tray_{result.tray_seq:08d}.jpg"))


def crop_with_margin(image: np.ndarray, box: Box, margin: float) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.expanded(margin).clipped(w, h).to_int()
    return image[y1:y2, x1:x2]


class InferencePipeline:
    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        decision: DecisionRule,
        plc: PLCInterface,
        cfg: dict[str, Any],
        logger: JudgmentLogger | None = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.decision = decision
        self.plc = plc
        self.cfg = cfg
        self.logger = logger
        self.tray_seq = 0

    # ---- 안전 동작 -----------------------------------------------------
    def _safety(self, tray_seq: int, policy_key: str, code: AlarmCode, message: str) -> str:
        action = self.cfg["safety"].get(policy_key, "stop_line")
        try:
            self.plc.alarm(code, message)
            if action == "reject_all":
                self.plc.reject_all(tray_seq, message)
            else:
                self.plc.stop_line(message)
        except PLCError as e:  # PLC 마저 실패: 상위(운영 감시)에서 처리하도록 기록만
            action = f"{action} (plc failed: {e})"
        return action

    # ---- 메인 --------------------------------------------------------
    def process(self, image: np.ndarray | None) -> TrayResult:
        self.tray_seq += 1
        res = TrayResult(tray_seq=self.tray_seq)
        t0 = time.perf_counter()

        if image is None:
            res.error = "camera frame is None"
            res.safety_action = self._safety(res.tray_seq, "on_camera_lost", AlarmCode.CAMERA_LOST, res.error)
            return self._finish(res, t0, None)

        try:
            t = time.perf_counter()
            boxes = self.detector.detect(image)
            res.timings_ms["detect"] = (time.perf_counter() - t) * 1e3

            t = time.perf_counter()
            margin = self.cfg["classifier"]["crop_margin"]
            use_batch = self.cfg["classifier"].get("batch", True)
            if use_batch and hasattr(self.classifier, "classify_boxes"):
                h, w = image.shape[:2]
                results = self.classifier.classify_boxes(image, [b.expanded(margin).clipped(w, h) for b in boxes])  # 판 이미지 1회 전송, GPU 크롭
            elif use_batch and hasattr(self.classifier, "classify_batch"):
                results = self.classifier.classify_batch([crop_with_margin(image, box, margin) for box in boxes])
            else:
                results = [self.classifier.classify(crop_with_margin(image, box, margin)) for box in boxes]
            for i, (box, (label, score)) in enumerate(zip(boxes, results)):
                res.eggs.append(EggResult(i, box.to_int(), label, score, self.decision.decide(label, score)))
            res.timings_ms["classify"] = (time.perf_counter() - t) * 1e3
        except Exception as e:  # 모델 오류
            res.error = f"inference error: {e}"
            res.safety_action = self._safety(res.tray_seq, "on_model_load_fail", AlarmCode.INFERENCE_ERROR, res.error)
            return self._finish(res, t0, image)

        t = time.perf_counter()
        try:
            for e in res.eggs:
                if e.reject:
                    self.plc.send_reject(RejectCommand(res.tray_seq, e.index, e.label, e.score))
        except PLCError as e:
            res.error = f"plc error: {e}"
            res.safety_action = self._safety(res.tray_seq, "on_plc_lost", AlarmCode.PLC_LOST, res.error)
        res.timings_ms["plc"] = (time.perf_counter() - t) * 1e3
        return self._finish(res, t0, image)

    def _finish(self, res: TrayResult, t0: float, image: np.ndarray | None) -> TrayResult:
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1e3
        limit = self.cfg.get("logging", {}).get("max_latency_ms", 0) or 0
        if limit and res.timings_ms["total"] > limit:
            self.plc.alarm(AlarmCode.LATENCY_EXCEEDED, f"{res.timings_ms['total']:.1f}ms > {limit}ms")
        if self.logger:
            self.logger.log(res, image)
        return res
