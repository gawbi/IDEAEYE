import numpy as np
import pytest

from src.common.config import load_config
from src.datasets.coords import Box
from src.infer.pipeline import DecisionRule, InferencePipeline
from src.plc.mock import MockPLC


class BatchStubClassifier:
    """classify_batch 를 제공하는 stub — 파이프라인이 배치 경로를 타는지 확인."""

    def __init__(self):
        self.batch_calls, self.single_calls = 0, 0

    def classify(self, crop):
        self.single_calls += 1
        return "1", 0.9

    def classify_batch(self, crops):
        self.batch_calls += 1
        return [("2", 0.9) if i % 2 else ("1", 0.9) for i, _ in enumerate(crops)]


class StubDetector:
    def detect(self, image):
        return [Box(0, 0, 20, 20), Box(30, 0, 50, 20), Box(60, 0, 80, 20)]


def test_pipeline_uses_batch_classify():
    cfg = load_config("infer")
    clf = BatchStubClassifier()
    plc = MockPLC()
    plc.connect()
    pipe = InferencePipeline(StubDetector(), clf, DecisionRule(["2"], 0.5), plc, cfg)
    res = pipe.process(np.zeros((40, 100, 3), dtype=np.uint8))
    assert clf.batch_calls == 1 and clf.single_calls == 0
    assert res.rejected == [1]
    cfg["classifier"]["batch"] = False
    pipe2 = InferencePipeline(StubDetector(), clf, DecisionRule(["2"], 0.5), plc, cfg)
    pipe2.process(np.zeros((40, 100, 3), dtype=np.uint8))
    assert clf.single_calls == 3


class BoxesStubClassifier(BatchStubClassifier):
    def __init__(self):
        super().__init__()
        self.boxes_calls = 0
        self.last_boxes = None

    def classify_boxes(self, image, boxes):
        self.boxes_calls += 1
        self.last_boxes = boxes
        return [("2", 0.9) if i % 2 else ("1", 0.9) for i, _ in enumerate(boxes)]


def test_pipeline_prefers_classify_boxes():
    cfg = load_config("infer")
    clf = BoxesStubClassifier()
    plc = MockPLC()
    plc.connect()
    pipe = InferencePipeline(StubDetector(), clf, DecisionRule(["2"], 0.5), plc, cfg)
    res = pipe.process(np.zeros((40, 100, 3), dtype=np.uint8))
    assert clf.boxes_calls == 1 and clf.batch_calls == 0
    assert res.rejected == [1]
    b = clf.last_boxes[0]  # 여백 10% 적용 후 이미지 범위로 클리핑된 박스
    assert b.x1 == 0 and b.y1 == 0 and b.x2 == 22 and b.y2 == 22


def test_pad_batches():
    from src.infer.adapters import pad_batches

    assert pad_batches(0, 16) == []
    assert pad_batches(12, 16) == [(0, 12)]
    assert pad_batches(17, 16) == [(0, 16), (16, 17)]
    assert pad_batches(5, 0) == [(0, 5)]


def test_label_map_3cls(tmp_path):
    torch = pytest.importorskip("torch")
    import pandas as pd

    from src.train.classifier import cap_per_class

    df = pd.DataFrame({"state": ["1", "2", "3", "4", "5", "6"], "cls": ["1/1", "2/1", "3/1", "4/1", "5/1", "6/1"]})
    lm = {"1": "1", "2": "2", "default": "9"}
    df["label3"] = df["state"].map(lambda s: lm.get(s, lm["default"]))
    assert sorted(df["label3"].unique()) == ["1", "2", "9"]
    assert len(cap_per_class(df, 1, 0, "label3")) == 3
