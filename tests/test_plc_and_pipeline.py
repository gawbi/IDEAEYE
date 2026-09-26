import json

import numpy as np
import pytest

from src.common.config import load_config
from src.datasets.coords import Box
from src.infer.pipeline import DecisionRule, InferencePipeline, JudgmentLogger, crop_with_margin
from src.plc.base import AlarmCode, PLCError, RejectCommand
from src.plc.mock import MockPLC


# ---------------- PLC mock


def test_mock_plc_records_calls():
    plc = MockPLC()
    plc.connect()
    assert plc.is_connected
    plc.send_reject(RejectCommand(1, 3, "미세크랙", 0.9))
    plc.heartbeat()
    plc.alarm(AlarmCode.LATENCY_EXCEEDED, "slow")
    assert plc.names() == ["connect", "send_reject", "heartbeat", "alarm"]
    assert plc.rejects()[0].egg_index == 3


def test_mock_plc_failures():
    with pytest.raises(PLCError):
        MockPLC(fail_connect=True).connect()
    plc = MockPLC()
    with pytest.raises(PLCError):
        plc.heartbeat()  # 연결 전
    plc = MockPLC(fail_after_calls=1)
    plc.connect()
    plc.heartbeat()
    with pytest.raises(PLCError):
        plc.heartbeat()
    assert not plc.is_connected


# ---------------- 파이프라인 stub


class StubDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image):
        return self.boxes


class StubClassifier:
    def __init__(self, results):
        self.results = list(results)
        self.i = 0

    def classify(self, crop):
        assert crop.size > 0
        r = self.results[self.i % len(self.results)]
        self.i += 1
        return r


class BrokenClassifier:
    def classify(self, crop):
        raise RuntimeError("engine not loaded")


@pytest.fixture
def infer_cfg():
    cfg = load_config("infer")
    cfg["plc"]["backend"] = "mock"
    return cfg


def _pipeline(infer_cfg, detector, classifier, plc=None, logger=None):
    d = infer_cfg["decision"]
    rule = DecisionRule(d["reject_classes"], d["crack_threshold"])
    plc = plc or MockPLC()
    plc.connect()
    return InferencePipeline(detector, classifier, rule, plc, infer_cfg, logger), plc


def test_pipeline_rejects_and_signals_plc(infer_cfg, tmp_path):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = [Box(0, 0, 50, 50), Box(60, 0, 110, 50), Box(120, 0, 170, 50)]
    clf = StubClassifier([("정상", 0.99), ("크랙", 0.8), ("크랙", 0.2)])  # 마지막은 임계값(0.5) 미만
    logger = JudgmentLogger(tmp_path / "logs")
    pipe, plc = _pipeline(infer_cfg, StubDetector(boxes), clf, logger=logger)

    res = pipe.process(img)
    assert res.error is None
    assert res.rejected == [1]
    assert [c.egg_index for c in plc.rejects()] == [1]
    assert set(res.timings_ms) == {"detect", "classify", "plc", "total"}
    line = json.loads((tmp_path / "logs" / "judgments.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["tray_seq"] == 1 and len(line["eggs"]) == 3


def test_pipeline_camera_lost_stops_line(infer_cfg):
    pipe, plc = _pipeline(infer_cfg, StubDetector([]), StubClassifier([("정상", 1.0)]))
    res = pipe.process(None)
    assert res.error and res.safety_action == "stop_line"
    assert "stop_line" in plc.names()
    assert any(p["code"] == AlarmCode.CAMERA_LOST.value for n, p in plc.calls if n == "alarm")


def test_pipeline_model_failure_rejects_all(infer_cfg):
    infer_cfg["safety"]["on_model_load_fail"] = "reject_all"
    pipe, plc = _pipeline(infer_cfg, StubDetector([Box(0, 0, 10, 10)]), BrokenClassifier())
    res = pipe.process(np.zeros((20, 20, 3), dtype=np.uint8))
    assert res.safety_action == "reject_all"
    assert "reject_all" in plc.names()


def test_pipeline_plc_loss_is_reported(infer_cfg):
    plc = MockPLC(fail_after_calls=0)  # 첫 명령부터 실패
    pipe, plc = _pipeline(infer_cfg, StubDetector([Box(0, 0, 10, 10)]), StubClassifier([("크랙", 0.9)]), plc=plc)
    res = pipe.process(np.zeros((20, 20, 3), dtype=np.uint8))
    assert res.error.startswith("plc error")
    assert "plc failed" in res.safety_action  # PLC 자체가 죽어 안전 동작도 실패 → 기록됨


def test_crop_with_margin_clips():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = crop_with_margin(img, Box(40, 40, 50, 50), margin=0.5)
    assert crop.shape[:2] == (15, 15)
