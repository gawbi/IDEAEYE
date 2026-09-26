import json

import pytest

from src.eval.metrics import grouped_metrics, operating_point, per_class_metrics
from src.eval.tracker import NO_GIT, ExperimentTracker


def test_per_class_metrics():
    y_true = ["정상", "정상", "미세크랙", "미세크랙", "먹통"]
    y_pred = ["정상", "미세크랙", "미세크랙", "정상", "먹통"]
    m = per_class_metrics(y_true, y_pred)
    assert m["미세크랙"]["support"] == 2
    assert m["미세크랙"]["recall"] == pytest.approx(0.5)
    assert m["미세크랙"]["precision"] == pytest.approx(0.5)
    assert m["먹통"]["f1"] == pytest.approx(1.0)


def test_grouped_metrics_by_washed():
    rows = [
        {"washed": "Y", "y_true": "크랙", "y_pred": "크랙"},
        {"washed": "Y", "y_true": "크랙", "y_pred": "정상"},
        {"washed": "N", "y_true": "크랙", "y_pred": "크랙"},
    ]
    g = grouped_metrics(rows, "washed")
    assert g["Y"]["크랙"]["recall"] == pytest.approx(0.5)
    assert g["N"]["크랙"]["recall"] == pytest.approx(1.0)


def test_operating_point_recall_target():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    pos = [True, True, False, True, True, False, False, False]
    op = operating_point(scores, pos, target_recall=1.0)
    assert op["recall"] == 1.0 and op["threshold"] == pytest.approx(0.5)
    assert op["fpr"] == pytest.approx(1 / 4)  # 0.7 하나가 음성 오검출
    op2 = operating_point(scores, pos, target_recall=0.5)
    assert op2["threshold"] == pytest.approx(0.8) and op2["fpr"] == 0.0


def test_tracker_records_everything(tmp_path):
    cfg = {"model": "yolo11n", "lr": 0.001}
    tr = ExperimentTracker("unit", cfg, seed=123, root=tmp_path / "experiments", notes="test")
    assert (tr.dir / "config.yaml").exists()
    meta = json.loads((tr.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 123 and meta["git_hash"]  # git 없으면 'no-git'
    assert "python" in meta["env"]

    overall = per_class_metrics(["a", "b"], ["a", "b"])
    by = {"state_detail": overall, "washed": {"Y": overall, "N": overall}}
    p = tr.log_metrics(overall, by=by, headline={"crack_recall": 0.95})
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["by"]["washed"]["Y"]["a"]["recall"] == 1.0
    assert (tr.dir / "metrics.md").exists()
    summary = (tmp_path / "experiments" / "SUMMARY.md").read_text(encoding="utf-8")
    assert tr.dir.name in summary and "crack_recall=0.9500" in summary
    if meta["git_hash"] == NO_GIT:
        assert "| NO |" in summary  # 재현 불가 표시
