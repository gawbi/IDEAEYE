import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.train.classifier import FocalLoss, crack_operating_points, evaluate, letterbox, make_loss, preprocess_image  # noqa: E402


def test_letterbox_and_preprocess():
    img = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    out = letterbox(img, 128)
    assert out.shape == (128, 128, 3)
    for mode in ("none", "clahe", "unsharp", "clahe+unsharp"):
        p = preprocess_image(img, mode)
        assert p.shape == img.shape and p.dtype == np.uint8


def test_losses_run():
    logits = torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))
    for kind in ("ce", "weighted_ce", "focal", "weighted_focal"):
        loss = make_loss(kind, [100, 10, 5, 1], torch.device("cpu"))(logits, y)
        assert torch.isfinite(loss)
    assert FocalLoss(gamma=0.0)(logits, y).item() == pytest.approx(torch.nn.functional.cross_entropy(logits, y).item(), abs=1e-5)


def test_evaluate_and_operating_points():
    classes = ["1/1", "1/2", "2/6", "2/8"]
    y = np.array([0, 0, 1, 2, 2, 3, 3, 3])
    probs = np.zeros((8, 4))
    probs[np.arange(8), y] = 0.9
    probs[0] = [0.1, 0.1, 0.7, 0.1]  # 정상 하나를 크랙으로 오판
    probs[5] = [0.8, 0.1, 0.05, 0.05]  # 미세크랙 하나 놓침
    probs /= probs.sum(1, keepdims=True)
    ev = evaluate(probs, y, classes)
    assert ev["overall"]["2/8"]["recall"] == pytest.approx(2 / 3)
    assert ev["overall"]["1/1"]["recall"] == pytest.approx(0.5)
    ops = crack_operating_points(probs, y, classes, targets=(1.0,))
    op = ops["recall@1.00"]
    assert op["recall"] == 1.0 and 0 < op["fpr"] <= 1.0
    assert set(op["subclass_recall"]) == {"2/6", "2/8"}
