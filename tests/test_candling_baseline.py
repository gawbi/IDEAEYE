import cv2
import numpy as np

from src.eval.candling_baseline import blood_spot_score, egg_mask


def _egg(spot: bool) -> np.ndarray:
    img = np.full((300, 240, 3), 15, dtype=np.uint8)
    cv2.ellipse(img, (120, 150), (95, 130), 0, 0, 360, (200, 200, 200), -1)
    if spot:
        cv2.circle(img, (130, 160), 9, (60, 60, 60), -1)
    return img


def test_egg_mask_covers_egg_not_background():
    m = egg_mask(cv2.cvtColor(_egg(False), cv2.COLOR_BGR2GRAY))
    assert m[150, 120] and not m[5, 5]


def test_blood_spot_score_separates():
    clean, spot = blood_spot_score(_egg(False)), blood_spot_score(_egg(True))
    assert spot["max"] > clean["max"] + 0.2
    assert spot["area"] > clean["area"]
