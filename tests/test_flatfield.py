import numpy as np

from src.datasets.flatfield import flat_field_correct


def test_flat_field_removes_gradient():
    h, w = 120, 160
    ramp = np.linspace(0.4, 1.0, w, dtype=np.float32)[None, :].repeat(h, 0)  # 좌측이 어두운 조명
    img = (np.full((h, w), 200, np.float32) * ramp).astype(np.uint8)
    img3 = np.stack([img] * 3, -1)
    ref = img3.copy()
    out = flat_field_correct(img3, reference=ref)
    left, right = out[:, :20, 0].mean(), out[:, -20:, 0].mean()
    assert abs(left - right) < 8  # 보정 후 좌우 밝기 차이가 거의 사라짐
    assert abs(img[:, :20].mean() - img[:, -20:].mean()) > 80  # 원본은 차이가 큼
    self_corrected = flat_field_correct(img3)
    assert self_corrected.shape == img3.shape and self_corrected.dtype == np.uint8
