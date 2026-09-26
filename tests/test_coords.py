import pytest

from src.datasets.coords import Box, to_top_left


def test_top_left_passthrough():
    b = to_top_left(10, 20, 50, 80, image_height=100, origin="top_left")
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 50, 80)


def test_bottom_left_flips_y():
    # 좌하단 원점: y_min=20(아래), y_max=80(위) → 좌상단 원점에서는 y1=100-80=20, y2=100-20=80
    b = to_top_left(10, 20, 50, 80, image_height=100, origin="bottom_left")
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 50, 80)
    # 비대칭 예: 아래쪽에 붙은 박스는 뒤집으면 위쪽으로 간다
    b2 = to_top_left(0, 0, 10, 10, image_height=100, origin="bottom_left")
    assert (b2.y1, b2.y2) == (90, 100)


def test_invalid_origin():
    with pytest.raises(ValueError):
        to_top_left(0, 0, 1, 1, 10, origin="center")


def test_unsorted_inputs_are_normalized():
    b = to_top_left(50, 80, 10, 20, 100, "top_left")
    assert b.x1 <= b.x2 and b.y1 <= b.y2


def test_expand_clip_inside():
    b = Box(10, 10, 30, 30).expanded(0.5)  # 여백 50% → 0..40
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, 40, 40)
    c = Box(-5, -5, 200, 50).clipped(100, 40)
    assert (c.x1, c.y1, c.x2, c.y2) == (0, 0, 100, 40)
    assert Box(0, 0, 10, 10).is_inside(10, 10)
    assert not Box(0, 0, 11, 10).is_inside(10, 10)
