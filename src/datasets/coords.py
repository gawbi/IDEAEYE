"""bbox 좌표계 변환.

AI Hub 문서는 (x_min, y_min)을 '좌하단', (x_max, y_max)를 '우상단'으로 표기한다.
이 모듈은 설정값 origin 에 따라 항상 이미지 좌상단 원점(top-left, y 아래로 증가) 픽셀 좌표로 통일한다.
"""
from __future__ import annotations

from dataclasses import dataclass

ORIGINS = ("top_left", "bottom_left")


@dataclass(frozen=True)
class Box:
    """좌상단 원점 기준 픽셀 bbox. x1<=x2, y1<=y2 를 보장한다."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def clipped(self, img_w: float, img_h: float) -> "Box":
        return Box(
            max(0.0, min(self.x1, img_w)),
            max(0.0, min(self.y1, img_h)),
            max(0.0, min(self.x2, img_w)),
            max(0.0, min(self.y2, img_h)),
        )

    def is_inside(self, img_w: float, img_h: float) -> bool:
        return 0 <= self.x1 <= self.x2 <= img_w and 0 <= self.y1 <= self.y2 <= img_h

    def expanded(self, margin_ratio: float) -> "Box":
        """여백 비율만큼 사방으로 확장 (크롭용)."""
        mx = self.width * margin_ratio
        my = self.height * margin_ratio
        return Box(self.x1 - mx, self.y1 - my, self.x2 + mx, self.y2 + my)

    def to_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))


def to_top_left(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_height: float,
    origin: str = "top_left",
) -> Box:
    """원본 좌표를 좌상단 원점 Box 로 변환한다.

    origin == "bottom_left" 이면 y 축을 뒤집는다: y' = H - y.
    """
    if origin not in ORIGINS:
        raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")

    if origin == "bottom_left":
        y_min, y_max = image_height - y_max, image_height - y_min

    x1, x2 = sorted((float(x_min), float(x_max)))
    y1, y2 = sorted((float(y_min), float(y_max)))
    return Box(x1, y1, x2, y2)
