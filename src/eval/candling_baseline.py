"""Phase 5: 딥러닝 없는 밝기 임계값 기반 혈반 검출 베이스라인 (투광 MONO 크롭).

아이디어: 투광 영상에서 혈반·육반은 계란 내부의 국소적으로 어두운 반점 → 계란 마스크 안에서 black-hat(국소 암부) 반응의 크기로 점수화.
점수가 클수록 혈반 가능성. 선행연구(단순 임계값, 갈색란 ~88%) 재현 목적.
"""
from __future__ import annotations

import cv2
import numpy as np


def egg_mask(gray: np.ndarray) -> np.ndarray:
    """투광 크롭에서 밝은 계란 영역 마스크 (Otsu + 가장 큰 연결성분 + 침식으로 가장자리 제외)."""
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(gray, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == biggest).astype(np.uint8) * 255
    # 구멍 메우기: 어두운 혈반·먹통 영역은 Otsu 에서 배경으로 떨어져 마스크에 구멍이 생기므로, 테두리에서 닿지 않는 배경을 계란으로 편입
    flood = m.copy()
    ff = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    m = cv2.bitwise_or(m, cv2.bitwise_not(flood))
    k = max(3, int(0.06 * min(gray.shape)) | 1)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m > 0


def blood_spot_score(img_bgr: np.ndarray, kernel_frac: float = 0.12, blur: int = 3) -> dict[str, float]:
    """국소 암부 반응. 반환: max(최대 반응), area(임계 이상 픽셀 비율), contrast(계란 내부 대비)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    mask = egg_mask(gray)
    if mask.sum() < 50:
        return {"max": 0.0, "area": 0.0, "contrast": 0.0, "egg_px": float(mask.sum())}
    k = max(5, int(kernel_frac * min(gray.shape)) | 1)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    blackhat = closed.astype(np.int16) - gray.astype(np.int16)  # 주변보다 어두운 정도
    inside = blackhat[mask]
    egg_mean = float(gray[mask].mean())
    rel = inside / max(1.0, egg_mean)  # 계란 밝기 대비 상대 암부
    return {
        "max": float(rel.max()),
        "area": float((rel > 0.15).mean()),
        "contrast": float(gray[mask].std() / max(1.0, egg_mean)),
        "egg_px": float(mask.sum()),
    }
