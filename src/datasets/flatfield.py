"""Flat-field(조명 불균일) 보정 (Phase 6, CLAUDE.md 4장 '조명/난각두께').

정석: 균일 확산판을 찍은 참조 영상 F 로 I' = I * mean(F) / F. 참조가 없으면 영상 자체의 저주파 성분(큰 가우시안 블러)을 조명장으로 추정한다.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_illumination(img: np.ndarray, sigma_frac: float = 0.25) -> np.ndarray:
    """영상의 저주파 성분(조명장) 추정. sigma 는 짧은 변의 비율."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sigma = max(1.0, sigma_frac * min(gray.shape[:2]))
    return cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)


def flat_field_correct(img: np.ndarray, reference: np.ndarray | None = None, sigma_frac: float = 0.25, eps: float = 1.0, reference_smooth_frac: float = 0.02) -> np.ndarray:
    """reference(참조 영상, 같은 크기) 가 있으면 그것으로(노이즈 제거용 약한 블러만), 없으면 자체 추정 조명장(강한 블러)으로 보정. 반환 uint8."""
    field = estimate_illumination(reference, reference_smooth_frac) if reference is not None else estimate_illumination(img, sigma_frac)
    gain = field.mean() / np.maximum(field, eps)
    if img.ndim == 3:
        gain = gain[:, :, None]
    out = np.clip(img.astype(np.float32) * gain, 0, 255)
    return out.astype(np.uint8)
