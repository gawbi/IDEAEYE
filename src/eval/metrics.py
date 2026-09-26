"""소분류 단위 지표 (CLAUDE.md 설계 원칙 2, 4).

- per_class_metrics: 클래스별 Precision / Recall / Support
- grouped_metrics: 세척여부·난각색 등 임의 축으로 나눠서 per_class_metrics 반복
- operating_point: "파각 Recall >= target 일 때 정상란 오검출률" 계산
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Hashable, Sequence

import numpy as np


def per_class_metrics(y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable] | None = None) -> dict[str, dict[str, float]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred), key=str)
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    out: dict[str, dict[str, float]] = {}
    for lab in labels:
        support = tp[lab] + fn[lab]
        precision = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        recall = tp[lab] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[str(lab)] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support)}
    return out


def grouped_metrics(rows: list[dict[str, Any]], group_field: str, true_field: str = "y_true", pred_field: str = "y_pred") -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(group_field)), []).append(r)
    return {g: per_class_metrics([r[true_field] for r in rs], [r[pred_field] for r in rs]) for g, rs in groups.items()}


def operating_point(scores: Sequence[float], is_positive: Sequence[bool], target_recall: float) -> dict[str, float]:
    """양성 점수 기준으로 Recall >= target_recall 을 만족하는 가장 높은 임계값과 그때의 음성 오검출률(FPR)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(is_positive, dtype=bool)
    if y.sum() == 0:
        raise ValueError("no positive samples")
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    cum_tp = np.cumsum(y_sorted)
    recall = cum_tp / y.sum()
    idx = int(np.argmax(recall >= target_recall))
    threshold = float(s_sorted[idx])
    pred_pos = s >= threshold
    fpr = float((pred_pos & ~y).sum() / max(1, (~y).sum()))
    achieved = float((pred_pos & y).sum() / y.sum())
    return {"threshold": threshold, "recall": achieved, "fpr": fpr, "target_recall": float(target_recall)}
