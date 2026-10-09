"""다중 시점(계란 1개를 120° 씩 회전해 3회 촬영) 점수 융합과 계란 단위 최종 판정.

규칙(configs/infer.yaml decision.multiview):
  method: max | mean | kofn | hybrid
  - max    : 한 시점이라도 임계값 이상이면 배출 (Recall 최대, 오검출도 증가)
  - mean   : 평균 점수 ≥ 임계값
  - kofn   : k 개 이상 시점이 임계값 이상 (예: 2 of 3 — 오검출 억제)
  - hybrid : max ≥ high_threshold 또는 mean ≥ threshold (한 시점에서만 보이는 큰 크랙은 max 로, 희미한 크랙은 평균으로 잡음)
독립 가정 하의 Recall/FPR 변환 공식은 simulate_fusion() 참고 (실측 다중 시점 데이터로 반드시 재검증).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultiViewRule:
    method: str = "hybrid"
    threshold: float = 0.5        # mean / kofn / max 의 기준
    high_threshold: float = 0.9   # hybrid 의 단일 시점 확정 기준
    k: int = 2                    # kofn
    n_views: int = 3

    def decide(self, scores: list[float]) -> tuple[bool, float]:
        """반환: (배출 여부, 융합 점수)."""
        if not scores:
            return False, 0.0
        mx, mean = max(scores), sum(scores) / len(scores)
        if self.method == "max":
            return mx >= self.threshold, mx
        if self.method == "mean":
            return mean >= self.threshold, mean
        if self.method == "kofn":
            hits = sum(s >= self.threshold for s in scores)
            return hits >= min(self.k, len(scores)), hits / len(scores)
        if self.method == "hybrid":
            return (mx >= self.high_threshold) or (mean >= self.threshold), max(mean, mx if mx >= self.high_threshold else 0.0)
        raise ValueError(self.method)


@dataclass
class EggState:
    egg_id: str
    scores: list[float] = field(default_factory=list)
    severities: list[int] = field(default_factory=list)
    first_ts: float = field(default_factory=time.time)
    last_ts: float = field(default_factory=time.time)


@dataclass
class EggDecision:
    egg_id: str
    reject: bool
    fused_score: float
    n_views: int
    severity: int
    scores: list[float]
    reason: str


class EggAggregator:
    """시점별 (egg_id, score, severity) 를 모아 n_views 가 차거나 timeout 이 지나면 최종 판정을 낸다.
    egg_id 는 기기의 회전 신호·자리 번호·사이클 번호로 만든다 (예: f"c{cam}r{roi}_{cycle}")."""

    def __init__(self, rule: MultiViewRule, timeout_s: float = 2.0):
        self.rule = rule
        self.timeout_s = timeout_s
        self._eggs: dict[str, EggState] = {}

    def add_view(self, egg_id: str, score: float, severity: int = 0, ts: float | None = None) -> EggDecision | None:
        ts = ts if ts is not None else time.time()
        st = self._eggs.get(egg_id)
        if st is None:
            st = self._eggs[egg_id] = EggState(egg_id, first_ts=ts, last_ts=ts)
        st.scores.append(float(score))
        st.severities.append(int(severity))
        st.last_ts = ts
        if len(st.scores) >= self.rule.n_views:
            return self._finalize(egg_id, "complete")
        return None

    def flush_timeouts(self, now: float | None = None) -> list[EggDecision]:
        """회전 신호 누락 등으로 시점이 덜 모인 계란을 timeout 후 현재 점수로 판정 (안전 쪽: 부족한 시점은 미검출 위험 → 판정 사유 기록)."""
        now = now if now is not None else time.time()
        out = []
        for egg_id in [e for e, s in self._eggs.items() if now - s.last_ts >= self.timeout_s]:
            out.append(self._finalize(egg_id, "timeout"))
        return out

    def _finalize(self, egg_id: str, reason: str) -> EggDecision:
        st = self._eggs.pop(egg_id)
        reject, fused = self.rule.decide(st.scores)
        return EggDecision(egg_id, reject, fused, len(st.scores), max(st.severities) if reject and st.severities else 0, st.scores, reason)

    @property
    def pending(self) -> int:
        return len(self._eggs)


def simulate_fusion(recall_view: float, fpr_view: float, n: int = 3, method: str = "max", k: int = 2) -> dict[str, float]:
    """시점 간 독립 가정 하의 최종 Recall / FPR (낙관적 근사 — 같은 크랙이 여러 시점에 보이거나 한 시점에서만 보이는 상관이 있으므로 실측 필요)."""
    from math import comb

    def at_least_k(p: float, kk: int) -> float:
        return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(kk, n + 1))

    if method == "max":
        return {"recall": 1 - (1 - recall_view) ** n, "fpr": 1 - (1 - fpr_view) ** n}
    if method == "kofn":
        return {"recall": at_least_k(recall_view, k), "fpr": at_least_k(fpr_view, k)}
    raise ValueError(method)


def required_view_recall(target_recall: float, n: int = 3, method: str = "max", k: int = 2) -> float:
    """최종 Recall 목표를 만족하는 시점별 Recall 하한 (독립 가정, 이분 탐색)."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if simulate_fusion(mid, 0.0, n, method, k)["recall"] >= target_recall:
            hi = mid
        else:
            lo = mid
    return hi


def rule_from_config(cfg: dict[str, Any]) -> MultiViewRule:
    mv = (cfg.get("decision") or {}).get("multiview") or {}
    return MultiViewRule(method=mv.get("method", "hybrid"), threshold=float(mv.get("threshold", 0.5)), high_threshold=float(mv.get("high_threshold", 0.9)), k=int(mv.get("k", 2)), n_views=int(mv.get("n_views", 3)))
