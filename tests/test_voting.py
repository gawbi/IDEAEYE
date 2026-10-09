import pytest

from src.infer.voting import EggAggregator, MultiViewRule, required_view_recall, rule_from_config, simulate_fusion


def test_rules():
    assert MultiViewRule("max", 0.5).decide([0.1, 0.6, 0.2])[0]
    assert not MultiViewRule("mean", 0.5).decide([0.1, 0.6, 0.2])[0]
    assert MultiViewRule("kofn", 0.5, k=2).decide([0.6, 0.7, 0.1])[0]
    assert not MultiViewRule("kofn", 0.5, k=2).decide([0.6, 0.2, 0.1])[0]
    hy = MultiViewRule("hybrid", 0.5, high_threshold=0.9)
    assert hy.decide([0.95, 0.1, 0.1])[0]        # 한 시점 확정
    assert hy.decide([0.6, 0.55, 0.5])[0]         # 평균
    assert not hy.decide([0.8, 0.1, 0.1])[0]      # 둘 다 아님
    assert MultiViewRule().decide([]) == (False, 0.0)


def test_aggregator_completes_and_times_out():
    agg = EggAggregator(MultiViewRule("kofn", 0.5, k=2, n_views=3), timeout_s=1.0)
    assert agg.add_view("e1", 0.7, 2, ts=0.0) is None
    assert agg.add_view("e1", 0.2, 0, ts=0.1) is None
    d = agg.add_view("e1", 0.9, 3, ts=0.2)
    assert d is not None and d.reject and d.n_views == 3 and d.severity == 3 and d.reason == "complete"
    assert agg.pending == 0
    agg.add_view("e2", 0.95, 1, ts=5.0)
    assert agg.flush_timeouts(now=5.5) == []
    out = agg.flush_timeouts(now=6.5)
    assert len(out) == 1 and out[0].reason == "timeout" and out[0].n_views == 1
    assert out[0].reject is True  # 시점 부족 시 가용 시점 기준으로 판정 (안전 쪽: 강한 양성 1시점이면 배출), 사유 'timeout' 기록


def test_simulation_and_required_recall():
    s = simulate_fusion(0.8, 0.1, 3, "max")
    assert s["recall"] == pytest.approx(1 - 0.2 ** 3) and s["fpr"] == pytest.approx(1 - 0.9 ** 3)
    k = simulate_fusion(0.9, 0.096, 3, "kofn", 2)
    assert k["recall"] == pytest.approx(3 * 0.81 * 0.1 + 0.729) and k["fpr"] < 0.03
    r = required_view_recall(0.95, 3, "kofn", 2)
    assert simulate_fusion(r, 0.0, 3, "kofn", 2)["recall"] >= 0.95 and 0.85 < r < 0.9


def test_rule_from_config_defaults():
    rule = rule_from_config({"decision": {"multiview": {"method": "kofn", "threshold": 0.4, "k": 2}}})
    assert rule.method == "kofn" and rule.threshold == 0.4 and rule.n_views == 3
    assert rule_from_config({}).method == "hybrid"
