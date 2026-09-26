import random

import pytest

from src.datasets.aihub_xml import flatten, parser_from_config
from src.datasets.split import SPLITS, assign_split, connected_group_keys, find_leaks, group_split, split_class_table, split_ratio_summary


def _records(n_groups=40, per_group=(5, 40), seed=0):
    """date 20종 × region 7종이 얽힌 레코드. connected 모드에서는 성분이 몇 개로 합쳐진다."""
    rng = random.Random(seed)
    recs = []
    for g in range(n_groups):
        date, region = f"2023-01-{g % 20 + 1:02d}", f"R{g % 7}"
        for i in range(rng.randint(*per_group)):
            recs.append({"tray_id": f"T{g}_{i // 10}", "date": date, "region_code": region, "state_detail": rng.choice(["정상", "미세크랙", "먹통"])})
    return recs


def _sparse_records(seed=0):
    """date 와 region 이 1:1 로만 짝지어진 레코드 → connected 성분이 작아 비율을 맞출 수 있다."""
    rng = random.Random(seed)
    recs = []
    for g in range(60):
        for i in range(rng.randint(5, 30)):
            recs.append({"tray_id": f"T{g}_{i // 10}", "date": f"2023-02-{g + 1:02d}", "region_code": f"R{g}", "state_detail": rng.choice(["정상", "미세크랙"])})
    return recs


def test_group_split_ratios_and_determinism():
    sizes = {f"g{i}": random.Random(i).randint(1, 50) for i in range(200)}
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = group_split(sizes, ratios, seed=42)
    b = group_split(sizes, ratios, seed=42)
    assert a == b
    total = sum(sizes.values())
    for s, r in ratios.items():
        got = sum(n for g, n in sizes.items() if a[g] == s) / total
        assert abs(got - r) < 0.03, (s, got)
    assert group_split(sizes, ratios, seed=7) != a


def test_combined_mode_no_leak_on_group_and_tray(split_cfg):
    recs = assign_split(_records(), split_cfg["group_keys"], split_cfg["ratios"], split_cfg["seed"], mode="combined")
    assert set(r["split"] for r in recs) <= set(SPLITS)
    assert find_leaks(recs, ["group", "tray_id"]) == {}


def test_connected_mode_no_leak_on_each_key(split_cfg):
    recs = assign_split(_sparse_records(), split_cfg["group_keys"], split_cfg["ratios"], split_cfg["seed"], mode="connected")
    leaks = find_leaks(recs, ["group", "tray_id"] + list(split_cfg["group_keys"]))
    assert leaks == {}, leaks
    ratios = split_ratio_summary(recs)
    assert abs(ratios["train"] - 0.8) < 0.08


def test_connected_components_merge_shared_values():
    recs = [
        {"date": "d1", "region_code": "r1"},
        {"date": "d1", "region_code": "r2"},
        {"date": "d2", "region_code": "r2"},
        {"date": "d3", "region_code": "r3"},
    ]
    comps = connected_group_keys(recs, ["date", "region_code"])
    assert comps[0] == comps[1] == comps[2]
    assert comps[3] != comps[0]


def test_invalid_mode():
    with pytest.raises(ValueError):
        assign_split([], ["date"], {"train": 1.0}, 0, mode="nope")


def test_find_leaks_detects_mixed_group():
    recs = [{"tray_id": "A", "split": "train"}, {"tray_id": "A", "split": "test"}, {"tray_id": "B", "split": "val"}]
    leaks = find_leaks(recs, ["tray_id"])
    assert leaks == {"tray_id": {"A": {"train", "test"}}}


def test_split_on_fake_aihub(data_cfg, split_cfg, fake_root_tl):
    root, _ = fake_root_tl
    recs = [r for a in parser_from_config(data_cfg).iter_dir(root) for r in flatten(a)]
    mode = split_cfg.get("group_mode", "combined")
    recs = assign_split(recs, split_cfg["group_keys"], split_cfg["ratios"], split_cfg["seed"], mode=mode)
    keys = ["group", "tray_id"] + (list(split_cfg["group_keys"]) if mode == "connected" else [])
    assert find_leaks(recs, keys) == {}
    table = split_class_table(recs, "state_detail")
    assert set(SPLITS) <= set(table.columns)
    assert table["total"].sum() == len(recs)
    ratios = split_ratio_summary(recs)
    assert abs(sum(ratios.values()) - 1.0) < 1e-9
