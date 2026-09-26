"""Group split — 같은 판·촬영일·지역코드가 train/val/test 에 섞이지 않게 분할 (CLAUDE.md 설계 원칙 3).

레코드는 flatten() 이 만든 dict 목록(객체 단위)이다. 그룹 키는 configs/split.yaml 의 group_keys.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable

import pandas as pd

SPLITS = ("train", "val", "test")


GROUP_MODES = ("combined", "connected")


def make_group_key(row: dict[str, Any], keys: Iterable[str]) -> str:
    return "|".join(str(row.get(k)) for k in keys)


def connected_group_keys(records: list[dict[str, Any]], keys: list[str]) -> list[str]:
    """각 키 값이 단독으로도 두 분할에 걸치지 않도록, 키 값들을 union-find 로 연결해 성분 ID 를 만든다.

    예: (date=d1, region=r1), (date=d1, region=r2), (date=d2, region=r2) 는 모두 한 성분.
    키가 촘촘히 얽혀 있으면 성분이 거대해져 분할 비율을 맞추기 어려울 수 있다 (감사 후 키 조정).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in records:
        nodes = [f"{k}={r.get(k)}" for k in keys]
        for n in nodes[1:]:
            union(nodes[0], n)
    return [find(f"{keys[0]}={r.get(keys[0])}") for r in records]


def group_split(group_sizes: dict[str, int], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """그룹 → 분할 이름. 그룹을 seed 로 섞은 뒤, 목표 비율에 가장 못 미치는 분할에 순서대로 배정한다."""
    total = sum(group_sizes.values())
    if total == 0:
        return {}
    names = list(ratios)
    rng = random.Random(seed)
    groups = sorted(group_sizes)  # 결정적 순서 후 셔플
    rng.shuffle(groups)
    # 큰 그룹부터 배정하면 비율 오차가 줄어든다 (셔플로 동률 순서를 무작위화)
    groups.sort(key=lambda g: -group_sizes[g])

    assigned: dict[str, str] = {}
    filled = {n: 0 for n in names}
    for g in groups:
        deficit = {n: ratios[n] - filled[n] / total for n in names}
        target = max(names, key=lambda n: deficit[n])
        assigned[g] = target
        filled[target] += group_sizes[g]
    return assigned


def assign_split(
    records: list[dict[str, Any]],
    group_keys: list[str],
    ratios: dict[str, float],
    seed: int,
    mode: str = "combined",
) -> list[dict[str, Any]]:
    """각 레코드에 'split' 과 'group' 키를 추가한 새 목록을 돌려준다.

    mode="combined": 키 조합(튜플)이 그룹. 조합이 같은 레코드만 같은 분할을 보장.
    mode="connected": 키 값 하나라도 공유하면 같은 그룹. 각 키 단독으로도 누수 없음을 보장.
    """
    if mode not in GROUP_MODES:
        raise ValueError(f"mode must be one of {GROUP_MODES}")
    if mode == "connected":
        group_of = connected_group_keys(records, group_keys)
    else:
        group_of = [make_group_key(r, group_keys) for r in records]
    sizes = Counter(group_of)
    mapping = group_split(dict(sizes), ratios, seed)
    return [{**r, "group": g, "split": mapping[g]} for r, g in zip(records, group_of)]


def find_leaks(records: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, dict[str, set[str]]]:
    """key 별로 둘 이상의 분할에 걸친 값을 찾는다. 비어 있으면 누수 없음."""
    leaks: dict[str, dict[str, set[str]]] = {}
    for key in keys:
        seen: dict[str, set[str]] = defaultdict(set)
        for r in records:
            seen[str(r.get(key))].add(r["split"])
        bad = {v: s for v, s in seen.items() if len(s) > 1}
        if bad:
            leaks[key] = bad
    return leaks


def split_class_table(records: list[dict[str, Any]], class_field: str = "state_detail") -> pd.DataFrame:
    df = pd.DataFrame(records)
    table = pd.crosstab(df[class_field], df["split"]).reindex(columns=list(SPLITS), fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def rare_class_warnings(table: pd.DataFrame, min_test: int) -> list[str]:
    msgs = []
    for cls, row in table.iterrows():
        if row.get("test", 0) < min_test:
            msgs.append(f"{cls}: test={int(row.get('test', 0))} < {min_test}")
    return msgs


def split_ratio_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    c = Counter(r["split"] for r in records)
    n = sum(c.values())
    return {s: c.get(s, 0) / n if n else 0.0 for s in SPLITS}
