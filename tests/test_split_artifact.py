"""생성된 실제 분할 파일(data/splits/aihub_split.csv)이 있으면 누수가 없는지 검사한다. 없으면 skip."""
from pathlib import Path

import pandas as pd
import pytest

from src.common.config import load_config, resolve_path
from src.datasets.split import SPLITS, find_leaks


@pytest.fixture(scope="module")
def split_df():
    cfg = load_config("split")
    path = resolve_path(cfg["output"])
    if not path.exists():
        pytest.skip(f"no split file: {path}")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def test_split_values(split_df):
    assert set(split_df["split"].unique()) <= set(SPLITS)
    assert split_df["split"].notna().all()


def test_no_group_or_tray_leak(split_df):
    recs = split_df.to_dict("records")
    assert find_leaks(recs, ["group", "tray_id"]) == {}


def test_tray_unique_per_light(split_df):
    assert not split_df.duplicated(["light", "tray_id"]).any()


def test_each_light_has_all_splits(split_df):
    for light, sub in split_df.groupby("light"):
        assert set(sub["split"].unique()) == set(SPLITS), light
        share = sub["split"].value_counts(normalize=True)
        assert 0.6 <= share.get("train", 0) <= 0.92, (light, share.to_dict())
