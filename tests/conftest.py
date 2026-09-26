from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.datasets.fake_aihub import make_fake_aihub  # noqa: E402


@pytest.fixture(scope="session")
def data_cfg():
    return load_config("data")


@pytest.fixture(scope="session")
def official_cfg():
    return load_config("aihub_official_counts")


@pytest.fixture(scope="session")
def split_cfg():
    return load_config("split")


@pytest.fixture(scope="session")
def fake_root_tl(tmp_path_factory):
    """top_left 원점으로 기록된 가짜 데이터."""
    root = tmp_path_factory.mktemp("aihub_tl")
    stats = make_fake_aihub(root, n_trays_per_light=6, origin="top_left", seed=1)
    return root, stats


@pytest.fixture(scope="session")
def fake_root_bl(tmp_path_factory):
    """bottom_left 원점으로 기록된 가짜 데이터 (문서상 표기 재현)."""
    root = tmp_path_factory.mktemp("aihub_bl")
    stats = make_fake_aihub(root, n_trays_per_light=6, origin="bottom_left", seed=1)
    return root, stats
