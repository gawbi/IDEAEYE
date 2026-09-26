import pytest

from src.common.experiments import resolve_experiment


def test_resolve_latest(tmp_path):
    (tmp_path / "20260101_000000_cls_mono_effb0").mkdir()
    (tmp_path / "20260102_000000_cls_mono_effb0").mkdir()
    (tmp_path / "20260103_000000_cls_mono_effb0_smoke").mkdir()
    p = resolve_experiment("latest:cls_mono_effb0", root=tmp_path)
    assert p.name == "20260102_000000_cls_mono_effb0"
    with pytest.raises(FileNotFoundError):
        resolve_experiment("latest:nope", root=tmp_path)
