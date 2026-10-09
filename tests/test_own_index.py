import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("own_build_index", Path(__file__).resolve().parents[1] / "scripts" / "own_build_index.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _frames(tmp_path, n_segments=9, label_fn=None):
    rows = []
    for cam in (0, 1):
        for roi in range(3):
            for seg in range(n_segments):
                p = tmp_path / "v1" / f"c{cam}r{roi}s{seg}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                lab, sev = label_fn(cam, roi, seg) if label_fn else ("normal", "")
                rows.append({"video": "v1.mkv", "cam": cam, "roi": roi, "segment": seg, "crop_path": str(p), "label": lab, "severity": sev, "shell_color": "white", "stamp": "y"})
    pd.DataFrame(rows).to_csv(tmp_path / "v1" / "index.csv", index=False, encoding="utf-8-sig")
    return tmp_path


def test_egg_ids_group_three_views(tmp_path):
    root = _frames(tmp_path, label_fn=lambda c, r, s: ("crack", "2") if (c, r, s) == (0, 0, 4) else ("normal", ""))
    df = mod.build(root, views=3, seed=1)
    assert len(df) == 2 * 3 * 9
    assert df["egg_id"].nunique() == 2 * 3 * 3
    e = df[(df["cam"] == 0) & (df["roi"] == 0) & (df["segment"].isin([3, 4, 5]))]
    assert e["egg_id"].nunique() == 1 and (e["egg_label"] == "crack").all() and (e["egg_severity"] == 2).all()
    assert set(df.loc[df["label"] == "crack", "cls"]) == {"2/2"} and set(df.loc[df["label"] == "normal", "cls"]) == {"1/0"}
    # 같은 계란의 3시점은 같은 분할
    assert (df.groupby("egg_id")["split"].nunique() == 1).all()
    assert set(df["split"]) <= {"train", "val", "test"} and "train" in set(df["split"])


def test_unsure_and_empty_excluded(tmp_path):
    root = _frames(tmp_path, n_segments=3, label_fn=lambda c, r, s: ("unsure", "") if s == 0 else ("empty", "") if s == 1 else ("normal", ""))
    df = mod.build(root, views=3, seed=1)
    assert len(df) == 2 * 3 * 1 and (df["label"] == "normal").all()
