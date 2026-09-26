import pandas as pd
import yaml

from src.datasets.aihub_xml import flatten, parser_from_config
from src.datasets.yolo_export import export_yolo


def test_export_yolo_on_fake(data_cfg, fake_root_tl, tmp_path):
    root, _ = fake_root_tl
    df = pd.DataFrame([r for a in parser_from_config(data_cfg).iter_dir(root) for r in flatten(a)])
    trays = df.drop_duplicates(["light", "tray_id"])[["light", "tray_id"]].copy()
    trays["split"] = ["train", "val", "test"] * (len(trays) // 3) + ["train"] * (len(trays) % 3)
    cfg = {**data_cfg, "image_source": "dir", "paths": {**data_cfg["paths"], "aihub_root": str(root), "aihub_zip_root": None}}
    det_cfg = {"out_root": str(tmp_path / "det"), "seed": 1, "workers": 1, "class_field": "state", "per_light": {"MONO": {"train": 2, "val": 1, "test": 1}}}
    p = export_yolo(cfg, det_cfg, df, trays, "MONO", root=root, workers=1)
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert d["nc"] == len(d["names"]) == len(d["_codes"]) and d["train"] == "images/train"
    imgs = list((tmp_path / "det" / "MONO" / "images").rglob("*.jpg"))
    lbls = list((tmp_path / "det" / "MONO" / "labels").rglob("*.txt"))
    assert len(imgs) == len(lbls) == 4
    line = lbls[0].read_text(encoding="utf-8").splitlines()[0].split()
    assert len(line) == 5 and 0 <= int(line[0]) < d["nc"]
    assert all(0.0 <= float(v) <= 1.0 for v in line[1:])
