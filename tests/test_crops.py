import cv2
import numpy as np
import pandas as pd

from src.datasets.aihub_xml import flatten, parser_from_config
from src.datasets.coords import Box
from src.datasets.crops import crop_egg, run_crops


def test_crop_egg_margin_and_resize():
    img = np.zeros((1080, 1440, 3), dtype=np.uint8)
    crop = crop_egg(img, Box(200, 300, 430, 600), margin=0.10, long_side=320)
    # 230×300 + 10% 여백 → 276×360 → 긴 변 320 으로 축소
    assert max(crop.shape[:2]) == 320
    assert abs(crop.shape[1] / crop.shape[0] - 276 / 360) < 0.02
    small = crop_egg(img, Box(0, 0, 50, 60), margin=0.0, long_side=320)
    assert small.shape[:2] == (60, 50)  # 작으면 확대하지 않음


def test_run_crops_on_fake(data_cfg, fake_root_tl, tmp_path):
    root, stats = fake_root_tl
    df = pd.DataFrame([r for a in parser_from_config(data_cfg).iter_dir(root) for r in flatten(a)])
    cfg = {**data_cfg, "image_source": "dir", "paths": {**data_cfg["paths"], "aihub_root": str(root), "aihub_zip_root": None}}
    crops_cfg = {"out_root": str(tmp_path / "crops"), "margin": 0.1, "long_side": 64, "jpeg_quality": 90, "lights": ["MONO", "COLOR"],
                 "workers": 1, "index_csv": str(tmp_path / "crops" / "index.csv")}
    split_df = df.drop_duplicates(["light", "tray_id"])[["light", "tray_id"]].assign(split="train")
    index = run_crops(cfg, crops_cfg, df, split_df, root=root, workers=1)
    assert len(index) == stats["objects"] and index["exists"].all()
    assert set(index["split"]) == {"train"}
    sample = cv2.imread(index["crop_path"].iloc[0])
    assert sample is not None and max(sample.shape[:2]) <= 64
    # 재실행: 이미 있는 크롭은 건너뛰고 index 는 그대로
    again = run_crops(cfg, crops_cfg, df, split_df, root=root, workers=1)
    assert len(again) == len(index)
