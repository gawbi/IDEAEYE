import json

import yaml
from PIL import Image

from src.datasets.roboflow_yolo import analyze_dataset, read_yolo_labels, source_id, update_manifest, write_datasets_doc, write_fixed_data_yaml


def test_read_yolo_labels_bbox_and_polygon(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n2 0.1\n", encoding="utf-8")
    rows = read_yolo_labels(p)
    assert len(rows) == 2
    assert rows[0].fmt == "bbox" and (rows[0].w, rows[0].h) == (0.2, 0.4)
    poly = rows[1]
    assert poly.fmt == "polygon" and poly.n_points == 4
    assert (round(poly.cx, 3), round(poly.cy, 3), round(poly.w, 3), round(poly.h, 3)) == (0.3, 0.2, 0.4, 0.2)


def _make(root, split, items):
    """items: [(filename, [(cls, cx, cy, w, h)] | None)]. None 이면 라벨 파일 없음."""
    (root / split / "images").mkdir(parents=True)
    (root / split / "labels").mkdir(parents=True)
    for name, rows in items:
        Image.new("L", (64, 48), 128).save(root / split / "images" / name)
        if rows is not None:
            (root / split / "labels" / (name.rsplit(".", 1)[0] + ".txt")).write_text(
                "\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows), encoding="utf-8"
            )


def _dataset(tmp_path, leak=False):
    root = tmp_path / "eggs"
    _make(root, "train", [
        ("farmA_tray7_jpg.rf.0123456789abcdef0123456789abcdef.jpg", [(0, 0.5, 0.5, 0.2, 0.2)]),
        ("farmA_tray7_jpg.rf.fedcba9876543210fedcba9876543210.jpg", [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.2, 0.2, 0.1, 0.1)]),
        ("farmA_tray8_jpg.rf.00000000000000000000000000000000.jpg", []),
        ("farmB_tray1_jpg.rf.11111111111111111111111111111111.jpg", None),
    ])
    valid_src = "farmA_tray7" if leak else "farmC_tray2"
    _make(root, "valid", [(f"{valid_src}_jpg.rf.22222222222222222222222222222222.jpg", [(1, 0.5, 0.5, 0.3, 0.3)])])
    (root / "data.yaml").write_text(yaml.safe_dump({
        "train": "../train/images", "val": "../valid/images", "test": "../test/images",
        "nc": 2, "names": ["cracked-egg", "normal-egg"],
        "roboflow": {"workspace": "w", "project": "eggs", "version": 8, "license": "CC BY 4.0", "url": "https://universe.roboflow.com/w/eggs/dataset/8"},
    }), encoding="utf-8")
    (root / "README.roboflow.txt").write_text(
        "eggs - v8\n==\n\nThis dataset was exported via roboflow.com on May 7, 2024 at 11:16 AM GMT\n\n"
        "The dataset includes 5 images.\n\nThe following pre-processing was applied to each image:\n* Resize to 640x640 (Stretch)\n* Grayscale (CRT phosphor)\n\n"
        "The following augmentation was applied to create 3 versions of each source image:\n* 50% probability of horizontal flip\n\n", encoding="utf-8")
    (root / "README.dataset.txt").write_text("# eggs\nhttps://universe.roboflow.com/w/eggs\n\nLicense: CC BY 4.0\n", encoding="utf-8")
    return root


def test_source_id():
    assert source_id("IMG_1_jpg.rf.0123456789abcdef0123456789abcdef.jpg") == "IMG_1"
    assert source_id("frame 12_png.rf.0123456789abcdef0123456789abcdef.png") == "frame 12"
    assert source_id("plain.jpg") == "plain"


def test_analyze_counts_and_no_leak(tmp_path):
    root = _dataset(tmp_path)
    s = analyze_dataset(root)
    tr = s["splits"]["train"]
    assert tr["images"] == 4 and tr["labels"] == 3 and tr["missing_labels"] == 1 and tr["empty_labels"] == 1
    assert tr["instances"] == {"cracked-egg": 2, "normal-egg": 1}
    assert tr["sources"] == 3  # farmA_tray7(증강 2장), farmA_tray8, farmB_tray1
    assert tr["sizes"] == {"64x48": 4}
    assert tr["label_format"] == "bbox"
    assert s["splits"]["val"]["instances"] == {"normal-egg": 1}
    assert "test" not in s["splits"]
    assert s["source_leak"] == {}
    assert s["totals"] == {"images": 5, "sources": 4, "instances": {"cracked-egg": 2, "normal-egg": 2}}
    assert s["readme"]["augmentation_versions"] == 3 and "Grayscale (CRT phosphor)" in s["readme"]["preprocessing"]
    assert s["license"] == "CC BY 4.0" and s["version"] == 8


def test_analyze_detects_source_leak(tmp_path):
    s = analyze_dataset(_dataset(tmp_path, leak=True))
    lk = s["source_leak"]["train∩val"]
    assert lk["count"] == 1 and lk["examples"] == ["farmA_tray7"]
    assert lk["ambiguous_generic_names"] == 0 and lk["confirmed_by_name"] == ["farmA_tray7"]


def test_generic_names_flagged_ambiguous(tmp_path):
    root = tmp_path / "num"
    _make(root, "train", [
        ("7_jpg.rf.0123456789abcdef0123456789abcdef.jpg", [(0, 0.5, 0.5, 0.2, 0.2)]),
        ("download_jpg.rf.0123456789abcdef0123456789abcdef.jpg", [(0, 0.5, 0.5, 0.2, 0.2)]),
        ("images-19-_jpg.rf.0123456789abcdef0123456789abcdef.jpg", [(0, 0.5, 0.5, 0.2, 0.2)]),
        ("LINE_ALBUM_-_240410_10_jpg.rf.0123456789abcdef0123456789abcdef.jpg", [(0, 0.5, 0.5, 0.2, 0.2)]),
    ])
    _make(root, "valid", [
        ("7_jpg.rf.fedcba9876543210fedcba9876543210.jpg", [(1, 0.5, 0.5, 0.2, 0.2)]),
        ("download_jpg.rf.fedcba9876543210fedcba9876543210.jpg", [(1, 0.5, 0.5, 0.2, 0.2)]),
        ("images-19-_jpg.rf.fedcba9876543210fedcba9876543210.jpg", [(1, 0.5, 0.5, 0.2, 0.2)]),
        ("LINE_ALBUM_-_240410_10_jpg.rf.fedcba9876543210fedcba9876543210.jpg", [(1, 0.5, 0.5, 0.2, 0.2)]),
    ])
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 2, "names": ["a", "b"]}), encoding="utf-8")
    lk = analyze_dataset(root)["source_leak"]["train∩val"]
    assert lk["count"] == 4 and lk["ambiguous_generic_names"] == 3
    assert lk["confirmed_by_name"] == ["LINE_ALBUM_-_240410_10"]


def test_fixed_data_yaml_and_docs(tmp_path):
    root = _dataset(tmp_path)
    write_fixed_data_yaml(root, project_root=tmp_path)
    assert (root / "data.roboflow.yaml").exists()
    fixed = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert fixed["path"] == "eggs" and fixed["train"] == "train/images" and fixed["val"] == "valid/images"
    assert "test" not in fixed and fixed["names"] == ["cracked-egg", "normal-egg"]
    # 두 번 실행해도 원본이 덮이지 않는다
    write_fixed_data_yaml(root, project_root=tmp_path)
    assert "roboflow" in yaml.safe_load((root / "data.roboflow.yaml").read_text(encoding="utf-8"))

    s = analyze_dataset(root)
    m = update_manifest(tmp_path, s)
    assert "eggs" in json.loads(m.read_text(encoding="utf-8"))
    doc = write_datasets_doc(tmp_path, tmp_path / "docs" / "datasets.md")
    text = doc.read_text(encoding="utf-8")
    assert "## eggs" in text and "| train | 4 | 3 | 3 | 1 | 1 | 2 | 1 |" in text
