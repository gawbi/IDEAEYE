import pandas as pd

from src.datasets.aihub_xml import flatten, parser_from_config
from src.datasets.audit import (
    ImageSource,
    build_image_index,
    collect,
    compare_official,
    count_table,
    distribution_tables,
    imread,
    locate_image,
    matching_keys,
    origin_evidence,
    run_audit,
    share_table,
)


def test_image_source_reads_from_zip_and_dir(fake_root_tl, tmp_path):
    import zipfile

    root, _ = fake_root_tl
    jpgs = sorted((root / "MONO").glob("*.jpg"))
    z = tmp_path / "TS_01. MONO.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(jpgs[0], "/" + jpgs[0].name)  # AI Hub zip 은 루트에 '/' 접두 경로
    src = ImageSource(dirs=[root / "COLOR"], zips=[z])
    assert src.has(jpgs[0].name) and src.get(jpgs[0].name).shape[:2] == (480, 640)
    color = sorted((root / "COLOR").glob("*.jpg"))[0].name
    assert src.get(color) is not None
    assert src.get("nope.jpg") is None


def test_share_table_orders_numeric_codes():
    import pandas as pd

    df = pd.DataFrame({"light": ["MONO"] * 4 + ["COLOR"] * 2, "state": ["10", "2", "2", "1", "1", "3"]})
    t = share_table(df, "state")
    assert list(t.index) == ["1", "2", "3", "10"]
    assert t.loc["2", "MONO_%"] == 50.0


def test_count_and_official_comparison(data_cfg, official_cfg, fake_root_tl):
    root, stats = fake_root_tl
    _, df = collect(parser_from_config(data_cfg), root)
    assert len(df) == stats["objects"]
    t = count_table(df, "state_detail")
    assert t["total"].sum() == stats["objects"]
    cmp = compare_official(df, official_cfg)
    assert {"light", "level", "class", "official", "actual", "diff", "match"} <= set(cmp.columns)
    # 가짜 데이터는 공식 총계와 당연히 다르므로 불일치가 표시되어야 한다
    assert (~cmp["match"]).any()


def test_distribution_tables(data_cfg, fake_root_tl):
    root, _ = fake_root_tl
    _, df = collect(parser_from_config(data_cfg), root)
    dist = distribution_tables(df)
    for k in ("tray", "date", "region_code", "washed", "breed"):
        assert k in dist and isinstance(dist[k], pd.DataFrame)


def test_matching_keys_strip_light_token(data_cfg, fake_root_tl):
    root, _ = fake_root_tl
    _, df = collect(parser_from_config(data_cfg), root)
    m = matching_keys(df)
    assert "candidates" in m and "tray_id (원문)" in m["candidates"]
    assert m["candidates"]["tray_id (원문)"]["matched"] == 0  # MONO_/COLOR_ 접두어로 다름


def test_origin_evidence_prefers_true_origin(data_cfg, fake_root_tl, fake_root_bl):
    """가짜 데이터의 실제 원점이 무엇이든 수치 근거가 그것을 가리켜야 한다."""
    for root, expect in ((fake_root_tl[0], "top_left"), (fake_root_bl[0], "bottom_left")):
        parser = parser_from_config(data_cfg)
        index = build_image_index(root, data_cfg["image_exts"])
        votes = []
        for a in parser.iter_dir(root):
            img = imread(locate_image(a, index))
            votes.append(origin_evidence(a, img)["prefer"])
        assert votes.count(expect) > len(votes) / 2, (expect, votes)


def test_run_audit_end_to_end(data_cfg, official_cfg, fake_root_tl, tmp_path):
    root, _ = fake_root_tl
    out = tmp_path / "audit"
    report = run_audit(data_cfg, official_cfg, out, n_vis=3, seed=0, root=root)
    assert "error" not in report
    assert report["origin"]["verdict"] == "top_left"
    assert (out / "report.md").exists() and (out / "official_vs_actual.csv").exists()
    assert len(list(out.glob("bbox_*.jpg"))) == 3


def test_run_audit_missing_root(data_cfg, official_cfg, tmp_path):
    report = run_audit(data_cfg, official_cfg, tmp_path / "audit", root=tmp_path / "nothing")
    assert "error" in report
