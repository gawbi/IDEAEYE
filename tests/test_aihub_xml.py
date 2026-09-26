from pathlib import Path

from src.datasets.aihub_xml import AIHubXMLParser, flatten, parser_from_config


def test_parse_counts_and_meta(data_cfg, fake_root_tl):
    root, stats = fake_root_tl
    parser = parser_from_config(data_cfg)
    anns = list(parser.iter_dir(root))
    assert len(anns) == stats["trays"]
    assert sum(len(a.objects) for a in anns) == stats["objects"]

    a = anns[0]
    assert a.width == 640 and a.height == 480
    assert a.light in ("MONO", "COLOR")
    assert a.meta["breed"] in ("하이라인브라운", "이사브라운", "로만브라운라이트")
    assert a.meta["washed"] in ("Y", "N")
    assert a.date and len(a.date) == 10
    assert a.tray_id == Path(a.image_file).stem
    for o in a.objects:
        assert o.name.isdigit()  # 계란 객체 ID
        assert o.state and o.state_detail
        assert o.raw.get("seq")
        assert o.box.is_inside(a.width, a.height)


def test_light_codes_and_class_map(data_cfg, fake_root_tl, tmp_path):
    """실데이터처럼 type/state 가 숫자 코드일 때 light_codes / class_map 으로 이름이 붙는다."""
    root, _ = fake_root_tl
    src = sorted((root / "MONO").glob("*.xml"))[0]
    text = src.read_text(encoding="utf-8").replace("<type>MONO</type>", "<type>1</type>").replace("<state>정상</state>", "<state>1</state>")
    xml = tmp_path / "coded.xml"
    xml.write_text(text, encoding="utf-8")
    p = AIHubXMLParser(data_cfg["xml_schema"], light_from=["meta"], light_codes={"1": "MONO"}, class_map={"state": {"1": "정상"}})
    a = p.parse(xml)
    assert a.light == "MONO"
    assert all(o.state != "1" for o in a.objects)
    p2 = AIHubXMLParser(data_cfg["xml_schema"], light_from=["meta"])
    assert p2.parse(xml).light == "UNKNOWN"


def test_light_detected_from_path_when_meta_missing(data_cfg, fake_root_tl):
    root, _ = fake_root_tl
    schema = {**data_cfg["xml_schema"], "meta_fields": {}}  # meta 없음 → 경로에서 판별
    parser = AIHubXMLParser(schema, origin="top_left", light_from=["path"])
    a = parser.parse(next((root / "COLOR").glob("*.xml")))
    assert a.light == "COLOR"


def test_origin_setting_changes_boxes(data_cfg, fake_root_bl, fake_root_tl):
    """같은 seed 로 만든 두 데이터셋: bottom_left 로 기록된 xml 을 bottom_left 로 읽으면 top_left 결과와 같아야 한다."""
    root_bl, _ = fake_root_bl
    root_tl, _ = fake_root_tl
    p_bl = AIHubXMLParser(data_cfg["xml_schema"], origin="bottom_left")
    p_tl = AIHubXMLParser(data_cfg["xml_schema"], origin="top_left")
    xml_bl = sorted((root_bl / "MONO").glob("*.xml"))[0]
    xml_tl = sorted((root_tl / "MONO").glob("*.xml"))[0]
    boxes_bl = [o.box for o in p_bl.parse(xml_bl).objects]
    boxes_tl = [o.box for o in p_tl.parse(xml_tl).objects]
    assert boxes_bl == boxes_tl
    # 잘못된 설정으로 읽으면 달라진다
    wrong = [o.box for o in p_tl.parse(xml_bl).objects]
    assert wrong != boxes_tl


def test_flatten_has_split_keys(data_cfg, fake_root_tl):
    root, _ = fake_root_tl
    a = next(parser_from_config(data_cfg).iter_dir(root))
    rows = flatten(a)
    assert len(rows) == len(a.objects)
    for k in ("tray_id", "light", "date", "region_code", "washed", "breed", "state", "state_detail", "x1", "raw_y_min"):
        assert k in rows[0]
