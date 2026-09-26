"""AI Hub 형식을 흉내 낸 합성 데이터 생성기 (단위 테스트 전용).

2026-09-23 실제 XML 구조 확인 후 그 구조를 따른다: 평면 <annotation>, 객체는 <bndbox> 요소(seq/object/state/state_detail/x_min..).
실데이터는 state·state_detail 이 숫자 코드지만 여기서는 클래스 이름 문자열을 넣는다 (class_map 미적용 경로 테스트).
type 필드는 실데이터가 1/2/3 코드이고 여기서는 MONO/COLOR/STORAGE 문자열을 넣는다.
"""
from __future__ import annotations

import random
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

from PIL import Image, ImageDraw

MONO_CLASSES = {  # state: [(state_detail, weight)]
    "정상": [("정상", 60)],
    "크랙": [("완전크랙", 4), ("일자형크랙", 4), ("원형크랙", 2), ("거미줄형크랙", 2), ("잠재파형크랙", 3), ("파형크랙", 4), ("미세크랙", 4)],
    "혈반": [("먹통", 10), ("혈반육반", 2)],
    "탈색": [("탈색", 2)],
    "이물질": [("이물질", 2)],
    "기형": [("기형", 1)],
}
COLOR_CLASSES = {
    "정상": [("정상", 70)],
    "크랙": [("완전크랙", 5), ("일자형크랙", 5), ("미세크랙", 4)],
    "이물질": [("이물질", 6)],
    "탈색": [("탈색", 5)],
    "외형이상": [("외형이상", 5)],
}
BREEDS = ["하이라인브라운", "이사브라운", "로만브라운라이트"]
REGIONS = ["A01", "A02", "B01", "B02", "C01"]
DATES = ["2023-03-01", "2023-03-02", "2023-03-15", "2023-04-10", "2023-04-11", "2023-05-20"]


def _weighted_choice(rng: random.Random, table: dict[str, list[tuple[str, int]]]) -> tuple[str, str]:
    flat = [(s, d, w) for s, lst in table.items() for d, w in lst]
    total = sum(w for _, _, w in flat)
    r = rng.uniform(0, total)
    acc = 0.0
    for s, d, w in flat:
        acc += w
        if r <= acc:
            return s, d
    return flat[-1][0], flat[-1][1]


def make_fake_aihub(
    root: str | Path,
    n_trays_per_light: int = 4,
    origin: str = "top_left",
    seed: int = 0,
    size: tuple[int, int] = (640, 480),
    grid: tuple[int, int] = (6, 5),
    fill_prob: float = 0.7,
) -> dict[str, int]:
    """root/{MONO,COLOR}/ 아래에 jpg+xml 쌍을 만든다. origin 에 따라 y 좌표 기록 방식이 바뀐다."""
    rng = random.Random(seed)
    root = Path(root)
    W, H = size
    cols, rows = grid
    cell_w, cell_h = W // cols, H // rows
    n_objects = 0
    next_object_id = 5100000

    for light, table in (("MONO", MONO_CLASSES), ("COLOR", COLOR_CLASSES)):
        d = root / light
        d.mkdir(parents=True, exist_ok=True)
        for t in range(n_trays_per_light):
            date = rng.choice(DATES)
            region = rng.choice(REGIONS)
            stem = f"{light}_{date.replace('-', '')}_{region}_{t:04d}"
            bg = 20 if light == "MONO" else 200
            img = Image.new("RGB", (W, H), (bg, bg, bg))
            draw = ImageDraw.Draw(img)

            ann = Element("annotation")
            SubElement(ann, "category").text = "1"
            SubElement(ann, "regional_code").text = region
            SubElement(ann, "varieties").text = rng.choice(BREEDS)
            SubElement(ann, "wash").text = rng.choice(["Y", "N"])
            SubElement(ann, "classification").text = "1"
            SubElement(ann, "days").text = str(rng.randint(100, 150))
            SubElement(ann, "type").text = light
            SubElement(ann, "create_date").text = date
            SubElement(ann, "date").text = f"{date} {rng.randint(8, 17):02d}:{rng.randint(0, 59):02d}:00"
            SubElement(ann, "temperature").text = f"{rng.uniform(18, 28):.1f}"
            SubElement(ann, "humidity").text = f"{rng.uniform(40, 70):.0f}"
            SubElement(ann, "file_name").text = stem + ".jpg"
            SubElement(ann, "fresh_day").text = "0"
            SubElement(ann, "breeding_env").text = rng.choice(["케이지", "평사"])
            size_el = SubElement(ann, "size")
            SubElement(size_el, "width").text = str(W)
            SubElement(size_el, "height").text = str(H)
            count_el = SubElement(ann, "object_count")

            seq = 0
            for r in range(rows):
                for c in range(cols):
                    if rng.random() > fill_prob:
                        continue
                    pad_x, pad_y = int(cell_w * 0.15), int(cell_h * 0.12)
                    x1, y1 = c * cell_w + pad_x, r * cell_h + pad_y
                    x2, y2 = (c + 1) * cell_w - pad_x, (r + 1) * cell_h - pad_y
                    state, detail = _weighted_choice(rng, table)
                    color = (235, 235, 235) if light == "MONO" else (150, 90, 40)
                    draw.ellipse([x1, y1, x2, y2], fill=color)
                    if state == "크랙":
                        draw.line([x1 + 5, y1 + 5, x2 - 5, y2 - 5], fill=(60, 60, 60), width=2)

                    seq += 1
                    bb = SubElement(ann, "bndbox")
                    SubElement(bb, "seq").text = str(seq)
                    SubElement(bb, "object").text = str(next_object_id)
                    next_object_id += 1
                    SubElement(bb, "state").text = state
                    SubElement(bb, "state_detail").text = detail
                    if origin == "bottom_left":
                        ymin_rec, ymax_rec = H - y2, H - y1
                    else:
                        ymin_rec, ymax_rec = y1, y2
                    SubElement(bb, "x_min").text = str(x1)
                    SubElement(bb, "y_min").text = str(ymin_rec)
                    SubElement(bb, "x_max").text = str(x2)
                    SubElement(bb, "y_max").text = str(ymax_rec)
                    n_objects += 1
            count_el.text = str(seq)

            img.save(d / (stem + ".jpg"), quality=90)
            ElementTree(ann).write(d / (stem + ".xml"), encoding="utf-8", xml_declaration=True)

    return {"trays": 2 * n_trays_per_light, "objects": n_objects}


if __name__ == "__main__":  # python -m src.datasets.fake_aihub <out_dir>
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/fake_aihub"
    print(make_fake_aihub(out, n_trays_per_light=10))
