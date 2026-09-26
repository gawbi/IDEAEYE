"""AI Hub「계란 데이터」(dataSetSn=71504) XML 파서.

실제 스키마를 아직 확인하지 못했으므로 태그 경로는 configs/data.yaml 의 xml_schema 에서 읽는다.
각 필드는 후보 경로 목록이며 앞에서부터 시도한다. 실데이터 확인 후 후보를 정리할 것.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .coords import Box, to_top_left

LIGHTS = ("STORAGE", "MONO", "COLOR")  # STORAGE 를 먼저 검사 (경로에 STORAGE 와 MONO 가 함께 있을 수 있음)


@dataclass
class EggObject:
    name: str
    state: str
    state_detail: str
    box: Box
    raw: dict[str, Any] = field(default_factory=dict)  # 변환 전 원본 좌표 등


@dataclass
class TrayAnnotation:
    xml_path: Path
    image_file: str
    width: int
    height: int
    light: str  # MONO | COLOR | UNKNOWN
    meta: dict[str, Any]
    objects: list[EggObject]

    @property
    def tray_id(self) -> str:
        return Path(self.image_file).stem

    @property
    def date(self) -> str | None:
        dt = self.meta.get("datetime")
        if not dt:
            return None
        m = re.search(r"(\d{4})[-./]?(\d{2})[-./]?(\d{2})", str(dt))
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else str(dt)[:10]


class AIHubXMLParser:
    def __init__(
        self,
        schema: dict[str, Any],
        origin: str = "top_left",
        light_from: list[str] | None = None,
        light_codes: dict[str, str] | None = None,
        class_map: dict[str, dict[str, str]] | None = None,
    ):
        self.schema = schema
        self.origin = origin
        self.light_from = light_from or ["meta", "path", "filename"]
        self.light_codes = {str(k): v for k, v in (light_codes or {}).items()}
        self.class_map = {k: {str(a): b for a, b in (v or {}).items()} for k, v in (class_map or {}).items()}

    def _map(self, field: str, value: str | None) -> str | None:
        """코드 → 이름 매핑 (class_map 에 없으면 원값 유지)."""
        if value is None:
            return None
        return self.class_map.get(field, {}).get(str(value), value)

    # ---- 내부 유틸 -----------------------------------------------------
    @staticmethod
    def _find_text(elem: ET.Element, candidates: list[str]) -> str | None:
        for path in candidates:
            node = elem.find(path)
            if node is None:
                node = elem.find(f".//{path}")
            if node is not None and node.text is not None and node.text.strip() != "":
                return node.text.strip()
        return None

    def _find_objects(self, root: ET.Element) -> list[ET.Element]:
        for path in self.schema["object"]:
            found = root.findall(path)
            if not found:
                found = root.findall(f".//{path}")
            if found:
                return found
        return []

    def _detect_light(self, meta: dict[str, Any], xml_path: Path, image_file: str) -> str:
        for source in self.light_from:
            text = ""
            if source == "meta":
                text = str(meta.get("light") or "")
                if text in self.light_codes:
                    return self.light_codes[text]
            elif source == "path":
                text = str(xml_path)
            elif source == "filename":
                text = image_file
            upper = text.upper()
            for light in LIGHTS:
                if light in upper:
                    return light
        return "UNKNOWN"

    # ---- 공개 API ------------------------------------------------------
    def parse(self, xml_path: str | Path) -> TrayAnnotation:
        xml_path = Path(xml_path)
        root = ET.parse(xml_path).getroot()
        s = self.schema

        image_file = self._find_text(root, s["filename"]) or xml_path.with_suffix(".jpg").name
        width = int(float(self._find_text(root, s["width"]) or 0))
        height = int(float(self._find_text(root, s["height"]) or 0))

        meta: dict[str, Any] = {}
        for key, cands in s.get("meta_fields", {}).items():
            meta[key] = self._map(key, self._find_text(root, cands))

        light = self._detect_light(meta, xml_path, image_file)

        objects: list[EggObject] = []
        of = s["object_fields"]
        for obj in self._find_objects(root):
            coords = {}
            for k in ("x_min", "y_min", "x_max", "y_max"):
                v = self._find_text(obj, of[k])
                if v is None:
                    raise ValueError(f"{xml_path}: object missing {k}")
                coords[k] = float(v)
            box = to_top_left(coords["x_min"], coords["y_min"], coords["x_max"], coords["y_max"], height, self.origin)
            if "seq" in of:
                coords["seq"] = self._find_text(obj, of["seq"])
            objects.append(
                EggObject(
                    name=self._find_text(obj, of["name"]) or "",
                    state=self._map("state", self._find_text(obj, of["state"])) or "",
                    state_detail=self._map("state_detail", self._find_text(obj, of["state_detail"])) or "",
                    box=box,
                    raw=coords,
                )
            )

        return TrayAnnotation(xml_path, image_file, width, height, light, meta, objects)

    def iter_dir(self, root: str | Path, pattern: str = "*.xml") -> Iterator[TrayAnnotation]:
        for p in sorted(Path(root).rglob(pattern)):
            yield self.parse(p)


def flatten(ann: TrayAnnotation) -> list[dict[str, Any]]:
    """객체 단위 평탄화 레코드 (pandas / split 입력용)."""
    base = {
        "xml_path": str(ann.xml_path),
        "image_file": ann.image_file,
        "tray_id": ann.tray_id,
        "light": ann.light,
        "date": ann.date,
        "width": ann.width,
        "height": ann.height,
        **{f"meta_{k}": v for k, v in ann.meta.items()},
    }
    # split 모듈이 쓰는 표준 키
    base["region_code"] = ann.meta.get("region_code")
    base["washed"] = ann.meta.get("washed")
    base["breed"] = ann.meta.get("breed")

    rows = []
    for i, o in enumerate(ann.objects):
        rows.append(
            {
                **base,
                "obj_index": i,
                "seq": o.raw.get("seq"),
                "name": o.name,
                "object_id": o.name,
                "state": o.state,
                "state_detail": o.state_detail,
                "x1": o.box.x1,
                "y1": o.box.y1,
                "x2": o.box.x2,
                "y2": o.box.y2,
                "raw_x_min": o.raw.get("x_min"),
                "raw_y_min": o.raw.get("y_min"),
                "raw_x_max": o.raw.get("x_max"),
                "raw_y_max": o.raw.get("y_max"),
            }
        )
    return rows


def parser_from_config(cfg: dict[str, Any]) -> AIHubXMLParser:
    return AIHubXMLParser(
        schema=cfg["xml_schema"],
        origin=cfg["coords"]["origin"],
        light_from=cfg.get("light_from"),
        light_codes=cfg.get("light_codes"),
        class_map=cfg.get("class_map"),
    )
