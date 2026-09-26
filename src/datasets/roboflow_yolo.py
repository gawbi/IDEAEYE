"""Roboflow YOLO(v5/v8/v11) export 정리·분석 (CLAUDE.md 3.2 보조 데이터).

폴더 규칙: data/roboflow/<project-slug>/{train,valid,test}/{images,labels}, data.yaml, README.*.txt
- data.yaml 은 Roboflow 원본(../train/images 상대경로가 어긋남)을 data.roboflow.yaml 로 보관하고,
  프로젝트 루트 기준 path 를 가진 data.yaml 로 다시 쓴다.
- Roboflow 파일명 `<source>_jpg.rf.<hash>.jpg` 에서 `<source>` 가 원본 이미지 ID. 증강본은 같은 source 를 공유하므로
  재분할 시 source 단위 group split 이 필요하다. 분할 간 source 중복(누수)을 검사한다.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_RF_RE = re.compile(r"^(?P<source>.+?)(?:_(?:jpg|jpeg|png|bmp))?\.rf\.[0-9a-f]{32}$", re.IGNORECASE)
_LONG_PREFIX = "\\\\?\\"
# 서로 다른 원본이 이름만 같을 수 있는 "일반 이름": 숫자만, download(1), images-3-, image, img_12, untitled, photo, screenshot...
_GENERIC_RE = re.compile(
    r"^(?:\d+|download(?:[\s_-]*\(?\d*\)?)?|images?[\s_-]*\(?\d*\)?[\s_-]*|img[\s_-]*\d*|untitled(?:[\s_-]*\w*)?|photo[\s_-]*\d*|screenshot.*|unnamed[\s_-]*\d*)$",
    re.IGNORECASE,
)


def is_generic_name(source: str) -> bool:
    return bool(_GENERIC_RE.match(source.strip()))


def long_path(p: str | Path) -> Path:
    """Windows MAX_PATH(260자) 회피용 접두사를 붙인다. Roboflow 파일명 + 한글 프로젝트 경로가 쉽게 260자를 넘는다."""
    p = Path(p)
    if os.name != "nt":
        return p
    s = str(p.resolve()) if not str(p).startswith(_LONG_PREFIX) else str(p)
    if s.startswith(_LONG_PREFIX):
        return Path(s)
    if s.startswith("\\\\"):
        return Path(_LONG_PREFIX + "UNC\\" + s[2:])
    return Path(_LONG_PREFIX + s)


def short_path(p: str | Path) -> Path:
    """표시·상대경로 계산용으로 접두사를 제거한다."""
    s = str(p)
    if s.startswith(_LONG_PREFIX + "UNC\\"):
        return Path("\\\\" + s[len(_LONG_PREFIX) + 4 :])
    if s.startswith(_LONG_PREFIX):
        return Path(s[len(_LONG_PREFIX) :])
    return Path(s)


def source_id(filename: str | Path) -> str:
    """Roboflow 파일명에서 원본 이미지 ID 를 추출한다. 패턴이 아니면 stem 그대로."""
    stem = Path(filename).stem
    m = _RF_RE.match(stem)
    return m.group("source") if m else stem


@dataclass(frozen=True)
class YoloLabel:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    fmt: str  # "bbox" | "polygon"
    n_points: int = 0


def read_yolo_labels(path: Path) -> list[YoloLabel]:
    """YOLO detect(cls cx cy w h) 와 segment(cls x1 y1 x2 y2 ...) 형식을 모두 읽는다. 폴리곤은 외접 bbox 로 환산."""
    rows: list[YoloLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            rows.append(YoloLabel(cls, *vals, fmt="bbox"))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            rows.append(YoloLabel(cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, fmt="polygon", n_points=len(xs)))
    return rows


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    missing_labels: int = 0
    empty_labels: int = 0
    instances: dict[str, int] = field(default_factory=dict)
    sources: int = 0
    sizes: dict[str, int] = field(default_factory=dict)  # "WxH" → count
    boxes_per_image: dict[str, float] = field(default_factory=dict)  # min/mean/max
    box_rel_size: dict[str, float] = field(default_factory=dict)  # bbox 변의 상대 길이 (mean)
    label_format: str = "none"  # bbox | polygon | mixed | none


def analyze_split(split_dir: Path, names: list[str], max_size_probe: int | None = None) -> tuple[SplitStats, set[str]]:
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    st = SplitStats()
    sources: set[str] = set()
    if not img_dir.exists():
        return st, sources
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    st.images = len(images)
    inst: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    n_boxes: list[int] = []
    rel_w: list[float] = []
    rel_h: list[float] = []
    fmts: set[str] = set()
    for i, img in enumerate(images):
        sources.add(source_id(img))
        if max_size_probe is None or i < max_size_probe:
            try:
                with Image.open(img) as im:
                    sizes[f"{im.width}x{im.height}"] += 1
            except OSError:
                sizes["unreadable"] += 1
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            st.missing_labels += 1
            n_boxes.append(0)
            continue
        st.labels += 1
        rows = read_yolo_labels(lbl)
        if not rows:
            st.empty_labels += 1
        n_boxes.append(len(rows))
        for lb in rows:
            inst[names[lb.cls] if 0 <= lb.cls < len(names) else f"cls{lb.cls}"] += 1
            rel_w.append(lb.w)
            rel_h.append(lb.h)
            fmts.add(lb.fmt)
    st.instances = dict(sorted(inst.items(), key=lambda kv: -kv[1]))
    st.label_format = "none" if not fmts else ("mixed" if len(fmts) > 1 else next(iter(fmts)))
    st.sources = len(sources)
    st.sizes = dict(sizes.most_common(10))
    if n_boxes:
        st.boxes_per_image = {"min": min(n_boxes), "mean": round(sum(n_boxes) / len(n_boxes), 2), "max": max(n_boxes)}
    if rel_w:
        st.box_rel_size = {"w_mean": round(sum(rel_w) / len(rel_w), 4), "h_mean": round(sum(rel_h) / len(rel_h), 4)}
    return st, sources


def parse_readme(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    rf = root / "README.roboflow.txt"
    if rf.exists():
        text = rf.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"exported via roboflow\.com on (.+?) GMT", text)
        if m:
            info["exported"] = m.group(1).strip()
        m = re.search(r"includes (\d+) images", text)
        if m:
            info["readme_images"] = int(m.group(1))
        section = None
        for line in text.splitlines():
            if "pre-processing was applied" in line:
                section = "preprocessing"
                info[section] = []
            elif "augmentation was applied" in line:
                section = "augmentation"
                info[section] = []
                m = re.search(r"create (\d+) versions", line)
                if m:
                    info["augmentation_versions"] = int(m.group(1))
            elif line.startswith("* ") and section:
                info[section].append(line[2:].strip())
            elif line.strip() == "" and section and info.get(section):
                section = None
    ds = root / "README.dataset.txt"
    if ds.exists():
        text = ds.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"License:\s*(.+)", text)
        if m:
            info["license"] = m.group(1).strip()
        m = re.search(r"(https://universe\.roboflow\.com/\S+)", text)
        if m:
            info["url"] = m.group(1)
    return info


def load_data_yaml(root: Path) -> dict[str, Any]:
    orig = root / "data.roboflow.yaml"
    src = orig if orig.exists() else root / "data.yaml"
    with open(src, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_fixed_data_yaml(root: Path, project_root: Path) -> Path:
    """Roboflow 의 ../train/images 상대경로를 프로젝트 루트 기준 path 로 고쳐 쓴다. 원본은 data.roboflow.yaml 로 보관."""
    orig = root / "data.roboflow.yaml"
    if not orig.exists():
        (root / "data.yaml").rename(orig)
    d = load_data_yaml(root)
    try:
        rel = short_path(root.resolve()).relative_to(short_path(Path(project_root).resolve()))
        path_value = rel.as_posix()
    except ValueError:
        path_value = str(short_path(root.resolve()))
    fixed: dict[str, Any] = {"path": path_value}
    for key, sub in SPLIT_DIRS.items():
        if (root / sub / "images").exists():
            fixed[key] = f"{sub}/images"
    fixed["nc"] = d.get("nc", len(d.get("names", [])))
    fixed["names"] = d.get("names", [])
    if "roboflow" in d:
        fixed["roboflow"] = d["roboflow"]
    fixed["_note"] = "generated by src/datasets/roboflow_yolo.py — original: data.roboflow.yaml"
    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(fixed, f, allow_unicode=True, sort_keys=False)
    return root / "data.yaml"


def analyze_dataset(root: Path, max_size_probe: int | None = None) -> dict[str, Any]:
    root = Path(root)
    d = load_data_yaml(root)
    names = list(d.get("names", []))
    rf = d.get("roboflow", {})
    summary: dict[str, Any] = {
        "slug": root.name,
        "root": str(short_path(root)),
        "workspace": rf.get("workspace"),
        "project": rf.get("project"),
        "version": rf.get("version"),
        "license": rf.get("license"),
        "url": rf.get("url"),
        "classes": names,
        "readme": parse_readme(root),
        "splits": {},
    }
    src_sets: dict[str, set[str]] = {}
    for key, sub in SPLIT_DIRS.items():
        st, srcs = analyze_split(root / sub, names, max_size_probe)
        if st.images:
            summary["splits"][key] = asdict(st)
            src_sets[key] = srcs
    keys = list(src_sets)
    leaks = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            common = src_sets[keys[i]] & src_sets[keys[j]]
            if common:
                # 숫자·download·images-3- 같은 일반 이름은 서로 다른 원본이 이름만 같을 수 있어 확정 불가 → 별도 집계
                ambiguous = {c for c in common if is_generic_name(c)}
                leaks[f"{keys[i]}∩{keys[j]}"] = {
                    "count": len(common),
                    "ambiguous_generic_names": len(ambiguous),
                    "confirmed_by_name": sorted(common - ambiguous)[:10],
                    "examples": sorted(common)[:5],
                }
    summary["source_leak"] = leaks
    summary["totals"] = {
        "images": sum(s["images"] for s in summary["splits"].values()),
        "sources": len(set().union(*src_sets.values())) if src_sets else 0,
        "instances": dict(sum((Counter(s["instances"]) for s in summary["splits"].values()), Counter())),
    }
    summary["analyzed"] = date.today().isoformat()
    return summary


def summary_markdown(s: dict[str, Any]) -> str:
    r = s.get("readme", {})
    lines = [f"## {s['slug']}", ""]
    lines.append(f"- 출처: {s.get('url') or r.get('url')} (v{s.get('version')}, {s.get('license') or r.get('license')})")
    lines.append(f"- 클래스: {', '.join(s['classes'])}")
    fmts = sorted({st.get("label_format", "none") for st in s["splits"].values()})
    lines.append(f"- 라벨 형식: {', '.join(fmts)} (polygon 이면 YOLO segment 형식, detect 학습 시 bbox 로 자동 환산됨)")
    lines.append(f"- 이미지 {s['totals']['images']}장 / 원본(source) {s['totals']['sources']}장")
    if r.get("preprocessing"):
        lines.append(f"- 전처리: {'; '.join(r['preprocessing'])}")
    if r.get("augmentation"):
        lines.append(f"- 증강(x{r.get('augmentation_versions', '?')}): {'; '.join(r['augmentation'])}")
    if not s["source_leak"]:
        lines.append("- 분할 간 source 중복: 없음")
    for pair, lk in s["source_leak"].items():
        conf = lk.get("confirmed_by_name", [])
        lines.append(
            f"- 분할 간 source 이름 중복 ({pair}): {lk['count']}건 — 일반 이름(동명이인 가능) {lk.get('ambiguous_generic_names', 0)}건, "
            f"이름으로 확정 {len(conf)}건{(' 예: ' + ', '.join(conf[:3])) if conf else ''}. 재분할 시 같은 이름은 같은 그룹으로 묶을 것(보수적)"
        )
    lines += ["", "| split | images | sources | labels | missing | empty | " + " | ".join(s["classes"]) + " | boxes/img (min/mean/max) | sizes |", "|---|---:|---:|---:|---:|---:|" + "---:|" * len(s["classes"]) + "---|---|"]
    for name, st in s["splits"].items():
        inst = " | ".join(str(st["instances"].get(c, 0)) for c in s["classes"])
        bpi = st.get("boxes_per_image", {})
        sizes = ", ".join(f"{k}×{v}" for k, v in list(st["sizes"].items())[:3])
        lines.append(f"| {name} | {st['images']} | {st['sources']} | {st['labels']} | {st['missing_labels']} | {st['empty_labels']} | {inst} | {bpi.get('min')}/{bpi.get('mean')}/{bpi.get('max')} | {sizes} |")
    return "\n".join(lines) + "\n"


def update_manifest(roboflow_root: Path, summary: dict[str, Any]) -> Path:
    path = roboflow_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if isinstance(manifest, list):  # download_roboflow.py 가 만든 리스트 형식과 호환
        manifest = {m.get("project", f"item{i}"): m for i, m in enumerate(manifest)}
    manifest[summary["slug"]] = summary
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_datasets_doc(roboflow_root: Path, doc_path: Path) -> Path:
    path = roboflow_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    lines = [
        "# 데이터셋 인벤토리 (보조 공개 데이터)",
        "",
        "`scripts/ingest_roboflow_zip.py` 가 자동 생성. **여기서 나온 수치는 KPI 로 쓰지 않는다** (CLAUDE.md 3.2).",
        "재분할 시에는 반드시 `source` 단위 group split 을 쓴다 (증강본이 같은 source 를 공유).",
        "",
    ]
    for slug in sorted(manifest):
        s = manifest[slug]
        if "splits" in s:
            lines.append(summary_markdown(s))
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    return doc_path


def main(argv: list[str] | None = None) -> int:
    """이미 풀린 데이터셋 폴더를 재분석한다: python -m src.datasets.roboflow_yolo data/roboflow/<slug>"""
    import argparse
    import sys

    from ..common.config import PROJECT_ROOT, load_config, resolve_path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--doc", default="docs/datasets.md")
    args = ap.parse_args(argv)

    root = long_path(resolve_path(args.dataset_dir))
    rf_root = long_path(resolve_path(load_config("data")["paths"]["roboflow_root"]))
    write_fixed_data_yaml(root, PROJECT_ROOT)
    summary = analyze_dataset(root)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_manifest(rf_root, summary)
    doc = write_datasets_doc(rf_root, resolve_path(args.doc))
    print(summary_markdown(summary))
    print(f"→ {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
