"""Roboflow YOLO export zip 을 data/roboflow/<slug>/ 로 풀고 정리·분석한다.

사용법:
  python scripts/ingest_roboflow_zip.py "Cracked-Eggs.v8-10_april_balanced.yolov8.zip"
  python scripts/ingest_roboflow_zip.py some.zip --slug my-name --force

동작:
  1. zip → data/roboflow/_tmp_<name>/ 에 풀고 data.yaml 의 roboflow.project 로 slug 결정 (없으면 zip 이름에서 유도)
  2. data/roboflow/<slug>/ 로 이동 (이미 있으면 --force 필요)
  3. zip 파일을 data/roboflow/_zips/ 로 옮겨 보관
  4. data.yaml 경로 수정(원본은 data.roboflow.yaml), summary.json / manifest.json / docs/datasets.md 갱신
  5. 분할 간 source 이미지 중복(누수) 검사 결과 출력
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from src.datasets.roboflow_yolo import (  # noqa: E402
    analyze_dataset,
    long_path,
    short_path,
    summary_markdown,
    update_manifest,
    write_datasets_doc,
    write_fixed_data_yaml,
)


def slug_from_zip_name(name: str) -> str:
    base = re.split(r"\.v\d+", Path(name).name, maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--force", action="store_true", help="같은 slug 폴더가 있으면 지우고 다시 푼다")
    ap.add_argument("--keep-zip", action="store_true", help="zip 을 _zips/ 로 옮기지 않는다")
    ap.add_argument("--doc", default="docs/datasets.md")
    args = ap.parse_args()

    zip_path = resolve_path(args.zip)
    if not zip_path.exists():
        print(f"zip not found: {zip_path}")
        return 2
    rf_root = resolve_path(load_config("data")["paths"]["roboflow_root"])
    rf_root.mkdir(parents=True, exist_ok=True)
    rf_root = long_path(rf_root)  # Windows 260자 제한 회피 (한글 경로 + 긴 Roboflow 파일명)

    tmp = rf_root / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as z:
        longest = max((len(n) for n in z.namelist()), default=0)
        z.extractall(tmp)
    if longest + len(str(short_path(rf_root))) + 40 > 260:
        print("NOTE: 경로가 260자를 넘어 \\\\?\\ 접두사로 처리함. 다른 도구(탐색기·Ultralytics)에서 문제가 되면 "
              "Windows '긴 경로 사용(LongPathsEnabled)' 을 켜거나 프로젝트를 짧은 경로로 옮기세요.")
    # zip 안에 최상위 폴더 하나만 있으면 한 단계 벗긴다
    entries = [p for p in tmp.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for p in inner.iterdir():
            shutil.move(str(p), tmp / p.name)
        inner.rmdir()

    slug = args.slug
    if not slug and (tmp / "data.yaml").exists():
        import yaml

        d = yaml.safe_load((tmp / "data.yaml").read_text(encoding="utf-8")) or {}
        slug = (d.get("roboflow") or {}).get("project")
    slug = slug or slug_from_zip_name(zip_path.name)

    dest = rf_root / slug
    if dest.exists() and any(dest.iterdir()):
        if not args.force:
            shutil.rmtree(tmp)
            print(f"already exists: {dest}  (--force 로 덮어쓰기)")
            return 1
        shutil.rmtree(dest)
    tmp.rename(dest)

    if not args.keep_zip:
        archive = rf_root / "_zips"
        archive.mkdir(exist_ok=True)
        target = archive / zip_path.name
        if zip_path.resolve() != target.resolve():
            shutil.move(str(zip_path), target)
            print(f"zip → {target}")

    write_fixed_data_yaml(dest, PROJECT_ROOT)
    summary = analyze_dataset(dest)
    (dest / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_manifest(rf_root, summary)
    doc = write_datasets_doc(rf_root, resolve_path(args.doc))

    print(summary_markdown(summary))
    if summary["source_leak"]:
        print("WARN: 분할 간 source 이미지 중복 → 재분할 필요:", summary["source_leak"])
    print(f"→ {short_path(dest)}\n→ {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
