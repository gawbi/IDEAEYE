"""Roboflow Universe 보조 데이터 다운로드 (CLAUDE.md 3.2).

사용법:
  1. https://app.roboflow.com → Settings → API Keys 에서 Private API Key 복사 (무료 계정 가능)
  2. PowerShell:  $env:ROBOFLOW_API_KEY = "여기에_키"
  3. python scripts/download_roboflow.py            # configs/roboflow.yaml 의 전체 목록
     python scripts/download_roboflow.py --only egg-crack-detection-eu65a

주의: 이 데이터는 sanity check / 사전학습 용도. 여기서 나온 수치는 KPI 로 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="roboflow")
    ap.add_argument("--only", nargs="*", default=None, help="project 슬러그로 필터")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key and not args.dry_run:
        print("ROBOFLOW_API_KEY 환경변수가 없습니다. 파일 상단 사용법을 참고해 키를 설정하세요.")
        return 2

    out_root = resolve_path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    targets = [d for d in cfg["datasets"] if not args.only or d["project"] in args.only]

    if args.dry_run:
        for d in targets:
            print(f"[dry-run] {d['workspace']}/{d['project']} v{d.get('version') or 'latest'} → {out_root / d['project']}")
        return 0

    from roboflow import Roboflow  # 지연 import: 키 없이 --dry-run 가능

    rf = Roboflow(api_key=api_key)
    manifest = []
    for d in targets:
        dest = out_root / d["project"]
        if dest.exists() and any(dest.iterdir()):
            print(f"skip (exists): {dest}")
            continue
        try:
            project = rf.workspace(d["workspace"]).project(d["project"])
            version = d.get("version")
            if version is None:
                versions = project.versions()
                if not versions:
                    print(f"no versions: {d['project']}")
                    continue
                version = max(int(v.version.split("/")[-1]) for v in versions)
            ds = project.version(version).download(cfg["export_format"], location=str(dest))
            info = {"workspace": d["workspace"], "project": d["project"], "version": version, "location": ds.location, "note": d.get("note")}
            print(f"downloaded: {info}")
            manifest.append(info)
        except Exception as e:  # 개별 실패는 건너뛰고 계속
            print(f"FAILED {d['workspace']}/{d['project']}: {e}")
            manifest.append({"project": d["project"], "error": str(e)})

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
