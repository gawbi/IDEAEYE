"""기기 영상 반입: python scripts/ingest_video.py <영상 또는 폴더> [--config video_ingest] [--out data/own/frames]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.datasets.video_ingest import ingest  # noqa: E402

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--config", default="video_ingest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = resolve_path(args.out or cfg["out_root"])
    src = Path(args.source)
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS) if src.is_dir() else [src]
    total = 0
    for v in videos:
        info, rows = ingest(v, cfg, out_root)
        total += len(rows)
        print(f"{v.name}: fps {info['fps']:.0f}, frames {info['frames']}, panes {len(info['panes'])}, rois {len(info['rois'])}, static segments {len(info['segments'])}, crops {len(rows)}")
    print(f"→ {out_root}  (crops {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
