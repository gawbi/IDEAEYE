"""미검출(FN) vs 검출 크롭 비교 격자 (사람 육안 검토용).

python scripts/fn_contact_sheet.py --code 2/7 --out "검수/06_미검출_파형크랙"
입력: outputs/fn_analysis/<code>/ (fn_analysis.py 가 복사한 FN 크롭), outputs/fn_analysis/crack_test_with_stats.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import resolve_path  # noqa: E402
from src.train.classifier import imread, letterbox  # noqa: E402


def grid(paths: list[str], title: str, out: Path, cols: int = 8, tile: int = 200) -> int:
    tiles = []
    for p in paths:
        img = imread(p)
        tiles.append(letterbox(img, tile) if img is not None else np.zeros((tile, tile, 3), np.uint8))
    n = len(tiles)
    rows = max(1, (n + cols - 1) // cols)
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    g = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    canvas = np.vstack([np.full((40, g.shape[1], 3), 30, np.uint8), g])
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else None
    ImageDraw.Draw(pil).text((10, 8), title, fill=(255, 255, 0), font=font)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(str(out))
    return n


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="2/7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    code_dir = args.code.replace("/", "_")
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fn_paths = sorted(glob.glob(str(resolve_path(f"outputs/fn_analysis/{code_dir}/*.jpg"))))[: args.n]
    n_fn = grid(fn_paths, f"MONO {args.code} FN (misses at crack-recall-95% operating point): {len(fn_paths)} crops", out / f"FN_{code_dir}_grid.jpg")

    df = pd.read_csv(resolve_path("outputs/fn_analysis/crack_test_with_stats.csv"), dtype={"cls": str}, encoding="utf-8-sig")
    det = df[(df["cls"] == args.code) & (df["detected"])]
    det = det.sample(n=min(args.n, len(det)), random_state=1)
    n_det = grid(det["crop_path"].tolist(), f"MONO {args.code} detected (for comparison): {len(det)} crops", out / f"detected_{code_dir}_grid.jpg")
    print(f"FN {n_fn}, detected {n_det} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
