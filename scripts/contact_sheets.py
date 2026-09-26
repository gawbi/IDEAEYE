"""클래스 코드별 크롭 격자 이미지 (사람이 코드→이름 가설을 눈으로 검증하는 용도).

python scripts/contact_sheets.py [--per-class 20] [--cols 5]  → outputs/review/<light>_<code>.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.train.classifier import imread, letterbox  # noqa: E402


def hypothesis_name(light: str, code: str, hyp: dict) -> str:
    st, det = code.split("/")
    h = hyp.get(light, {})
    s = h.get("state", {}).get(st, {})
    d = h.get("state_detail", {}).get(code, {}) if isinstance(h.get("state_detail"), dict) else {}
    name = s.get("name", "?") if isinstance(s, dict) else "?"
    if isinstance(d, dict) and d.get("name"):
        name += " / " + str(d["name"])
    return name


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    hyp = load_config("class_codes")
    idx = pd.read_csv(resolve_path("data/crops/index.csv"), dtype={"cls": str, "state": str, "state_detail": str}, encoding="utf-8-sig")
    idx = idx[idx["exists"] == True]  # noqa: E712
    out = resolve_path("outputs/review")
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 18) if Path(r"C:\Windows\Fonts\malgun.ttf").exists() else None
    lines = ["# 코드별 크롭 격자 (outputs/review)", "", "| light | code | 가설 이름 | n(전체) | 파일 |", "|---|---|---|---:|---|"]
    for (light, code), g in idx.groupby(["light", "cls"]):
        sample = g.sample(n=min(args.per_class, len(g)), random_state=args.seed)
        tiles = []
        for p in sample["crop_path"]:
            img = imread(p)
            tiles.append(letterbox(img, args.tile) if img is not None else np.zeros((args.tile, args.tile, 3), np.uint8))
        rows = (len(tiles) + args.cols - 1) // args.cols
        while len(tiles) < rows * args.cols:
            tiles.append(np.zeros((args.tile, args.tile, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * args.cols : (r + 1) * args.cols]) for r in range(rows)])
        header = np.full((40, grid.shape[1], 3), 30, np.uint8)
        canvas = np.vstack([header, grid])
        title = f"{light}  code {code}  가설: {hypothesis_name(light, code, hyp)}  (n={len(g):,})"
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((10, 8), title, fill=(255, 255, 0), font=font)
        fname = f"{light}_{code.replace('/', '-')}.jpg"
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        buf.tofile(str(out / fname))
        lines.append(f"| {light} | {code} | {hypothesis_name(light, code, hyp)} | {len(g):,} | {fname} |")
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
