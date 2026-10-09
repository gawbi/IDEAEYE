"""키보드 라벨링 도구 (OpenCV 창). python scripts/label_crops.py data/own/frames/<영상>/index.csv [--start N] [--only-unlabeled]

키: n normal · c crack · u unsure · e empty · 1/2/3 심각도(crack) · w 흰색 · b 갈색 · s 도장 토글 · ←/→ (또는 a/d) 이동 · q 저장 종료
라벨은 입력 즉시 CSV 에 저장된다. 상단에 진행률·현재 라벨·AI Hub 참고 점수(p_crack_aihub)가 표시된다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.classifier import imread, letterbox  # noqa: E402

COLS = ["label", "severity", "shell_color", "stamp", "note"]


def render(row: pd.Series, i: int, n: int, size: int = 720) -> np.ndarray:
    img = imread(str(row["crop_path"]))
    if img is None:
        img = np.zeros((size, size, 3), np.uint8)
    view = letterbox(img, size)
    canvas = np.zeros((size + 120, size, 3), np.uint8)
    canvas[120:] = view
    lab = str(row.get("label", "") or "")
    color = {"crack": (60, 60, 255), "normal": (60, 220, 60), "unsure": (0, 200, 255), "empty": (160, 160, 160)}.get(lab, (255, 255, 255))
    lines = [
        f"[{i + 1}/{n}] cam{row.get('cam', '?')} roi{row.get('roi', '?')} seg{row.get('segment', '?')}  {Path(str(row['crop_path'])).name}",
        f"label: {lab or '-'}   severity: {row.get('severity', '') or '-'}   color: {row.get('shell_color', '') or '-'}   stamp: {row.get('stamp', '') or '-'}",
        f"aihub p_crack: {row.get('p_crack_aihub', '')}   note: {row.get('note', '') or ''}",
        "n normal  c crack  u unsure  e empty | 1/2/3 severity | w white  b brown | s stamp | a/d move | q quit",
    ]
    for k, t in enumerate(lines):
        cv2.putText(canvas, t, (10, 24 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color if k == 1 else (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--only-unlabeled", action="store_true")
    args = ap.parse_args()
    path = Path(args.index)
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    for c in COLS:
        if c not in df:
            df[c] = ""
    order = [i for i in range(len(df)) if not args.only_unlabeled or not df.at[i, "label"]]
    if not order:
        print("라벨할 항목이 없습니다.")
        return 0
    pos = max(0, min(args.start, len(order) - 1))

    def save() -> None:
        df.to_csv(path, index=False, encoding="utf-8-sig")

    win = "label_crops"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    while True:
        i = order[pos]
        cv2.imshow(win, render(df.iloc[i], pos, len(order)))
        k = cv2.waitKeyEx(0)
        key = k & 0xFF
        if key == ord("q") or k == 27:
            save()
            break
        elif key in (ord("d"), ord(" ")) or k == 2555904:  # →
            pos = min(pos + 1, len(order) - 1)
        elif key == ord("a") or k == 2424832:  # ←
            pos = max(pos - 1, 0)
        elif key == ord("n"):
            df.at[i, "label"] = "normal"; df.at[i, "severity"] = ""; save(); pos = min(pos + 1, len(order) - 1)
        elif key == ord("c"):
            df.at[i, "label"] = "crack"; save()
        elif key == ord("u"):
            df.at[i, "label"] = "unsure"; df.at[i, "severity"] = ""; save(); pos = min(pos + 1, len(order) - 1)
        elif key == ord("e"):
            df.at[i, "label"] = "empty"; df.at[i, "severity"] = ""; save(); pos = min(pos + 1, len(order) - 1)
        elif key in (ord("1"), ord("2"), ord("3")):
            df.at[i, "label"] = "crack"; df.at[i, "severity"] = chr(key); save(); pos = min(pos + 1, len(order) - 1)
        elif key == ord("w"):
            df.at[i, "shell_color"] = "white"; save()
        elif key == ord("b"):
            df.at[i, "shell_color"] = "brown"; save()
        elif key == ord("s"):
            df.at[i, "stamp"] = "n" if df.at[i, "stamp"] == "y" else "y"; save()
    cv2.destroyAllWindows()
    done = int((df["label"] != "").sum())
    print(f"saved {path}: labeled {done}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
