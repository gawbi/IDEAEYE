"""Phase 2: 판 이미지 → 계란 크롭 (zip 에서 직접 읽음, 프로세스 병렬).

출력: data/crops/<light>/<tray_id>_<seq>.jpg + data/crops/index.csv (크롭 경로 + 라벨 코드 + 분할 + 메타)
재실행 시 이미 있는 크롭은 건너뛴다.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ..common.config import resolve_path
from .audit import ImageSource, image_source_from_config
from .coords import Box

_SOURCE: ImageSource | None = None


def _init_worker(data_cfg: dict[str, Any], root: str | None) -> None:
    global _SOURCE
    _SOURCE = image_source_from_config(data_cfg, Path(root) if root else None)


def crop_egg(img: np.ndarray, box: Box, margin: float, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.expanded(margin).clipped(w, h).to_int()
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    ch, cw = crop.shape[:2]
    scale = long_side / max(ch, cw)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))), interpolation=cv2.INTER_AREA)
    return crop


def _process(task: dict[str, Any]) -> list[dict[str, Any]]:
    """task: {image_file, out_dir, tray_id, boxes: [(seq, x1, y1, x2, y2)], margin, long_side, quality}"""
    assert _SOURCE is not None
    img = _SOURCE.get(task["image_file"])
    rows: list[dict[str, Any]] = []
    if img is None:
        return [{"image_file": task["image_file"], "seq": s, "crop_path": None, "error": "image not found"} for s, *_ in task["boxes"]]
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq, x1, y1, x2, y2 in task["boxes"]:
        path = out_dir / f"{task['tray_id']}_{int(seq):02d}.jpg"
        crop = crop_egg(img, Box(x1, y1, x2, y2), task["margin"], task["long_side"])
        if crop.size == 0:
            rows.append({"image_file": task["image_file"], "seq": seq, "crop_path": None, "error": "empty crop"})
            continue
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, task["quality"]])
        if not ok:
            rows.append({"image_file": task["image_file"], "seq": seq, "crop_path": None, "error": "encode failed"})
            continue
        buf.tofile(str(path))
        rows.append({"image_file": task["image_file"], "seq": seq, "crop_path": str(path), "crop_w": crop.shape[1], "crop_h": crop.shape[0], "error": None})
    return rows


def build_tasks(df: pd.DataFrame, out_root: Path, margin: float, long_side: int, quality: int, skip_existing: bool = True) -> list[dict[str, Any]]:
    tasks = []
    for (light, image_file, tray_id), g in df.groupby(["light", "image_file", "tray_id"], sort=False):
        out_dir = out_root / light
        boxes = [(int(r.seq) if pd.notna(r.seq) else i + 1, float(r.x1), float(r.y1), float(r.x2), float(r.y2)) for i, r in enumerate(g.itertuples())]
        if skip_existing and all((out_dir / f"{tray_id}_{s:02d}.jpg").exists() for s, *_ in boxes):
            continue
        tasks.append({"image_file": image_file, "out_dir": str(out_dir), "tray_id": tray_id, "boxes": boxes, "margin": margin, "long_side": long_side, "quality": quality})
    return tasks


def run_crops(
    data_cfg: dict[str, Any],
    crops_cfg: dict[str, Any],
    df: pd.DataFrame,
    split_df: pd.DataFrame | None = None,
    root: Path | None = None,
    workers: int | None = None,
    limit: int | None = None,
    progress_every: int = 5000,
) -> pd.DataFrame:
    """df: 객체 단위 레코드(records.pkl). 반환: index DataFrame (크롭 경로 + 라벨 + 분할)."""
    out_root = resolve_path(crops_cfg["out_root"])
    lights = crops_cfg.get("lights") or sorted(df["light"].unique())
    sub = df[df["light"].isin(lights)].copy()
    if limit:
        keep = sub.drop_duplicates(["light", "image_file"]).head(limit)[["light", "image_file"]]
        sub = sub.merge(keep, on=["light", "image_file"])
    if "seq" not in sub:
        sub["seq"] = sub.groupby(["light", "image_file"]).cumcount() + 1
    sub["seq"] = sub["seq"].fillna(sub.groupby(["light", "image_file"]).cumcount() + 1).astype(int)

    tasks = build_tasks(sub, out_root, crops_cfg["margin"], crops_cfg["long_side"], crops_cfg["jpeg_quality"])
    workers = workers or crops_cfg.get("workers") or max(1, (os.cpu_count() or 2) - 1)
    print(f"crops: {len(sub)} objects in {sub['image_file'].nunique()} images, {len(tasks)} images to process, workers={workers}")

    results: list[dict[str, Any]] = []
    if tasks:
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(data_cfg, str(root) if root else None)) as ex:
                for i, rows in enumerate(ex.map(_process, tasks, chunksize=16), 1):
                    results.extend(rows)
                    if i % progress_every == 0:
                        print(f"  {i}/{len(tasks)} images", flush=True)
        else:
            _init_worker(data_cfg, str(root) if root else None)
            for i, t in enumerate(tasks, 1):
                results.extend(_process(t))
                if i % progress_every == 0:
                    print(f"  {i}/{len(tasks)} images", flush=True)

    # index: 모든 객체(이번에 만든 것 + 이미 있던 것)
    sub["crop_path"] = [str(out_root / l / f"{t}_{int(s):02d}.jpg") for l, t, s in zip(sub["light"], sub["tray_id"], sub["seq"])]
    errors = pd.DataFrame([r for r in results if r.get("error")])
    if not errors.empty:
        print(f"  errors: {len(errors)} (예: {errors.iloc[0].to_dict()})")
    sub["cls"] = sub["state"].astype(str) + "/" + sub["state_detail"].astype(str)
    cols = ["light", "tray_id", "seq", "object_id", "state", "state_detail", "cls", "date", "region_code", "washed", "breed", "crop_path", "x1", "y1", "x2", "y2"]
    index = sub[[c for c in cols if c in sub]].copy()
    if split_df is not None:
        index = index.merge(split_df[["light", "tray_id", "split"]], on=["light", "tray_id"], how="left")
    index["exists"] = [os.path.exists(p) for p in index["crop_path"]]
    index_path = resolve_path(crops_cfg["index_csv"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    print(f"index: {index_path}  crops present: {int(index['exists'].sum())}/{len(index)}")
    return index
