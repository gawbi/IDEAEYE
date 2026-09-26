"""Phase 3-A: 판 이미지 + bbox → Ultralytics YOLO 검출 데이터셋 (서브샘플, zip 에서 바이트 복사)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..common.config import load_config, resolve_path
from .audit import ImageSource, image_source_from_config

_SOURCE: ImageSource | None = None


def _init_worker(data_cfg: dict[str, Any], root: str | None) -> None:
    global _SOURCE
    _SOURCE = image_source_from_config(data_cfg, Path(root) if root else None)


def _copy_one(task: tuple[str, str, str, str]) -> tuple[str, bool]:
    image_file, out_image, label_text, out_label = task
    assert _SOURCE is not None
    data = _SOURCE.get_bytes(image_file)
    if data is None:
        return image_file, False
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    Path(out_label).parent.mkdir(parents=True, exist_ok=True)
    with open(out_image, "wb") as f:
        f.write(data)
    with open(out_label, "w", encoding="utf-8") as f:
        f.write(label_text)
    return image_file, True


def class_names_for(light: str, codes: list[str]) -> dict[int, str]:
    """코드 → 'state<code>' (+ 가설 이름). 코드표 확정 전이라 코드가 기본 이름."""
    try:
        hyp = load_config("class_codes").get(light, {}).get("state", {})
    except FileNotFoundError:
        hyp = {}
    names = {}
    for i, c in enumerate(codes):
        h = hyp.get(str(c), {}).get("name") if isinstance(hyp.get(str(c)), dict) else None
        names[i] = f"s{c}_{h}" if h else f"s{c}"
    return names


def export_yolo(
    data_cfg: dict[str, Any],
    det_cfg: dict[str, Any],
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    light: str,
    root: Path | None = None,
    workers: int | None = None,
    counts: dict[str, int] | None = None,
) -> Path:
    out_root = Path(det_cfg["out_root"]) / light
    seed = det_cfg.get("seed", 42)
    field = det_cfg.get("class_field", "state")
    counts = counts or det_cfg["per_light"][light]

    trays = split_df[split_df["light"] == light]
    chosen = []
    for split, n in counts.items():
        sub = trays[trays["split"] == split]
        chosen.append(sub.sample(n=min(n, len(sub)), random_state=seed) if n else sub)
    chosen_df = pd.concat(chosen)
    obj = df[df["light"] == light].merge(chosen_df[["light", "tray_id", "split"]], on=["light", "tray_id"])

    codes = sorted(obj[field].astype(str).unique(), key=lambda v: (len(v), v))
    idx = {c: i for i, c in enumerate(codes)}
    names = class_names_for(light, codes)

    tasks = []
    for (image_file, split), g in obj.groupby(["image_file", "split"]):
        W, H = float(g["width"].iloc[0]), float(g["height"].iloc[0])
        lines = []
        for r in g.itertuples():
            x1, y1, x2, y2 = max(0.0, r.x1), max(0.0, r.y1), min(W, r.x2), min(H, r.y2)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{idx[str(getattr(r, field))]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        out_image = out_root / "images" / split / Path(image_file).name
        out_label = out_root / "labels" / split / (Path(image_file).stem + ".txt")
        if out_image.exists() and out_label.exists():
            continue
        tasks.append((image_file, str(out_image), "\n".join(lines) + ("\n" if lines else ""), str(out_label)))

    workers = workers or det_cfg.get("workers") or max(1, (os.cpu_count() or 2) // 2)
    print(f"[{light}] trays={len(chosen_df)} objects={len(obj)} classes={names} to_copy={len(tasks)} workers={workers}")
    ok = 0
    if tasks:
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(data_cfg, str(root) if root else None)) as ex:
                for _, success in ex.map(_copy_one, tasks, chunksize=32):
                    ok += success
        else:
            _init_worker(data_cfg, str(root) if root else None)
            for t in tasks:
                ok += _copy_one(t)[1]
        print(f"[{light}] copied {ok}/{len(tasks)}")

    data_yaml = {
        "path": str(out_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(codes),
        "names": [names[i] for i in range(len(codes))],
        "_codes": codes,
        "_note": "class = AI Hub state code (코드→이름은 configs/class_codes.yaml 가설)",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    chosen_df.to_csv(out_root / "trays.csv", index=False, encoding="utf-8-sig")
    return out_root / "data.yaml"
