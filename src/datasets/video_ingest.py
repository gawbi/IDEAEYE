"""기기 촬영 영상 → 계란 크롭 데이터셋 (파각란검출기 시제품 영상용).

단계: (1) 검은 여백 제거·화면(카메라) 분리 → (2) 화면별 고정 ROI 자동 검출: 투광 계란의 빛(고휘도·고채도)을 시간 누적한 지도에서
밝은 자리 k 개 → (3) 화면별 프레임차 기반 정지 구간 분할(회전 사이의 정지 = 1 시점) → (4) 구간별 대표 프레임을 ROI 로 크롭,
ROI 안의 계란 유무 판정 → index.csv + 검수용 오버레이·모션 곡선.
ROI 는 configs/video_ingest.yaml 의 roi_override 로 수동 지정 가능.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Pane:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int  # 원본 프레임 좌표


@dataclass
class Roi:
    pane: int
    index: int
    x1: int
    y1: int
    x2: int
    y2: int  # 화면(pane) 내부 좌표


@dataclass
class Segment:
    pane: int
    index: int
    start_frame: int
    end_frame: int
    rep_frame: int
    start_sec: float
    end_sec: float
    motion_mean: float


# ---------------------------------------------------------------- 1. 화면 분리


def content_bbox(frames: list[np.ndarray], threshold: int) -> tuple[int, int, int, int]:
    """여러 프레임의 최대 밝기로 검은 여백을 제외한 내용 영역을 찾는다."""
    mx = np.max(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]), axis=0)
    rows = np.where(mx.max(axis=1) > threshold)[0]
    cols = np.where(mx.max(axis=0) > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        h, w = mx.shape
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def split_panes(bbox: tuple[int, int, int, int], n: int, layout: str) -> list[Pane]:
    x1, y1, x2, y2 = bbox
    panes = []
    for i in range(n):
        if layout == "vertical":
            h = (y2 - y1) / n
            panes.append(Pane(i, x1, int(round(y1 + i * h)), x2, int(round(y1 + (i + 1) * h))))
        else:
            w = (x2 - x1) / n
            panes.append(Pane(i, int(round(x1 + i * w)), y1, int(round(x1 + (i + 1) * w)), y2))
    return panes


# ---------------------------------------------------------------- 2. 계란 빛 마스크 / ROI 검출


def glow_mask(bgr: np.ndarray, v_min: int = 170, s_min: int = 60, v_white: int | None = None) -> np.ndarray:
    """투광 계란 영역: 밝고 채도 높은 주황/황색. v_white 를 주면 아주 밝은 흰색도 포함(흰 기계 부품이 섞일 수 있어 기본 비활성)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (v >= v_min) & (s >= s_min) & ((h <= 35) | (h >= 160))
    if v_white is not None:
        m = m | (v >= v_white)
    return m.astype(np.uint8)


def detect_rois(glow_map: np.ndarray, pane_index: int, k: int, margin: float, min_area_frac: float) -> list[Roi]:
    """glow_map: 시간 누적 계란 빛 지도(0~1). 상위 분위수 임계 → 연결성분 → 면적 상위 k, 좌→우 정렬."""
    g = cv2.GaussianBlur(glow_map.astype(np.float32), (0, 0), max(1.0, 0.01 * min(glow_map.shape)))
    if g.max() <= 0:
        return []
    thr = max(0.05, float(np.percentile(g[g > 0], 40))) if (g > 0).any() else 0.05
    m = (g >= thr).astype(np.uint8) * 255
    kk = max(3, int(0.02 * min(g.shape)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = g.shape
    comps = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * H * W]
    comps.sort(reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    for _, i in comps[:k]:
        boxes.append((stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]))
    if 0 < len(boxes) < k:  # 인접한 계란이 한 덩어리로 붙은 경우: x 좌표 k-means 로 분할
        ys, xs = np.where(m > 0)
        if len(xs) > k:
            pts = xs.astype(np.float32).reshape(-1, 1)
            _, labels, centers = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
            boxes = []
            for c in range(k):
                sel = labels.ravel() == c
                if sel.sum() < min_area_frac * H * W * 0.3:
                    continue
                x0, x1 = int(xs[sel].min()), int(xs[sel].max())
                y0, y1 = int(ys[sel].min()), int(ys[sel].max())
                boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
    rois = []
    for x, y, w, h in boxes:
        mx, my = int(w * margin), int(h * margin)
        rois.append((x, Roi(pane_index, 0, max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my))))
    rois.sort(key=lambda t: t[0])
    out = []
    for j, (_, r) in enumerate(rois):
        r.index = j
        out.append(r)
    return out


def egg_present(crop: np.ndarray, min_frac: float = 0.12) -> tuple[bool, float]:
    frac = float(glow_mask(crop).mean()) if crop.size else 0.0
    return frac >= min_frac, frac


# ---------------------------------------------------------------- 3. 정지 구간


def static_segments(motion: np.ndarray, frame_ids: np.ndarray, fps: float, threshold: float, min_sec: float, pane: int = 0) -> list[Segment]:
    """motion[i] = 분석 프레임 i 와 i-1 의 평균 절대차. 연속으로 threshold 미만인 구간을 정지 구간으로 본다."""
    static = motion < threshold
    segs: list[Segment] = []
    i = 0
    while i < len(static):
        if not static[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(static) and static[j + 1]:
            j += 1
        f0, f1 = int(frame_ids[i]), int(frame_ids[j])
        if (f1 - f0 + 1) / fps >= min_sec:
            rep = int(frame_ids[(i + j) // 2])
            segs.append(Segment(pane, len(segs), f0, f1, rep, f0 / fps, f1 / fps, float(motion[i : j + 1].mean())))
        i = j + 1
    return segs


# ---------------------------------------------------------------- 메인


def analyze_video(video: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride, scale = int(cfg["analysis_stride"]), float(cfg["analysis_scale"])

    # 1패스: 축소 프레임으로 내용 영역, 계란 빛 누적, 화면별 모션
    sample_frames: list[np.ndarray] = []
    ids: list[int] = []
    smalls: list[np.ndarray] = []
    count = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        idx = count
        count += 1
        if idx % stride:
            continue
        s = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        smalls.append(s)
        ids.append(idx)
        if len(sample_frames) < 40 and idx % (stride * 25) == 0:
            sample_frames.append(s)
    cap.release()
    if not ids:
        raise IOError("no frames")

    bbox_s = content_bbox(sample_frames or smalls[:1], cfg["content_threshold"])
    inv = 1.0 / scale
    bbox = tuple(int(round(v * inv)) for v in bbox_s)
    panes = split_panes(bbox, cfg["panes"], cfg["pane_layout"])
    panes_s = split_panes(bbox_s, cfg["panes"], cfg["pane_layout"])

    # 화면별 계란 빛 누적 지도와 모션
    glow_maps = [np.zeros((p.y2 - p.y1, p.x2 - p.x1), np.float32) for p in panes_s]
    motions: list[list[float]] = [[] for _ in panes_s]
    prev_gray: list[np.ndarray | None] = [None] * len(panes_s)
    for s in smalls:
        for pi, p in enumerate(panes_s):
            sub = s[p.y1 : p.y2, p.x1 : p.x2]
            glow_maps[pi] += glow_mask(sub, cfg.get("glow_v_min", 170), cfg.get("glow_s_min", 60), cfg.get("glow_v_white"))
            g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
            motions[pi].append(float(np.abs(g - prev_gray[pi]).mean()) if prev_gray[pi] is not None else 0.0)
            prev_gray[pi] = g
    glow_maps = [m / len(smalls) for m in glow_maps]

    rois: list[Roi] = []
    if cfg.get("roi_override"):
        for j, (p, x1, y1, x2, y2) in enumerate(cfg["roi_override"]):
            rois.append(Roi(int(p), j, int(x1), int(y1), int(x2), int(y2)))
    else:
        for p, gm in zip(panes, glow_maps):
            for r in detect_rois(gm, p.index, cfg["rois_per_pane"], cfg["roi_margin"], cfg["roi_min_area_frac"]):
                rois.append(Roi(p.index, r.index, int(r.x1 * inv), int(r.y1 * inv), int(r.x2 * inv), int(r.y2 * inv)))

    segments: list[Segment] = []
    for pi in range(len(panes)):
        segs = static_segments(np.array(motions[pi]), np.array(ids), fps, cfg["motion_threshold"], cfg["min_static_sec"], pane=pi)
        if cfg.get("max_segments"):
            segs = segs[: cfg["max_segments"]]
        segments += segs
    return {"video": str(video), "fps": fps, "frames": n, "content_bbox": bbox, "panes": [asdict(p) for p in panes], "rois": [asdict(r) for r in rois],
            "segments": [asdict(s) for s in segments], "motion": {"ids": ids, "values": motions},
            "glow_maps": [(m * 255).astype(np.uint8) for m in glow_maps]}


def draw_overlay(frame: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = info["content_bbox"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
    for p in info["panes"]:
        cv2.rectangle(out, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 255), 2)
        cv2.putText(out, f"cam{p['index']}", (p["x1"] + 8, p["y1"] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        for r in info["rois"]:
            if r["pane"] == p["index"]:
                cv2.rectangle(out, (p["x1"] + r["x1"], p["y1"] + r["y1"]), (p["x1"] + r["x2"], p["y1"] + r["y2"]), (0, 0, 255), 3)
                cv2.putText(out, f"roi{r['index']}", (p["x1"] + r["x1"] + 5, p["y1"] + r["y1"] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out


def extract_crops(video: Path, info: dict[str, Any], out_dir: Path, quality: int = 95, keep_empty: bool = False, present_min_frac: float = 0.12) -> list[dict[str, Any]]:
    """화면별 정지 구간의 대표 프레임에서 ROI 크롭 저장 (계란 없는 크롭은 keep_empty 가 아니면 제외). 반환: index 행."""
    cap = cv2.VideoCapture(str(video))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    stem = video.stem.replace(" ", "_")
    best_overlay: tuple[float, np.ndarray] | None = None
    panes = {p["index"]: p for p in info["panes"]}
    for s in info["segments"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["rep_frame"])
        ok, f = cap.read()
        if not ok:
            continue
        p = panes[s["pane"]]
        for r in info["rois"]:
            if r["pane"] != p["index"]:
                continue
            crop = f[p["y1"] + r["y1"] : p["y1"] + r["y2"], p["x1"] + r["x1"] : p["x1"] + r["x2"]]
            if crop.size == 0:
                continue
            present, frac = egg_present(crop, present_min_frac)
            if best_overlay is None or frac > best_overlay[0]:
                best_overlay = (frac, f)
            if not present and not keep_empty:
                continue
            name = f"{stem}_cam{p['index']}_roi{r['index']}_seg{s['index']:04d}.jpg"
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tofile(str(out_dir / name))
            rows.append({"video": video.name, "cam": p["index"], "roi": r["index"], "segment": s["index"], "rep_frame": s["rep_frame"],
                         "start_sec": round(s["start_sec"], 3), "end_sec": round(s["end_sec"], 3), "motion_mean": round(s["motion_mean"], 3),
                         "egg_present": present, "glow_frac": round(frac, 3), "crop_path": str(out_dir / name), "crop_w": crop.shape[1], "crop_h": crop.shape[0],
                         "label": "", "severity": "", "shell_color": "", "stamp": ""})
    cap.release()
    if best_overlay is not None:
        cv2.imencode(".jpg", draw_overlay(best_overlay[1], info), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(out_dir / f"{stem}_overlay.jpg"))
    return rows


def ingest(video: Path, cfg: dict[str, Any], out_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = analyze_video(video, cfg)
    out_dir = out_root / video.stem.replace(" ", "_")
    rows = extract_crops(video, info, out_dir, cfg.get("jpeg_quality", 95), cfg.get("keep_empty", False), cfg.get("present_min_frac", 0.12))
    for i, gm in enumerate(info.pop("glow_maps")):
        cv2.imencode(".png", gm)[1].tofile(str(out_dir / f"glow_cam{i}.png"))
    (out_dir / "analysis.json").write_text(json.dumps({k: v for k, v in info.items() if k != "motion"}, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["video", "cam", "roi", "segment", "rep_frame", "start_sec", "end_sec", "motion_mean", "egg_present", "glow_frac", "crop_path", "crop_w", "crop_h", "label", "severity", "shell_color", "stamp"]
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    try:  # 모션 곡선 (검수용)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array(info["motion"]["ids"]) / info["fps"]
        fig, axes = plt.subplots(len(info["panes"]), 1, figsize=(12, 2.6 * len(info["panes"])), sharex=True)
        axes = np.atleast_1d(axes)
        for pi, ax in enumerate(axes):
            ax.plot(t, info["motion"]["values"][pi], lw=0.8)
            ax.axhline(cfg["motion_threshold"], color="r", ls="--", lw=0.8)
            for s in info["segments"]:
                if s["pane"] == pi:
                    ax.axvspan(s["start_sec"], s["end_sec"], color="g", alpha=0.15)
            ax.set_ylabel(f"cam{pi} |diff|")
        axes[-1].set_xlabel("sec")
        axes[0].set_title(f"{video.name}: static segments={len(info['segments'])}, crops={len(rows)}")
        plt.tight_layout()
        plt.savefig(out_dir / "motion.png", dpi=110)
        plt.close()
    except Exception:
        pass
    return info, rows
