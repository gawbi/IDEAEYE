import cv2
import numpy as np
import pytest

from src.common.config import load_config
from src.datasets.video_ingest import content_bbox, detect_rois, ingest, split_panes, static_segments


def _make_video(path, fps=30, w=640, h=480, static_len=20, move_len=6, cycles=3):
    """검은 여백 + 세로 2화면 + 화면당 밝은 타원 3개. 정지 구간과 이동 구간이 번갈아 나온다."""
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened()
    x0, y0, x1, y1 = 80, 20, 560, 460  # 내용 영역
    ph = (y1 - y0) // 2
    rng = np.random.RandomState(0)
    frame_no = 0
    for c in range(cycles):
        for phase, length in (("static", static_len), ("move", move_len)):
            for k in range(length):
                f = np.zeros((h, w, 3), np.uint8)
                f[y0:y1, x0:x1] = 25
                for p in range(2):
                    py = y0 + p * ph
                    for j in range(3):
                        cx = x0 + 80 + j * 160 + (rng.randint(-15, 15) if phase == "move" else 0)
                        cy = py + ph // 2 + (rng.randint(-15, 15) if phase == "move" else 0)
                        cv2.ellipse(f, (cx, cy), (45, 60), 0, 0, 360, (60, 140, 230), -1)
                wr.write(f)
                frame_no += 1
    wr.release()
    return frame_no


def test_content_bbox_and_panes():
    f = np.zeros((100, 200, 3), np.uint8)
    f[10:90, 30:170] = 100
    bb = content_bbox([f], 12)
    assert bb == (30, 10, 170, 90)
    panes = split_panes(bb, 2, "vertical")
    assert panes[0].y1 == 10 and panes[1].y2 == 90 and panes[0].y2 == panes[1].y1


def test_detect_rois_finds_three_bright_blobs():
    from src.datasets.video_ingest import glow_mask

    img = np.full((200, 480, 3), 20, np.uint8)
    for j, cx in enumerate((80, 240, 400)):
        cv2.ellipse(img, (cx, 100), (45, 60), 0, 0, 360, (60, 140, 230), -1)
    gm = glow_mask(img).astype(np.float32)
    assert gm[100, 240] == 1 and gm[5, 5] == 0
    rois = detect_rois(gm, 0, 3, 0.1, 0.02)
    assert [r.index for r in rois] == [0, 1, 2]
    assert rois[0].x1 < rois[1].x1 < rois[2].x1
    assert rois[1].x1 <= 240 - 45 and rois[1].x2 >= 240 + 45


def test_static_segments():
    motion = np.array([0.5] * 10 + [8.0] * 3 + [0.4] * 10 + [9.0] * 2 + [0.3] * 2)
    ids = np.arange(len(motion)) * 2
    segs = static_segments(motion, ids, fps=30.0, threshold=2.5, min_sec=0.3)
    assert len(segs) == 2  # 마지막 2프레임 구간은 너무 짧아 제외
    assert segs[0].start_frame == 0 and segs[1].start_frame == 26


def test_ingest_synthetic_video(tmp_path):
    video = tmp_path / "sample.mp4"
    n = _make_video(video)
    cfg = {**load_config("video_ingest"), "analysis_stride": 1, "analysis_scale": 0.5, "min_static_sec": 0.3, "motion_threshold": 1.0}
    info, rows = ingest(video, cfg, tmp_path / "out")
    assert len(info["panes"]) == 2 and len(info["rois"]) == 6
    assert len(info["segments"]) == 3 * 2, [(s["pane"], s["start_sec"]) for s in info["segments"]]  # 화면별 3구간
    assert len(rows) == 3 * 6
    assert all(r["egg_present"] for r in rows)
    assert (tmp_path / "out" / "sample" / "index.csv").exists()
    assert (tmp_path / "out" / "sample" / "sample_overlay.jpg").exists()
    crop = cv2.imread(rows[0]["crop_path"])
    assert crop is not None and crop.shape[0] > 50
