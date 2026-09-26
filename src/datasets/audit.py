"""Phase 1 데이터 감사 (CLAUDE.md 3.1 / PROMPTS.md Phase 1).

1. bbox 시각화 N장 → outputs/audit/bbox_*.jpg (top_left / bottom_left 두 해석을 나란히 그림)
   + 밝기 기반 수치 근거(origin_evidence)로 좌표계 판단을 보조
2. MONO/COLOR 별 state / state_detail 개수표 + 공식 수치 대비 차이
3. 판·촬영일·지역코드·세척여부·품종 분포표
4. MONO↔COLOR 매칭 키 후보 탐색

실행: python -m src.datasets.audit --data configs/data.yaml --out outputs/audit --n-vis 20
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ..common.config import load_config, load_yaml, resolve_path
from .aihub_xml import AIHubXMLParser, TrayAnnotation, flatten, parser_from_config
from .coords import Box, to_top_left

# ---------------------------------------------------------------- I/O (한글 경로 안전)


def imread(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if not ok:
        raise IOError(f"encode failed: {path}")
    buf.tofile(str(path))


def build_image_index(root: Path, exts: list[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            index.setdefault(p.name, p)
    return index


def locate_image(ann: TrayAnnotation, index: dict[str, Path]) -> Path | None:
    cand = ann.xml_path.parent / ann.image_file
    if cand.exists():
        return cand
    return index.get(Path(ann.image_file).name)


class ImageSource:
    """판 이미지를 폴더 또는 zip(풀지 않음)에서 파일명으로 찾아 읽는다. 92GB 이미지 zip 을 풀지 않기 위한 장치."""

    def __init__(self, dirs: list[Path] | None = None, zips: list[Path] | None = None, exts: list[str] | None = None):
        import zipfile

        self.exts = [e.lower() for e in (exts or [".jpg", ".jpeg", ".png"])]
        self.dir_index: dict[str, Path] = {}
        for d in dirs or []:
            self.dir_index.update(build_image_index(Path(d), self.exts))
        self.zip_index: dict[str, tuple[Path, str]] = {}
        self._zips: dict[Path, zipfile.ZipFile] = {}
        for z in zips or []:
            z = Path(z)
            zf = zipfile.ZipFile(z)
            self._zips[z] = zf
            for member in zf.namelist():
                if Path(member).suffix.lower() in self.exts:
                    self.zip_index.setdefault(Path(member).name, (z, member))

    def __len__(self) -> int:
        return len(self.dir_index) + len(self.zip_index)

    def has(self, name: str) -> bool:
        name = Path(name).name
        return name in self.dir_index or name in self.zip_index

    def get_bytes(self, name: str) -> bytes | None:
        """원본 파일 바이트 (재인코딩 없이 복사할 때)."""
        name = Path(name).name
        if name in self.dir_index:
            return self.dir_index[name].read_bytes()
        if name in self.zip_index:
            z, member = self.zip_index[name]
            return self._zips[z].read(member)
        return None

    def get(self, name: str) -> np.ndarray | None:
        data = self.get_bytes(name)
        if data is None:
            return None
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def image_source_from_config(data_cfg: dict[str, Any], root: Path | None = None) -> ImageSource:
    mode = data_cfg.get("image_source", "both")
    dirs = [root or resolve_path(data_cfg["paths"]["aihub_root"])] if mode in ("dir", "both") else []
    zips: list[Path] = []
    if mode in ("zip", "both") and data_cfg["paths"].get("aihub_zip_root"):
        zroot = Path(data_cfg["paths"]["aihub_zip_root"])
        if zroot.exists():
            zips = sorted(zroot.glob(data_cfg.get("image_zip_glob", "**/*.zip")))
    return ImageSource(dirs, zips, data_cfg["image_exts"])


# ---------------------------------------------------------------- 1. 수집


def _parse_one(parser: AIHubXMLParser, path: str) -> TrayAnnotation:
    return parser.parse(path)


def collect(parser: AIHubXMLParser, root: Path, workers: int | None = None) -> tuple[list[TrayAnnotation], pd.DataFrame]:
    """root 아래 XML 전체 파싱. 파일이 많으면 프로세스 병렬 (137k 파일 기준 단일 프로세스 18분 → 수 분)."""
    files = sorted(str(p) for p in Path(root).rglob("*.xml"))
    if workers is None:
        import os

        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers > 1 and len(files) > 2000:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        with ProcessPoolExecutor(max_workers=workers) as ex:
            anns = list(ex.map(partial(_parse_one, parser), files, chunksize=256))
    else:
        anns = [parser.parse(p) for p in files]
    rows = [r for a in anns for r in flatten(a)]
    return anns, pd.DataFrame(rows)


def save_records(df: pd.DataFrame, path: Path) -> Path:
    """평탄화 레코드 캐시 (Phase 2 분할·크롭에서 재파싱 없이 사용)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return path


def load_records(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


# ---------------------------------------------------------------- 2. 개수표 / 공식 수치 비교


def count_table(df: pd.DataFrame, field: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    t = pd.crosstab(df[field], df["light"])
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False)


def compare_official(df: pd.DataFrame, official: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for light, spec in official.items():
        sub = df[df["light"] == light] if not df.empty else df
        n_trays = int(sub.drop_duplicates("tray_id").shape[0]) if not sub.empty else 0
        if spec.get("total_trays") is not None:
            rows.append({"light": light, "level": "total_trays", "class": "*", "official": spec["total_trays"], "actual": n_trays})
        rows.append({"light": light, "level": "total_objects", "class": "*", "official": spec.get("total_objects"), "actual": int(len(sub))})
        for level in ("state", "state_detail"):
            for cls, n in (spec.get(level) or {}).items():
                actual = int((sub[level].astype(str) == str(cls)).sum()) if not sub.empty else 0
                rows.append({"light": light, "level": level, "class": cls, "official": n, "actual": actual})
    out = pd.DataFrame(rows)
    out["diff"] = out["actual"] - out["official"].fillna(0).astype(int)
    out["ratio"] = (out["actual"] / out["official"].replace(0, np.nan)).round(3)
    out["match"] = out["diff"] == 0
    return out


def share_table(df: pd.DataFrame, field: str = "state") -> pd.DataFrame:
    """광원별 코드 분포(개수·비율%). 코드표 없이도 공식 비율표와 대조해 코드→이름 추정의 근거로 쓴다."""
    if df.empty:
        return pd.DataFrame()
    t = pd.crosstab(df[field].astype(str), df["light"])
    share = (t / t.sum(axis=0) * 100).round(1)
    share.columns = [f"{c}_%" for c in share.columns]
    out = pd.concat([t, share], axis=1)
    return out.sort_index(key=lambda s: s.map(lambda v: (0, int(v)) if str(v).lstrip("-").isdigit() else (1, str(v))))


# ---------------------------------------------------------------- 3. 분포표


def distribution_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    out: dict[str, pd.DataFrame] = {}
    per_tray = df.groupby(["light", "tray_id"]).size().rename("objects")
    out["tray"] = per_tray.groupby("light").describe()
    trays = df.drop_duplicates(["light", "tray_id"])
    for key in ("date", "region_code", "washed", "breed", "meta_farming_env", "meta_category", "meta_classification", "meta_fresh_day"):
        if key in df:
            t = pd.crosstab(trays[key].fillna("NA").astype(str), trays["light"])
            t["total"] = t.sum(axis=1)
            out[key] = t.sort_index()
    if "meta_days" in df:
        d = pd.to_numeric(trays["meta_days"], errors="coerce")
        out["days"] = d.groupby(trays["light"]).describe()
    return out


# ---------------------------------------------------------------- 4. MONO↔COLOR 매칭 키


def _strip_light(s: str) -> str:
    return re.sub(r"(?i)[_\-]?(MONO|COLOR)[_\-]?", "_", str(s)).strip("_")


def matching_keys(df: pd.DataFrame) -> dict[str, Any]:
    """MONO 와 COLOR 판이 어떤 키로 짝지어지는지 후보별로 교집합 크기를 센다."""
    if df.empty or df["light"].nunique() < 2:
        return {"note": "MONO/COLOR 둘 다 있어야 함", "candidates": {}}
    trays = df.drop_duplicates(["light", "tray_id"])
    mono = trays[trays["light"] == "MONO"]
    color = trays[trays["light"] == "COLOR"]
    cands: dict[str, dict[str, int]] = {}

    def score(name: str, f) -> None:
        m, c = set(map(f, mono.to_dict("records"))), set(map(f, color.to_dict("records")))
        cands[name] = {"mono": len(m), "color": len(c), "matched": len(m & c)}

    score("tray_id (원문)", lambda r: r["tray_id"])
    score("tray_id (MONO/COLOR 토큰 제거)", lambda r: _strip_light(r["tray_id"]))
    score("date+region_code+datetime", lambda r: (r.get("date"), r.get("region_code"), r.get("meta_datetime")))
    score("date+region_code", lambda r: (r.get("date"), r.get("region_code")))
    # 계란 객체 ID 가 광원 간에 공유되는지 (같은 계란을 MONO/COLOR 로 각각 찍었다면 ID 가 같을 수 있음)
    if "object_id" in df:
        ids = {l: set(df.loc[df["light"] == l, "object_id"].dropna().astype(str)) for l in ("MONO", "COLOR")}
        cands["object_id (계란 단위)"] = {"mono": len(ids["MONO"]), "color": len(ids["COLOR"]), "matched": len(ids["MONO"] & ids["COLOR"])}
    best = max(cands, key=lambda k: cands[k]["matched"])
    return {"candidates": cands, "best": best, "best_matched": cands[best]["matched"]}


# ---------------------------------------------------------------- 1. 좌표계 시각화 + 수치 근거


def _boxes_under(ann: TrayAnnotation, origin: str) -> list[Box]:
    return [to_top_left(o.raw["x_min"], o.raw["y_min"], o.raw["x_max"], o.raw["y_max"], ann.height, origin) for o in ann.objects]


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """배경 중앙값 대비 편차가 큰 픽셀 = 전경(계란). 투광/반사광 모두에서 동작하도록 절대편차를 쓴다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dev = cv2.absdiff(gray, np.full_like(gray, int(np.median(gray))))
    _, mask = cv2.threshold(dev, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def origin_evidence(ann: TrayAnnotation, img: np.ndarray) -> dict[str, float]:
    """두 해석 각각에 대해 (bbox 내부 전경 비율 - 외부 전경 비율)을 계산. 큰 쪽이 맞을 가능성이 높다."""
    fg = foreground_mask(img)
    H, W = fg.shape
    out: dict[str, float] = {}
    for origin in ("top_left", "bottom_left"):
        inside = np.zeros_like(fg)
        oob = 0
        for b in _boxes_under(ann, origin):
            if not b.is_inside(W, H):
                oob += 1
            x1, y1, x2, y2 = b.clipped(W, H).to_int()
            inside[y1:y2, x1:x2] = True
        if inside.any() and (~inside).any():
            out[origin] = float(fg[inside].mean() - fg[~inside].mean())
        else:
            out[origin] = 0.0
        out[f"{origin}_out_of_bounds"] = oob
    out["prefer"] = "top_left" if out["top_left"] >= out["bottom_left"] else "bottom_left"
    out["margin"] = abs(out["top_left"] - out["bottom_left"])
    return out


_FONT_CANDIDATES = [r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]


def _put_labels(img_bgr: np.ndarray, labels: list[tuple[str, int, int, tuple[int, int, int]]]) -> np.ndarray:
    """한글 라벨은 OpenCV 가 못 그리므로 PIL 로 그린다. 폰트가 없으면 OpenCV 로 대체(한글 깨짐)."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for f in _FONT_CANDIDATES:
        if Path(f).exists():
            font = ImageFont.truetype(f, 13)
            break
    if font is None:
        for text, x, y, color in labels:
            cv2.putText(img_bgr, text, (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, x, y, color in labels:
        d.text((x, max(0, y - 14)), text, fill=color[::-1], font=font)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


_PALETTE = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0), (128, 0, 128), (0, 128, 255), (203, 192, 255)]


def _state_color(state: str, seen: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    if state not in seen:
        seen[state] = _PALETTE[len(seen) % len(_PALETTE)]
    return seen[state]


def draw_two_panels(ann: TrayAnnotation, img: np.ndarray) -> np.ndarray:
    panels = []
    seen: dict[str, tuple[int, int, int]] = {}
    for origin in ("top_left", "bottom_left"):
        p = img.copy()
        labels = []
        for o, b in zip(ann.objects, _boxes_under(ann, origin)):
            x1, y1, x2, y2 = b.clipped(img.shape[1], img.shape[0]).to_int()
            color = _state_color(str(o.state), seen)
            cv2.rectangle(p, (x1, y1), (x2, y2), color, 2)
            cv2.circle(p, (int(o.raw["x_min"]), int(o.raw["y_min"])), 5, (255, 0, 0), -1)  # 원본 (x_min,y_min) 위치
            seq = o.raw.get("seq")
            text = f"#{seq} s{o.state}/{o.state_detail}" if seq else f"s{o.state}/{o.state_detail}"
            labels.append((text, x1, y1, color))
        p = _put_labels(p, labels)
        cv2.putText(p, f"origin={origin}  {ann.light} {ann.tray_id}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        panels.append(p)
    sep = np.full((img.shape[0], 8, 3), 128, dtype=np.uint8)
    return np.hstack([panels[0], sep, panels[1]])


def visualize_bboxes(anns: list[TrayAnnotation], source: "ImageSource | dict[str, Path]", out_dir: Path, n: int, seed: int) -> pd.DataFrame:
    """광원별로 고르게 n 장을 뽑아 두 해석을 나란히 그린다. source 는 ImageSource 또는 파일명→경로 dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    by_light: dict[str, list[TrayAnnotation]] = {}
    for a in anns:
        if a.objects:
            by_light.setdefault(a.light, []).append(a)
    for lst in by_light.values():
        rng.shuffle(lst)
    sample: list[TrayAnnotation] = []
    i = 0
    while len(sample) < n and any(i < len(v) for v in by_light.values()):
        for lst in by_light.values():
            if i < len(lst) and len(sample) < n:
                sample.append(lst[i])
        i += 1
    rows = []
    for a in sample:
        if isinstance(source, ImageSource):
            img = source.get(a.image_file)
        else:
            path = locate_image(a, source)
            img = imread(path) if path else None
        if img is None:
            rows.append({"tray_id": a.tray_id, "light": a.light, "image_found": False})
            continue
        if (img.shape[1], img.shape[0]) != (a.width, a.height) and a.width and a.height:
            size_note = f"xml={a.width}x{a.height} img={img.shape[1]}x{img.shape[0]}"
        else:
            size_note = "ok"
        ev = origin_evidence(a, img)
        imwrite(out_dir / f"bbox_{a.light}_{a.tray_id}.jpg", draw_two_panels(a, img))
        rows.append({"tray_id": a.tray_id, "light": a.light, "image_found": True, "size_check": size_note, **ev})
    return pd.DataFrame(rows)


def summarize_origin(ev: pd.DataFrame) -> dict[str, Any]:
    if ev.empty or "prefer" not in ev:
        return {"verdict": "undetermined", "reason": "no images"}
    ok = ev[ev["image_found"]]
    votes = ok["prefer"].value_counts().to_dict()
    verdict = max(votes, key=votes.get) if votes else "undetermined"
    return {
        "verdict": verdict,
        "votes": votes,
        "mean_margin": float(ok["margin"].mean()) if len(ok) else 0.0,
        "out_of_bounds": {o: int(ok[f"{o}_out_of_bounds"].sum()) for o in ("top_left", "bottom_left")},
        "note": "수치 근거는 보조 지표. outputs/audit/bbox_*.jpg 를 눈으로 확인해 configs/data.yaml coords.verified 를 갱신할 것",
    }


# ---------------------------------------------------------------- 실행


def run_audit(data_cfg: dict[str, Any], official: dict[str, Any], out_dir: Path, n_vis: int = 20, seed: int = 0, root: Path | None = None, workers: int | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = root or resolve_path(data_cfg["paths"]["aihub_root"])
    parser = parser_from_config(data_cfg)
    t0 = time.perf_counter()
    anns, df = collect(parser, root, workers=workers)
    parse_s = time.perf_counter() - t0
    if anns:
        save_records(df, out_dir / "records.pkl")
    source = image_source_from_config(data_cfg, root)

    report: dict[str, Any] = {"root": str(root), "n_xml": len(anns), "n_objects": int(len(df)), "parse_seconds": round(parse_s, 1), "images_available": len(source)}
    if not anns:
        report["error"] = f"XML 파일이 없음: {root}"
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    report["per_light"] = {l: {"trays": int(g["tray_id"].nunique()), "objects": int(len(g))} for l, g in df.groupby("light")}

    # 1. 좌표계
    ev = visualize_bboxes(anns, source, out_dir, n_vis, seed)
    ev.to_csv(out_dir / "origin_evidence.csv", index=False, encoding="utf-8-sig")
    report["origin"] = summarize_origin(ev)
    report["origin"]["configured"] = data_cfg["coords"]["origin"]

    # 2. 개수표 (코드 기준) + 공식 수치 비교 + 비율표
    tables = {"state": count_table(df, "state"), "state_detail": count_table(df, "state_detail")}
    tables["state_share"] = share_table(df, "state")
    tables["state_detail_share"] = share_table(df, "state_detail")
    tables["state_x_detail"] = pd.crosstab([df["light"], df["state"].astype(str)], df["state_detail"].astype(str))
    cmp = compare_official(df, official)
    cmp.to_csv(out_dir / "official_vs_actual.csv", index=False, encoding="utf-8-sig")
    report["official_mismatch"] = cmp[~cmp["match"]].to_dict("records")
    report["official_state_share"] = {l: spec.get("state_share") for l, spec in official.items() if spec.get("state_share")}

    # 3. 분포표
    dist = distribution_tables(df)
    tables.update(dist)
    for name, t in tables.items():
        t.to_csv(out_dir / f"table_{name}.csv", encoding="utf-8-sig")

    # 4. 매칭 키
    report["matching"] = matching_keys(df)

    # 5. bbox 크기 통계 (크롭 해상도 결정용)
    bw, bh = (df["x2"] - df["x1"]), (df["y2"] - df["y1"])
    report["bbox_px"] = {l: {"w_median": float(bw[df["light"] == l].median()), "h_median": float(bh[df["light"] == l].median()),
                             "w_p5": float(bw[df["light"] == l].quantile(0.05)), "h_p95": float(bh[df["light"] == l].quantile(0.95))}
                         for l in df["light"].unique()}

    # 보고서
    md = [f"# 데이터 감사 보고 ({root})", "", f"- XML: {len(anns)}  객체: {len(df)}  (파싱 {parse_s:.0f}s)", f"- 광원별: {report['per_light']}", ""]
    md += ["## 1. 좌표계", "```json", json.dumps(report["origin"], ensure_ascii=False, indent=2), "```", ""]
    md += ["## 2. 개수표 vs 공식 수치", cmp.to_markdown(index=False), ""]
    md += ["공식 대분류 비율(%):", "```json", json.dumps(report["official_state_share"], ensure_ascii=False), "```", ""]
    for name, t in tables.items():
        md += [f"## 표: {name}", t.to_markdown(), ""]
    md += ["## 4. MONO↔COLOR 매칭 키", "```json", json.dumps(report["matching"], ensure_ascii=False, indent=2), "```", ""]
    md += ["## 5. bbox 픽셀 크기", "```json", json.dumps(report["bbox_px"], ensure_ascii=False, indent=2), "```"]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI Hub 계란 데이터 감사")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--official", default="configs/aihub_official_counts.yaml")
    ap.add_argument("--root", default=None, help="aihub_root 재정의 (가짜 데이터 테스트용)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-vis", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="XML 파싱 프로세스 수 (기본: CPU-1)")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data_cfg = load_yaml(args.data)
    official = load_yaml(args.official)
    out_dir = resolve_path(args.out or data_cfg["paths"]["audit_out"])
    root = resolve_path(args.root) if args.root else None
    report = run_audit(data_cfg, official, out_dir, args.n_vis, args.seed, root, workers=args.workers)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n→ {out_dir / 'report.md'}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
