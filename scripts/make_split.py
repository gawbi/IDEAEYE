"""Phase 2: 판 단위 group split 생성 + 누수 검사 + 소분류별 분할 개수표.

python scripts/make_split.py                # outputs/audit/records.pkl 캐시 사용 (없으면 XML 재파싱)
python scripts/make_split.py --root <dir>   # 다른 라벨 루트
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve_path  # noqa: E402
from src.datasets.aihub_xml import parser_from_config  # noqa: E402
from src.datasets.audit import collect, load_records, save_records  # noqa: E402
from src.datasets.split import SPLITS, assign_split, find_leaks  # noqa: E402


def load_records_df(data_cfg: dict, split_cfg: dict, root: Path | None) -> pd.DataFrame:
    cache = resolve_path(split_cfg.get("records_cache", "outputs/audit/records.pkl"))
    if root is None and cache.exists():
        print(f"records cache: {cache}")
        return load_records(cache)
    root = root or resolve_path(data_cfg["paths"]["aihub_root"])
    print(f"parsing XML under {root} ...")
    _, df = collect(parser_from_config(data_cfg), root)
    if root == resolve_path(data_cfg["paths"]["aihub_root"]):
        save_records(df, cache)
    return df


def split_trays(df: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    """판 단위 레코드에 split/group 을 배정. per_light 면 광원별로 각각 비율을 맞춘다."""
    keys = list(split_cfg["group_keys"])
    trays = df.drop_duplicates(["light", "tray_id"])[["light", "tray_id", "image_file", "date", "region_code", "washed", "breed"]].copy()
    trays["n_objects"] = df.groupby(["light", "tray_id"]).size().reindex(pd.MultiIndex.from_frame(trays[["light", "tray_id"]])).values
    parts = []
    groups = trays.groupby("light") if split_cfg.get("per_light", True) else [("ALL", trays)]
    for light, sub in groups:
        recs = assign_split(sub.to_dict("records"), keys, split_cfg["ratios"], split_cfg["seed"], mode=split_cfg.get("group_mode", "combined"))
        parts.append(pd.DataFrame(recs))
    out = pd.concat(parts, ignore_index=True)
    out["group"] = out["light"] + "|" + out["group"]
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    data_cfg, split_cfg = load_config("data"), load_config("split")
    df = load_records_df(data_cfg, split_cfg, resolve_path(args.root) if args.root else None)
    if df.empty:
        print("no records")
        return 1
    df["cls"] = df["state"].astype(str) + "/" + df["state_detail"].astype(str)

    trays = split_trays(df, split_cfg)

    # 누수 검사: 그룹·판은 보장, 촬영일·지역코드 단독은 정보용
    recs = trays.to_dict("records")
    hard = find_leaks(recs, ["group", "tray_id"])
    if hard:
        print("LEAK DETECTED (group/tray):", {k: len(v) for k, v in hard.items()})
        return 1
    soft = find_leaks(recs, list(split_cfg["group_keys"]))
    print("group/tray 누수: 없음 |", "단독 키 걸침(정보):", {k: len(v) for k, v in soft.items()} or "없음")

    # 판 비율 / 객체 비율 (광원별)
    obj = df.merge(trays[["light", "tray_id", "split", "group"]], on=["light", "tray_id"], how="left")
    assert obj["split"].notna().all()
    print("\n## 판 수 (광원 × 분할)")
    print(pd.crosstab(trays["light"], trays["split"]).reindex(columns=list(SPLITS)).to_markdown())
    print("\n## 객체 비율 % (광원 × 분할)")
    t = pd.crosstab(obj["light"], obj["split"], normalize="index").reindex(columns=list(SPLITS)) * 100
    print(t.round(1).to_markdown())

    # 소분류별 분할 개수표
    table = pd.crosstab([obj["light"], obj["cls"]], obj["split"]).reindex(columns=list(SPLITS), fill_value=0)
    table["total"] = table.sum(axis=1)
    out = resolve_path(split_cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    trays.drop(columns=["n_objects"]).to_csv(out, index=False, encoding="utf-8-sig")
    (out.parent / "split_class_table.md").write_text(
        "# 소분류(코드)별 분할 개수\n\n코드→이름 가설: configs/class_codes.yaml\n\n" + table.to_markdown(), encoding="utf-8"
    )
    print("\n## 소분류(코드)별 분할 개수 (상위 표는 data/splits/split_class_table.md)")

    # 목표 클래스 test 개수
    print("\n## 목표 클래스 test 개수")
    rows = []
    for light, targets in (split_cfg.get("target_classes") or {}).items():
        sub = obj[obj["light"] == light]
        for code, name in targets.items():
            if code.endswith("/*"):
                mask = sub["state"].astype(str) == code.split("/")[0]
            else:
                mask = sub["cls"] == code
            c = Counter(sub.loc[mask, "split"])
            rows.append({"light": light, "code": code, "name(가설)": name, "train": c["train"], "val": c["val"], "test": c["test"],
                         "warn": "TEST<%d" % split_cfg["min_test_per_class"] if c["test"] < split_cfg["min_test_per_class"] else ""})
    tdf = pd.DataFrame(rows)
    print(tdf.to_markdown(index=False))
    (out.parent / "target_class_counts.md").write_text("# 목표 클래스 분할 개수 (코드 가설)\n\n" + tdf.to_markdown(index=False), encoding="utf-8")

    # 분할별 촬영일 (COLOR 처럼 날짜가 적은 경우 확인용)
    print("\n## 분할별 촬영일 수 / 지역 수")
    g = trays.groupby(["light", "split"]).agg(dates=("date", "nunique"), regions=("region_code", "nunique"), trays=("tray_id", "size"))
    print(g.to_markdown())
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
