"""experiments/ 의 결과(summary.json / eval.json / benchmark.json / latency.json)를 모아 experiments/RESULTS.md 표로 정리.

python scripts/collect_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import PROJECT_ROOT  # noqa: E402

TARGETS = {"2/6": "잠재파형크랙", "2/8": "미세크랙", "3/1": "혈반육반", "3/5": "먹통", "2/2": "일자형크랙?", "2/7": "파형크랙?"}


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = PROJECT_ROOT / "experiments"
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith("_smoke") and "smoke" not in d.name)
    lines = ["# 실험 결과 모음 (자동 생성: scripts/collect_results.py)", "", "코드→이름은 configs/class_codes.yaml 의 가설. 모든 수치는 AI Hub test 분할(다운로드본 내 10%) 기준.", ""]

    # A. 검출기
    lines += ["## A. 판 이미지 검출기 (YOLO11)", "", "| 실험 | mAP@0.5 | mAP@0.5:0.95 | 계란 위치 Recall(single-cls) | 추론 ms/img | 클래스별 AP50 |", "|---|---:|---:|---:|---:|---|"]
    for d in dirs:
        s = load(d / "summary.json")
        if s and "egg_localization_single_cls" in s:
            ap = ", ".join(f"{k}:{v['ap50']:.2f}" for k, v in s["per_class"].items())
            lines.append(f"| {d.name} | {s['overall']['mAP50']:.3f} | {s['overall']['mAP50-95']:.3f} | {s['egg_localization_single_cls']['recall']:.3f} | {s['speed_ms'].get('inference', 0):.1f} | {ap} |")
    lines += ["", "AI Hub 공식 DETR 베이스라인: MONO 0.53 / COLOR 0.54 (test mAP@0.5, 공식 test 는 비공개라 동일 조건 비교는 아님)", ""]

    # B. 분류기
    lines += ["## B. 크롭 분류기", "", "| 실험 | acc | macro recall | 정상 FPR @ 파각 Recall 90/95/98% | " + " | ".join(f"R({v})" for v in TARGETS.values()) + " | ms/img (b1/b16) |", "|---|---:|---:|---|" + "---:|" * len(TARGETS) + "---|"]
    for d in dirs:
        e = load(d / "eval.json")
        if e and "overall" in e and "accuracy" in e:
            ops = e.get("operating_points", {})
            fpr = "/".join(f"{ops[k]['fpr']:.3f}" for k in ("recall@0.90", "recall@0.95", "recall@0.98") if k in ops) or "-"
            rec = " | ".join(f"{e['overall'][k]['recall']:.3f}" if k in e["overall"] else "-" for k in TARGETS)
            lat = load(d / "latency.json") or {}
            ms = f"{lat.get('batch1', {}).get('ms_per_image', 0):.1f}/{lat.get('batch16', {}).get('ms_per_image', 0):.2f}" if lat else "-"
            lines.append(f"| {d.name} | {e['accuracy']:.4f} | {e['macro_recall']:.4f} | {fpr} | {rec} | {ms} |")
    lines += [""]

    # B'. 세척/품종/지역별 (분류기)
    lines += ["## B'. 세척여부·품종·지역별 macro recall (분류기)", ""]
    for d in dirs:
        e = load(d / "eval.json")
        if e and e.get("by"):
            for axis, table in e["by"].items():
                cells = []
                for val, m in table.items():
                    recs = [x["recall"] for x in m.values() if x["support"] > 0]
                    cells.append(f"{axis}={val}: {sum(recs) / len(recs):.3f} (n={sum(x['support'] for x in m.values())})")
                lines.append(f"- {d.name} — " + "; ".join(cells))
    lines += [""]

    # C. PatchCore / 임계값 베이스라인
    lines += ["## C. 이상탐지 · 임계값 베이스라인", "", "| 실험 | AUROC | 정상 FPR @ Recall 90/95/98% | 소분류 Recall @95% |", "|---|---:|---|---|"]
    for d in dirs:
        e = load(d / "eval.json")
        if e and "auroc" in e:
            ops = e["operating_points"]
            fpr = "/".join(f"{ops[k]['fpr']:.3f}" for k in ("recall@0.90", "recall@0.95", "recall@0.98") if k in ops)
            sub = ", ".join(f"{k}:{v:.2f}" for k, v in ops.get("recall@0.95", {}).get("subclass_recall", {}).items())
            lines.append(f"| {d.name} | {e['auroc']:.3f} | {fpr} | {sub} |")
        if e and "features" in e:
            for f, r in e["features"].items():
                ops = r["operating_points_vs_normal"]
                fpr = "/".join(f"{ops[k]['fpr']:.3f}" for k in ("recall@0.80", "recall@0.88", "recall@0.95") if k in ops)
                lines.append(f"| {d.name} [{f}] | {r['auroc_blood_vs_normal']:.3f} (vs 먹통 {r['auroc_blood_vs_dark']:.3f}) | {fpr} (Recall 80/88/95) | - |")
    lines += [""]

    # D. 벤치마크
    lines += ["## D. 지연 벤치마크", ""]
    for d in dirs:
        b = load(d / "benchmark.json")
        if b:
            st = b["stage_ms"]
            lines.append(f"- {d.name}: 계란 1개당 p50 {b['per_egg_ms'].get('p50', 0):.1f} ms / 판 총 p95 {st['total'].get('p95', 0):.1f} ms (detect p50 {st['detect'].get('p50', 0):.1f}, classify p50 {st['classify'].get('p50', 0):.1f}, 판당 계란 {b['eggs_per_image_mean']:.1f}) — {b['device']}, {b['classifier_providers']}")
    out = root / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
