"""experiments/ 기록 체계 (CLAUDE.md 설계 원칙 6).

ExperimentTracker 는 실험 디렉터리를 만들고 config 사본·seed·git hash·환경정보를 저장한다.
log_metrics() 는 지표를 대분류·소분류·세척여부·난각색 등 축별로 JSON + Markdown 으로 저장하고
experiments/SUMMARY.md 에 한 줄을 추가한다. 재현 불가(git hash 없음) 실험은 SUMMARY 에 표시된다.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common.config import PROJECT_ROOT, resolve_path

NO_GIT = "no-git"


def git_hash(cwd: Path = PROJECT_ROOT) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return NO_GIT
        h = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5)
        return h + ("-dirty" if dirty.stdout.strip() else "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return NO_GIT


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch  # noqa: WPS433

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception:  # pragma: no cover
        pass
    return info


def metrics_to_markdown(metrics: dict[str, dict[str, float]], title: str) -> str:
    lines = [f"### {title}", "", "| class | precision | recall | f1 | support |", "|---|---:|---:|---:|---:|"]
    for cls, m in metrics.items():
        lines.append(f"| {cls} | {m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | {m.get('f1', 0):.4f} | {m.get('support', 0)} |")
    return "\n".join(lines) + "\n"


class ExperimentTracker:
    def __init__(self, name: str, config: dict[str, Any] | str | Path, seed: int, root: str | Path = "experiments", notes: str = ""):
        self.name = name
        self.seed = seed
        self.root = resolve_path(root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.root / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)

        if isinstance(config, (str, Path)):
            shutil.copy(resolve_path(config), self.dir / "config.yaml")
            config_src = str(config)
        else:
            import yaml

            with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
            config_src = "<dict>"

        self.meta = {
            "name": name,
            "created": stamp,
            "seed": seed,
            "git_hash": git_hash(),
            "config_source": config_src,
            "notes": notes,
            "env": _env_info(),
            "status": "running",
        }
        self._write_meta()

    def _write_meta(self) -> None:
        with open(self.dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @property
    def reproducible(self) -> bool:
        return self.meta["git_hash"] != NO_GIT and not self.meta["git_hash"].endswith("-dirty")

    def log_metrics(self, metrics: dict[str, Any], by: dict[str, dict[str, Any]] | None = None, headline: dict[str, float] | None = None) -> Path:
        """metrics: 전체 클래스별 지표. by: {'state': {...}, 'state_detail': {...}, 'washed': {...}, 'shell_color': {...}}.
        headline: SUMMARY.md 에 올릴 대표 수치 (예: {'crack_recall': 0.95, 'normal_fpr': 0.03})."""
        payload = {"overall": metrics, "by": by or {}, "headline": headline or {}}
        with open(self.dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        md = [f"# {self.name}", "", f"- seed: {self.seed}", f"- git: {self.meta['git_hash']}", ""]
        md.append(metrics_to_markdown(metrics, "overall"))
        for axis, table in (by or {}).items():
            if table and isinstance(next(iter(table.values())), dict) and "recall" in next(iter(table.values())):
                md.append(metrics_to_markdown(table, f"by {axis}"))
            else:  # 중첩: axis 값 → 클래스별
                for val, sub in table.items():
                    md.append(metrics_to_markdown(sub, f"by {axis} = {val}"))
        with open(self.dir / "metrics.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        self.meta["status"] = "done"
        self.meta["headline"] = headline or {}
        self._write_meta()
        self.update_summary(headline or {})
        return self.dir / "metrics.json"

    def update_summary(self, headline: dict[str, float]) -> Path:
        path = self.root / "SUMMARY.md"
        header = "| experiment | seed | git | reproducible | headline |\n|---|---|---|---|---|\n"
        if not path.exists():
            path.write_text("# Experiments summary\n\n" + header, encoding="utf-8")
        hl = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in headline.items())
        row = f"| {self.dir.name} | {self.seed} | {self.meta['git_hash']} | {'yes' if self.reproducible else 'NO'} | {hl} |\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(row)
        return path
