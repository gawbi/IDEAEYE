"""experiments/ 디렉터리 탐색 유틸. 'latest:<name>' 표기로 가장 최근 실험 폴더를 찾는다."""
from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT


def resolve_experiment(spec: str, root: Path | None = None) -> Path:
    root = root or PROJECT_ROOT / "experiments"
    if spec.startswith("latest:"):
        name = spec.split(":", 1)[1]
        cands = sorted(d for d in root.iterdir() if d.is_dir() and d.name.endswith("_" + name))
        if not cands:
            raise FileNotFoundError(f"no experiment named {name!r} under {root}")
        return cands[-1]
    p = Path(spec)
    return p if p.is_absolute() else PROJECT_ROOT / p
