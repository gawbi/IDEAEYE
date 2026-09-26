"""YAML 설정 로더. 하드코딩 금지 원칙(CLAUDE.md 7장)에 따라 모든 값은 configs/*.yaml 에서 읽는다."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(path: str | Path) -> Path:
    """상대 경로는 프로젝트 루트 기준으로 해석한다."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(name: str) -> dict[str, Any]:
    """configs/<name>.yaml 을 읽는다. name 에 확장자를 붙여도 된다."""
    fname = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return load_yaml(CONFIG_DIR / fname)
