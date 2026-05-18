from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def merge_config(cli: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, value in cli.items():
        if value is not None:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class RoundSpec:
    iter_id: int
    run_id: int
    stage: str
    config: dict[str, Any]


def build_round_spec(config: dict[str, Any], *, stage: str) -> RoundSpec:
    return RoundSpec(
        iter_id=int(config.get("iter", 0)),
        run_id=int(config.get("run", 0)),
        stage=stage,
        config=dict(config),
    )
