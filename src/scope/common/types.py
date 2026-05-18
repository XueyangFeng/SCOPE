from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineContext:
    dataset: str
    config_path: Path
    config: dict[str, Any]
