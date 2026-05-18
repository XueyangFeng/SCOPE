from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPT_NAMES = ("generator", "update", "selector")


@dataclass(frozen=True)
class PromptBundle:
    generator: Path
    update: Path
    selector: Path

    def for_stage(self, stage: str) -> Path:
        if stage == "generator":
            return self.generator
        if stage == "update":
            return self.update
        if stage == "selector":
            return self.selector
        raise ValueError(f"Unknown prompt stage: {stage}")


def default_prompt_bundle(dataset: str) -> PromptBundle:
    root = Path(__file__).resolve().parents[1] / dataset / "prompts"
    return PromptBundle(
        generator=root / "generator.txt",
        update=root / "update.txt",
        selector=root / "selector.txt",
    )


def resolve_prompt_bundle(dataset: str, config: dict) -> PromptBundle:
    defaults = default_prompt_bundle(dataset)
    prompt_cfg = dict(config.get("prompts") or {})
    return PromptBundle(
        generator=Path(str(prompt_cfg.get("generator") or defaults.generator)),
        update=Path(str(prompt_cfg.get("update") or defaults.update)),
        selector=Path(str(prompt_cfg.get("selector") or defaults.selector)),
    )
