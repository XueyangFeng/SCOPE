from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from scope.common.rollout import CommandSpec
from scope.common.types import PipelineContext


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    source_path: Path | None = None
    score: float | None = None
    success: bool | None = None
    reason: str = ""


@dataclass(frozen=True)
class DatasetSummary:
    tasks: list[TaskRecord]

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def failed(self) -> list[TaskRecord]:
        return [task for task in self.tasks if task.success is False]

    @property
    def unknown(self) -> list[TaskRecord]:
        return [task for task in self.tasks if task.success is None]


class DatasetAdapter(Protocol):
    name: str

    def describe(self) -> str:
        ...

    def planned_steps(self, ctx: PipelineContext) -> list[str]:
        ...

    def discover_tasks(self, ctx: PipelineContext) -> DatasetSummary:
        ...

    def write_stub_rollout(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        ...

    def build_rollout_command(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> CommandSpec:
        ...

    def preview_rollout_outputs(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        ...
