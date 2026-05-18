from __future__ import annotations

from dataclasses import dataclass

from scope.common.config import RoundSpec
from scope.common.datasets import DatasetAdapter, DatasetSummary
from scope.common.types import PipelineContext


@dataclass(frozen=True)
class RoundPlan:
    dataset: str
    adapter_description: str
    spec: RoundSpec
    steps: list[str]
    summary: DatasetSummary


def plan_round(ctx: PipelineContext, adapter: DatasetAdapter) -> list[str]:
    return adapter.planned_steps(ctx)


def build_round_plan(ctx: PipelineContext, adapter: DatasetAdapter, spec: RoundSpec) -> RoundPlan:
    return RoundPlan(
        dataset=ctx.dataset,
        adapter_description=adapter.describe(),
        spec=spec,
        steps=adapter.planned_steps(ctx),
        summary=adapter.discover_tasks(ctx),
    )
