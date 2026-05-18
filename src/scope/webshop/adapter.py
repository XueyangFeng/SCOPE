from __future__ import annotations

from pathlib import Path
from typing import Any

from scope.common.datasets import DatasetSummary, TaskRecord
from scope.common.io import load_json, save_json
from scope.common.rollout import CommandSpec
from scope.common.types import PipelineContext
from scope.webshop.rollout import build_rollout_command


class WebShopAdapter:
    name = "webshop"

    def describe(self) -> str:
        return "WebShop adapter: discovers tasks from source_trace_dir/*.json and reward"

    def planned_steps(self, ctx: PipelineContext) -> list[str]:
        return [
            "discover WebShop traces from source_trace_dir",
            "parse reward and treat reward < success_threshold as failed",
            "refresh memory from source traces",
            "select memory anchors for failed tasks",
            "run rollout with WebShop/HPL environment",
        ]

    def discover_tasks(self, ctx: PipelineContext) -> DatasetSummary:
        cfg = ctx.config
        explicit = [str(task_id) for task_id in cfg.get("task_ids") or []]
        trace_dir = Path(str(cfg.get("source_trace_dir") or ""))
        threshold = float(cfg.get("success_threshold", 1.0))

        if trace_dir and trace_dir.exists():
            records = []
            for path in sorted(trace_dir.glob("*.json")):
                task_id = path.stem
                if explicit and task_id not in explicit:
                    continue
                reward = _extract_reward(path)
                success = None if reward is None else reward >= threshold
                reason = "reward missing" if reward is None else f"reward={reward}"
                records.append(TaskRecord(task_id=task_id, source_path=path, score=reward, success=success, reason=reason))
            return DatasetSummary(records)

        return DatasetSummary([TaskRecord(task_id=task_id, reason="explicit task") for task_id in explicit])

    def write_stub_rollout(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        output_dir = Path(str(ctx.config.get("rollout_output_dir") or "./artifacts/webshop/rollout/run0"))
        written: list[Path] = []
        for task in tasks:
            path = output_dir / f"{task.task_id}.json"
            save_json(
                path,
                {
                    "task_id": task.task_id,
                    "reward": 1.0,
                    "trajectory": [
                        {"role": "agent", "content": "Prototype rollout placeholder."}
                    ],
                    "prototype": True,
                },
            )
            written.append(path)
        return written

    def build_rollout_command(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> CommandSpec:
        return build_rollout_command(ctx.config, [task.task_id for task in tasks])

    def preview_rollout_outputs(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        output_dir = Path(str(ctx.config.get("rollout_output_dir") or "./artifacts/webshop/rollout/run0"))
        return [output_dir / f"{task.task_id}.json" for task in tasks]


def _extract_reward(path: Path) -> float | None:
    try:
        data = load_json(path)
    except Exception:
        return None
    reward = _find_key(data, "reward")
    if reward is None:
        reward = _find_key(data, "score")
    try:
        return None if reward is None else float(reward)
    except Exception:
        return None


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None
