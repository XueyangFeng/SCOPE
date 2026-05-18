from __future__ import annotations

from pathlib import Path

from scope.common.datasets import DatasetSummary, TaskRecord
from scope.common.io import load_json, save_json
from scope.common.rollout import CommandSpec
from scope.common.types import PipelineContext
from scope.webarena.rollout import build_rollout_command


class WebArenaAdapter:
    name = "webarena"

    def describe(self) -> str:
        return "WebArena adapter: discovers tasks from actions/*.json or traces/*.jsonl"

    def planned_steps(self, ctx: PipelineContext) -> list[str]:
        return [
            "discover WebArena tasks from actions/traces",
            "treat action score <= success_threshold as failed",
            "refresh memory for failed or missing tasks",
            "select memory anchors",
            "run rollout with WebArena backend",
        ]

    def discover_tasks(self, ctx: PipelineContext) -> DatasetSummary:
        cfg = ctx.config
        explicit = [str(task_id) for task_id in cfg.get("task_ids") or []]
        actions_dir = Path(str(cfg.get("actions_dir") or ""))
        traces_dir = Path(str(cfg.get("traces_dir") or ""))
        threshold = float(cfg.get("success_threshold", 0.5))

        records: list[TaskRecord] = []
        if actions_dir and actions_dir.exists():
            for path in sorted(actions_dir.glob("*.json")):
                task_id = path.stem
                if explicit and task_id not in explicit:
                    continue
                score = _read_score(path)
                success = None if score is None else score > threshold
                reason = "score missing" if score is None else f"score={score}"
                records.append(TaskRecord(task_id=task_id, source_path=path, score=score, success=success, reason=reason))
            return DatasetSummary(records)

        if traces_dir and traces_dir.exists():
            for path in sorted(traces_dir.glob("*.jsonl")):
                if path.name.endswith(".dehydrated.jsonl"):
                    continue
                task_id = path.stem
                if explicit and task_id not in explicit:
                    continue
                records.append(TaskRecord(task_id=task_id, source_path=path, success=None, reason="trace discovered"))
            return DatasetSummary(records)

        return DatasetSummary([TaskRecord(task_id=task_id, reason="explicit task") for task_id in explicit])

    def write_stub_rollout(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        actions_dir = Path(str(ctx.config.get("rollout_output_dir") or ctx.config.get("actions_dir") or "./artifacts/webarena/actions"))
        written: list[Path] = []
        for task in tasks:
            path = actions_dir / f"{task.task_id}.json"
            save_json(
                path,
                {
                    "task_id": task.task_id,
                    "score": 1.0,
                    "actions": ["noop()"],
                    "prototype": True,
                },
            )
            written.append(path)
        return written

    def build_rollout_command(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> CommandSpec:
        return build_rollout_command(ctx.config, [task.task_id for task in tasks])

    def preview_rollout_outputs(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        actions_dir = Path(str(ctx.config.get("rollout_output_dir") or ctx.config.get("actions_dir") or "./artifacts/webarena/actions"))
        return [actions_dir / f"{task.task_id}.json" for task in tasks]


def _read_score(path: Path) -> float | None:
    try:
        data = load_json(path)
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("score") is None:
        return None
    try:
        return float(data["score"])
    except Exception:
        return None
