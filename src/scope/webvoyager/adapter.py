from __future__ import annotations

from pathlib import Path

from scope.common.datasets import DatasetSummary, TaskRecord
from scope.common.io import load_json, save_json
from scope.common.rollout import CommandSpec
from scope.common.types import PipelineContext
from scope.webvoyager.rollout import build_rollout_command


class WebVoyagerAdapter:
    name = "webvoyager"

    def describe(self) -> str:
        return "WebVoyager adapter: discovers tasks from results/task*/eval_result.json"

    def planned_steps(self, ctx: PipelineContext) -> list[str]:
        return [
            "discover WebVoyager task directories from results/task*",
            "treat missing eval_result.json or result != 1 as failed",
            "refresh memory from prior results",
            "select memory anchors",
            "run rollout with WebVoyager executor",
        ]

    def discover_tasks(self, ctx: PipelineContext) -> DatasetSummary:
        cfg = ctx.config
        explicit = [str(task_id) for task_id in cfg.get("task_ids") or []]
        results_dir = Path(str(cfg.get("results_dir") or ""))

        if results_dir and results_dir.exists():
            records = []
            for path in sorted(results_dir.iterdir()):
                if not path.is_dir() or not path.name.startswith("task"):
                    continue
                task_id = path.name[len("task") :]
                if explicit and task_id not in explicit:
                    continue
                eval_path = path / "eval_result.json"
                success, score, reason = _read_eval_result(eval_path)
                records.append(TaskRecord(task_id=task_id, source_path=path, score=score, success=success, reason=reason))
            return DatasetSummary(records)

        return DatasetSummary([TaskRecord(task_id=task_id, reason="explicit task") for task_id in explicit])

    def write_stub_rollout(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        results_dir = Path(str(ctx.config.get("rollout_output_dir") or ctx.config.get("results_dir") or "./artifacts/webvoyager/results"))
        written: list[Path] = []
        for task in tasks:
            task_dir = results_dir / f"task{task.task_id}"
            eval_path = task_dir / "eval_result.json"
            messages_path = task_dir / "interact_messages.json"
            save_json(eval_path, {"task_id": task.task_id, "result": 1, "prototype": True})
            save_json(
                messages_path,
                [
                    {"role": "agent", "content": "Prototype WebVoyager rollout placeholder."}
                ],
            )
            written.extend([eval_path, messages_path])
        return written

    def build_rollout_command(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> CommandSpec:
        return build_rollout_command(ctx.config, [task.task_id for task in tasks])

    def preview_rollout_outputs(self, ctx: PipelineContext, tasks: list[TaskRecord]) -> list[Path]:
        results_dir = Path(str(ctx.config.get("rollout_output_dir") or ctx.config.get("results_dir") or "./artifacts/webvoyager/results"))
        paths: list[Path] = []
        for task in tasks:
            task_dir = results_dir / f"task{task.task_id}"
            paths.extend([task_dir / "eval_result.json", task_dir / "interact_messages.json"])
        return paths


def _read_eval_result(path: Path) -> tuple[bool | None, float | None, str]:
    if not path.exists():
        return False, None, "missing eval_result.json"
    try:
        data = load_json(path)
    except Exception:
        return False, None, "invalid eval_result.json"
    if not isinstance(data, dict):
        return False, None, "eval result is not an object"
    result = data.get("result")
    try:
        score = None if result is None else float(result)
    except Exception:
        score = None
    return result == 1, score, f"result={result}"
