from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from scope.common.datasets import DatasetAdapter, DatasetSummary, TaskRecord
from scope.common.generator import generate_memory_with_llm
from scope.common.llm import load_llm_config
from scope.common.memory import (
    build_memory_layout,
    write_llm_memory_update,
    write_llm_prefix_memory,
    write_llm_selected_anchor,
    write_stub_memory_update,
    write_stub_prefix_memory,
    write_stub_selected_anchor,
)
from scope.common.prompts import resolve_prompt_bundle
from scope.common.rollout import run_command
from scope.common.selector import select_anchor_with_llm
from scope.common.types import PipelineContext
from scope.common.update import update_memory_with_llm


STAGES = ("refresh_memory", "select", "update", "rollout")


@dataclass
class StageResult:
    stage: str
    written: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    command: str = ""


def tasks_for_memory_refresh(summary: DatasetSummary) -> list[TaskRecord]:
    return summary.failed or summary.unknown or summary.tasks


def run_refresh_memory(ctx: PipelineContext, adapter: DatasetAdapter, *, dry_run: bool = False) -> StageResult:
    summary = adapter.discover_tasks(ctx)
    layout = build_memory_layout(ctx.config.get("memory_root", "./artifacts/memory_store"))
    prompts = resolve_prompt_bundle(ctx.dataset, ctx.config)
    llm_config = load_llm_config(ctx.config)
    result = StageResult(stage="refresh_memory")
    for task in tasks_for_memory_refresh(summary):
        if layout.has_prefix_memory(task.task_id):
            result.skipped.append(task.task_id)
            continue
        path = layout.prefix_memory_path(task.task_id)
        if not dry_run:
            if llm_config.enabled:
                try:
                    llm_result = generate_memory_with_llm(
                        task,
                        prompts=prompts,
                        llm_config=llm_config,
                        extra=ctx.config,
                    )
                    path = write_llm_prefix_memory(layout, task, dataset=ctx.dataset, llm_result=llm_result)
                except Exception as exc:
                    result.errors.append(f"{task.task_id}: {exc}")
                    continue
            else:
                path = write_stub_prefix_memory(layout, task, dataset=ctx.dataset, prompts=prompts)
        result.written.append(path)
    return result


def run_select(ctx: PipelineContext, adapter: DatasetAdapter, *, dry_run: bool = False) -> StageResult:
    summary = adapter.discover_tasks(ctx)
    layout = build_memory_layout(ctx.config.get("memory_root", "./artifacts/memory_store"))
    prompts = resolve_prompt_bundle(ctx.dataset, ctx.config)
    llm_config = load_llm_config(ctx.config)
    result = StageResult(stage="select")
    for task in tasks_for_memory_refresh(summary):
        if not layout.has_prefix_memory(task.task_id) and not dry_run:
            result.skipped.append(task.task_id)
            continue
        path = layout.selected_anchor_path(task.task_id)
        if not dry_run:
            if llm_config.enabled:
                try:
                    llm_result = select_anchor_with_llm(
                        task,
                        prompts=prompts,
                        layout=layout,
                        llm_config=llm_config,
                        extra=ctx.config,
                    )
                    path = write_llm_selected_anchor(layout, task, dataset=ctx.dataset, llm_result=llm_result)
                except Exception as exc:
                    result.errors.append(f"{task.task_id}: {exc}")
                    continue
            else:
                path = write_stub_selected_anchor(layout, task, dataset=ctx.dataset, prompts=prompts)
        result.written.append(path)
    return result


def run_update(ctx: PipelineContext, adapter: DatasetAdapter, *, dry_run: bool = False) -> StageResult:
    summary = adapter.discover_tasks(ctx)
    layout = build_memory_layout(ctx.config.get("memory_root", "./artifacts/memory_store"))
    prompts = resolve_prompt_bundle(ctx.dataset, ctx.config)
    llm_config = load_llm_config(ctx.config)
    result = StageResult(stage="update")
    for task in tasks_for_memory_refresh(summary):
        if not layout.has_selected_anchor(task.task_id) and not dry_run:
            result.skipped.append(task.task_id)
            continue
        path = layout.memory_update_path(task.task_id)
        if not dry_run:
            if llm_config.enabled:
                try:
                    llm_result = update_memory_with_llm(
                        task,
                        prompts=prompts,
                        layout=layout,
                        llm_config=llm_config,
                        extra=ctx.config,
                    )
                    path = write_llm_memory_update(layout, task, dataset=ctx.dataset, llm_result=llm_result)
                except Exception as exc:
                    result.errors.append(f"{task.task_id}: {exc}")
                    continue
            else:
                path = write_stub_memory_update(layout, task, dataset=ctx.dataset, prompts=prompts)
        result.written.append(path)
    return result


def run_rollout(ctx: PipelineContext, adapter: DatasetAdapter, *, dry_run: bool = False) -> StageResult:
    summary = adapter.discover_tasks(ctx)
    tasks = summary.tasks
    result = StageResult(stage="rollout")
    command_spec = adapter.build_rollout_command(ctx, tasks)
    result.command = command_spec.format()
    if dry_run:
        result.written.extend(adapter.preview_rollout_outputs(ctx, tasks))
        return result
    returncode = run_command(command_spec)
    if returncode != 0:
        result.errors.append(f"rollout command exited with code {returncode}")
    result.written.extend(adapter.preview_rollout_outputs(ctx, tasks))
    return result


def run_stage(ctx: PipelineContext, adapter: DatasetAdapter, stage: str, *, dry_run: bool = False) -> list[StageResult]:
    if stage == "all":
        return [
            run_refresh_memory(ctx, adapter, dry_run=dry_run),
            run_select(ctx, adapter, dry_run=dry_run),
            run_update(ctx, adapter, dry_run=dry_run),
            run_rollout(ctx, adapter, dry_run=dry_run),
        ]
    if stage == "refresh_memory":
        return [run_refresh_memory(ctx, adapter, dry_run=dry_run)]
    if stage == "select":
        return [run_select(ctx, adapter, dry_run=dry_run)]
    if stage == "update":
        return [run_update(ctx, adapter, dry_run=dry_run)]
    if stage == "rollout":
        return [run_rollout(ctx, adapter, dry_run=dry_run)]
    raise ValueError(f"Unsupported stage: {stage}")
