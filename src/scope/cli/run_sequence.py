from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from scope.common.config import build_round_spec, load_config
from scope.common.pipeline import build_round_plan
from scope.common.stages import run_stage
from scope.common.types import PipelineContext
from scope.registry import get_adapter


CLI_STAGES = ("all", "refresh_memory", "select", "update", "rollout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect a sequence of Scope rounds.")
    parser.add_argument(
        "--dataset",
        choices=("webarena", "webshop", "webvoyager"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    return parser


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "pipeline":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def render_templates(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        rendered: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_template") and key[:-9] not in value:
                rendered[key[:-9]] = str(item).format(**params)
            else:
                rendered[key] = render_templates(item, params)
        return rendered
    if isinstance(value, list):
        return [render_templates(item, params) for item in value]
    if isinstance(value, str):
        try:
            return value.format(**params)
        except (KeyError, IndexError, ValueError):
            return value
    return value


def build_round_config(sequence_cfg: dict[str, Any], round_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    base_pipeline = dict(sequence_cfg.get("pipeline") or {})
    base_round_cfg = {key: copy.deepcopy(value) for key, value in sequence_cfg.items() if key not in {"pipeline", "rounds"}}
    base = deep_merge(base_pipeline, base_round_cfg)
    merged = deep_merge(base, round_cfg)
    stage = str(merged.get("stage") or "all")
    if stage not in CLI_STAGES:
        raise ValueError(f"Invalid sequence stage: {stage}")

    params = {
        "iter": int(merged.get("iter", 0)),
        "run": int(merged.get("run", 0)),
        "source_iter": int(merged.get("source_iter", merged.get("iter", 0))),
        "source_run": int(merged.get("source_run", merged.get("run", 0))),
    }
    rendered = render_templates(merged, params)
    rendered["iter"] = params["iter"]
    rendered["run"] = params["run"]
    rendered.pop("stage", None)
    return stage, rendered


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    rounds = config.get("rounds") or []
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Sequence config must contain a non-empty rounds list")

    adapter = get_adapter(args.dataset)
    failed = 0
    for idx, round_cfg in enumerate(rounds):
        if not isinstance(round_cfg, dict):
            raise ValueError(f"Round #{idx} must be a mapping")
        stage, rendered_cfg = build_round_config(config, round_cfg)
        ctx = PipelineContext(
            dataset=args.dataset,
            config_path=args.config,
            config=rendered_cfg,
        )
        spec = build_round_spec(rendered_cfg, stage=stage)
        plan = build_round_plan(ctx, adapter, spec)
        print(f"[sequence] round={idx} iter={plan.spec.iter_id} run={plan.spec.run_id} stage={plan.spec.stage}")
        print(f"[sequence] tasks={plan.summary.total} failed={len(plan.summary.failed)} unknown={len(plan.summary.unknown)}")
        for step in plan.steps:
            print(f"[sequence] - {step}")
        stage_results = run_stage(ctx, adapter, stage, dry_run=args.dry_run)
        round_errors = []
        for result in stage_results:
            print(f"[sequence] {result.stage}: written={len(result.written)} skipped={len(result.skipped)} errors={len(result.errors)}")
            if result.command:
                print(f"[sequence] command: {result.command}")
            for error in result.errors[:3]:
                print(f"[sequence] error: {error}")
            round_errors.extend(result.errors)
        if round_errors:
            failed += 1
            if not args.continue_on_error:
                print(f"[sequence] stop at round={idx}")
                return 1
    print(f"[sequence] completed={len(rounds)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
