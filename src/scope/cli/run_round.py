from __future__ import annotations

import argparse
from pathlib import Path

from scope.common.config import build_round_spec, load_config
from scope.common.pipeline import build_round_plan
from scope.common.stages import run_stage
from scope.common.types import PipelineContext
from scope.registry import get_adapter


CLI_STAGES = ("all", "refresh_memory", "select", "update", "rollout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect one Scope round.")
    parser.add_argument(
        "--dataset",
        choices=("webarena", "webshop", "webvoyager"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=CLI_STAGES, default="all")
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    ctx = PipelineContext(
        dataset=args.dataset,
        config_path=args.config,
        config=config,
    )
    adapter = get_adapter(args.dataset)
    spec = build_round_spec(config, stage=args.stage)
    plan = build_round_plan(ctx, adapter, spec)

    print(f"dataset: {args.dataset}")
    print(f"config: {args.config}")
    print(f"adapter: {plan.adapter_description}")
    print(f"tasks: {plan.summary.total}")
    print(f"failed: {len(plan.summary.failed)}")
    print(f"unknown: {len(plan.summary.unknown)}")
    print("steps:")
    for step in plan.steps:
        print(f"- {step}")
    if args.dry_run:
        print("mode: dry-run")
    stage_results = run_stage(ctx, adapter, args.stage, dry_run=args.dry_run)
    print("stage results:")
    for result in stage_results:
        print(f"- {result.stage}: written={len(result.written)} skipped={len(result.skipped)} errors={len(result.errors)}")
        if result.command:
            print(f"  command: {result.command}")
        for path in result.written[:5]:
            print(f"  wrote: {path}")
        for error in result.errors[:3]:
            print(f"  error: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
