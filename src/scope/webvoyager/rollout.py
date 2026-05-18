from __future__ import annotations

from pathlib import Path
from typing import Any

from scope.common.rollout import CommandSpec, append_arg, append_flag


def build_rollout_command(config: dict[str, Any], task_ids: list[str]) -> CommandSpec:
    rollout_cfg = dict(config.get("rollout") or {})
    webvoyager_root = Path(str(rollout_cfg.get("webvoyager_root") or "${WEBVOYAGER_ROOT}"))
    python_bin = str(rollout_cfg.get("python") or "python")
    run_py = Path(str(rollout_cfg.get("run_path") or "run.py"))
    output_dir = Path(str(config.get("rollout_output_dir") or rollout_cfg.get("output_dir") or "outputs/webvoyager"))

    command = [python_bin, str(run_py)]
    append_arg(command, "--test_file", rollout_cfg.get("test_file") or "data/WebVoyager_trainset.jsonl")
    append_arg(command, "--output_dir", output_dir)
    append_arg(command, "--api_model", rollout_cfg.get("api_model") or "gpt-4o")
    append_arg(command, "--max_iter", rollout_cfg.get("max_iter", 15))
    append_arg(command, "--num_workers", rollout_cfg.get("num_workers", 4))
    append_arg(command, "--max_attached_imgs", rollout_cfg.get("max_attached_imgs", 1))
    append_arg(command, "--window_width", rollout_cfg.get("window_width", 1024))
    append_arg(command, "--window_height", rollout_cfg.get("window_height", 768))
    append_flag(command, "--text_only", rollout_cfg.get("text_only"))
    append_flag(command, "--headless", rollout_cfg.get("headless"))
    append_flag(command, "--save_accessibility_tree", rollout_cfg.get("save_accessibility_tree"))
    append_flag(command, "--group_by_site", rollout_cfg.get("group_by_site"))
    append_arg(command, "--eval_model", rollout_cfg.get("eval_model"))
    if task_ids:
        append_arg(command, "--task_ids", ",".join(task_ids))
    append_arg(command, "--prefix_root", config.get("memory_root"))
    append_flag(command, "--prefix_validate_replay", rollout_cfg.get("prefix_validate_replay"))
    append_arg(command, "--prefix_validate_on_fail", rollout_cfg.get("prefix_validate_on_fail"))
    append_flag(command, "--prefix_id_repair_enable", rollout_cfg.get("prefix_id_repair_enable"))
    append_arg(command, "--prefix_id_repair_model", rollout_cfg.get("prefix_id_repair_model"))
    append_arg(command, "--guided_think_mode", rollout_cfg.get("guided_think_mode"))
    append_arg(command, "--guided_think_apply_mode", rollout_cfg.get("guided_think_apply_mode"))
    append_arg(command, "--guided_think_model", rollout_cfg.get("guided_think_model"))
    append_arg(command, "--guided_api_base", rollout_cfg.get("guided_api_base"))
    append_arg(command, "--guided_api_key", rollout_cfg.get("guided_api_key"))
    append_arg(command, "--guided_think_timeout", rollout_cfg.get("guided_think_timeout"))
    append_arg(command, "--iter", config.get("iter"))
    append_arg(command, "--run", config.get("run"))

    return CommandSpec(
        command=command,
        cwd=webvoyager_root,
        env={},
        note="Requires the WebVoyager environment with Scope/DCRL prefix replay support.",
    )
