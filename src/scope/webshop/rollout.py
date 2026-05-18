from __future__ import annotations

from pathlib import Path
from typing import Any

from scope.common.rollout import CommandSpec, append_arg, append_flag


def build_rollout_command(config: dict[str, Any], task_ids: list[str]) -> CommandSpec:
    rollout_cfg = dict(config.get("rollout") or {})
    hpl_root = Path(str(rollout_cfg.get("hpl_root") or "${HPL_ROOT}"))
    python_bin = str(rollout_cfg.get("python") or "python")
    evaluation_path = Path(str(rollout_cfg.get("evaluation_path") or "evaluation.py"))
    output_dir = Path(str(config.get("rollout_output_dir") or rollout_cfg.get("output_dir") or "outputs/webshop"))

    command = [python_bin, str(evaluation_path)]
    append_arg(command, "--exp_path", rollout_cfg.get("exp_path"))
    append_arg(command, "--exp_config", rollout_cfg.get("exp_config"))
    append_arg(command, "--exp_name", rollout_cfg.get("exp_name"))
    append_arg(command, "--split", rollout_cfg.get("split"))
    append_arg(command, "--part_num", rollout_cfg.get("part_num"))
    append_arg(command, "--part_idx", rollout_cfg.get("part_idx"))
    append_arg(command, "--agent_path", rollout_cfg.get("agent_path"))
    append_arg(command, "--agent_config", rollout_cfg.get("agent_config"))
    append_arg(command, "--model_name", rollout_cfg.get("model_name") or "${DCRL_MODEL}")
    append_arg(command, "--api_base", rollout_cfg.get("api_base") or "${DCRL_BASE_URL}")
    append_arg(command, "--api_key", rollout_cfg.get("api_key") or "${DCRL_API_KEY}")
    append_arg(command, "--temperature", rollout_cfg.get("temperature"))
    append_arg(command, "--metaplan_path", rollout_cfg.get("metaplan_path"))
    append_arg(command, "--metaplan_type", rollout_cfg.get("metaplan_type"))
    append_arg(command, "--incorporation_type", rollout_cfg.get("incorporation_type"))
    append_arg(command, "--output_dir", output_dir)
    append_arg(command, "--prefix_root", config.get("memory_root"))
    append_arg(command, "--iter", config.get("iter"))
    append_arg(command, "--run", config.get("run"))
    if task_ids:
        append_arg(command, "--task_ids", ",".join(task_ids))
    append_arg(command, "--guided_think_mode", rollout_cfg.get("guided_think_mode"))
    append_arg(command, "--guided_think_apply_mode", rollout_cfg.get("guided_think_apply_mode"))
    append_arg(command, "--guided_think_model", rollout_cfg.get("guided_think_model"))
    append_arg(command, "--guided_api_base", rollout_cfg.get("guided_api_base"))
    append_arg(command, "--guided_api_key", rollout_cfg.get("guided_api_key"))
    append_arg(command, "--guided_think_timeout", rollout_cfg.get("guided_think_timeout"))
    append_arg(command, "--workers", rollout_cfg.get("workers"))
    append_flag(command, "--debug", rollout_cfg.get("debug"))
    append_flag(command, "--override", rollout_cfg.get("override"))

    extra_args = rollout_cfg.get("extra_args") or []
    if not isinstance(extra_args, list):
        raise ValueError("rollout.extra_args must be a list")
    command.extend(str(item) for item in extra_args)

    env: dict[str, str] = {}
    pythonpath_base = rollout_cfg.get("webshop_pythonpath_base")
    if pythonpath_base:
        base = Path(str(pythonpath_base))
        env["PYTHONPATH"] = ":".join([str(base), str(base / "webshop")])

    return CommandSpec(
        command=command,
        cwd=hpl_root,
        env=env,
        note="Requires the HPL/WebShop environment and a policy model endpoint.",
    )
