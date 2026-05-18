from __future__ import annotations

from pathlib import Path
from typing import Any

from scope.common.rollout import CommandSpec


def build_rollout_command(config: dict[str, Any], task_ids: list[str]) -> CommandSpec:
    rollout_cfg = dict(config.get("rollout") or {})
    vab_root = Path(str(rollout_cfg.get("vab_root") or "${VAB_WEBARENA_ROOT}"))
    script = str(rollout_cfg.get("script") or "wa_parallel_run_yi.sh")
    command = ["bash", script]

    env: dict[str, str] = {}
    wrapper_config = rollout_cfg.get("wrapper_config")
    if wrapper_config:
        wrapper_path = Path(str(wrapper_config))
        if not wrapper_path.is_absolute():
            wrapper_path = (Path.cwd() / wrapper_path).resolve()
        env["WRAPPER_CONFIG"] = str(wrapper_path)
    memory_root = config.get("memory_root")
    if memory_root:
        env["SCOPE_PREFIX_ROOT"] = str(memory_root)
    if task_ids:
        env["TASK_IDS"] = ",".join(task_ids)
    env["RUN_IN_BACKGROUND"] = str(rollout_cfg.get("run_in_background", "0"))

    return CommandSpec(
        command=command,
        cwd=vab_root,
        env=env,
        note="Requires the VisualAgentBench/WebArena-Lite environment and wrapper config.",
    )
