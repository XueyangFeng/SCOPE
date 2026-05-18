from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

import yaml

ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")

DEFAULTS = {
    'profile': 'webrl_llama',
    'run_script': 'run_with_prefix.py',
    'only_selected_node_tasks': 1,
    'prefix_root': '',
    'test_config_base_dir': 'config_files/wa/train_webarena_new',
    'task_range_start': 0,
    'task_range_end': 1000000000,
    'parallelism': 10,
    'chunk_size': 0,
    'retry_errors': 0,
    'task_timeout_seconds': 900,
    'task_timeout_kill_after_seconds': 30,
    'conda_env_name': 'vab310',
    'prestart_web_reset': 1,
    'prestart_vllm_restart': 1,
    'vllm_restart_base_url': 'http://localhost:18075',
    'prefix_validate_replay': 1,
    'prefix_validate_on_fail': 'restart',
    'prefix_reference_dir': '',
    'prefix_id_repair_enable': 1,
    'prefix_id_repair_model': 'deepseek-v3.2',
    'guided_think_mode': 'auto',
    'guided_think_apply_mode': 'replay',
    'guided_think_model': 'deepseek-v3.2',
    'guided_think_timeout': 8,
    'dataset': 'webarena',
    'iter': 1,
    'run': 0,
}

KEY_MAP = {
    'profile': 'PROFILE',
    'run_script': 'RUN_SCRIPT',
    'result_dir': 'RESULT_DIR',
    'dataset': 'DATASET',
    'provider': 'PROVIDER',
    'mode': 'MODE',
    'temperature': 'TEMPERATURE',
    'top_p': 'TOP_P',
    'max_tokens': 'MAX_TOKENS',
    'model_path': 'MODEL_PATH',
    'instruction_path': 'INSTRUCTION_PATH',
    'stop_token': 'STOP_TOKEN',
    'iter': 'CURRENT_ITER',
    'run': 'CURRENT_RUN',
    'only_selected_node_tasks': 'ONLY_SELECTED_NODE_TASKS',
    'prefix_root': 'PREFIX_ROOT',
    'test_config_base_dir': 'TEST_CONFIG_BASE_DIR',
    'task_range_start': 'TASK_RANGE_START',
    'task_range_end': 'TASK_RANGE_END',
    'parallelism': 'PARALLELISM',
    'chunk_size': 'CHUNK_SIZE',
    'retry_errors': 'RETRY_ERRORS',
    'task_timeout_seconds': 'TASK_TIMEOUT_SECONDS',
    'task_timeout_kill_after_seconds': 'TASK_TIMEOUT_KILL_AFTER_SECONDS',
    'conda_env_name': 'CONDA_ENV_NAME',
    'prestart_web_reset': 'PRESTART_WEB_RESET',
    'prestart_vllm_restart': 'PRESTART_VLLM_RESTART',
    'vllm_restart_base_url': 'VLLM_RESTART_BASE_URL',
    'prefix_validate_replay': 'PREFIX_VALIDATE_REPLAY',
    'prefix_validate_on_fail': 'PREFIX_VALIDATE_ON_FAIL',
    'prefix_reference_dir': 'PREFIX_REFERENCE_DIR',
    'prefix_id_repair_enable': 'PREFIX_ID_REPAIR_ENABLE',
    'prefix_id_repair_model': 'PREFIX_ID_REPAIR_MODEL',
    'guided_think_mode': 'GUIDED_THINK_MODE',
    'guided_think_apply_mode': 'GUIDED_THINK_APPLY_MODE',
    'guided_think_model': 'GUIDED_THINK_MODEL',
    'guided_think_timeout': 'GUIDED_THINK_TIMEOUT',
}


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Wrapper config must be a mapping: {path}')
    merged = dict(DEFAULTS)
    merged.update(data)
    return merged


def shell_assignments(cfg: dict) -> str:
    lines: list[str] = []
    for key, env_name in KEY_MAP.items():
        if key not in cfg:
            continue
        value = cfg[key]
        if value is None:
            continue
        lines.append(f'export {env_name}={shlex.quote(expand_env(str(value)))}')
    return '\n'.join(lines)


def expand_env(value: str) -> str:
    def replace_default(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or default

    value = ENV_DEFAULT_RE.sub(replace_default, value)
    return os.path.expanduser(os.path.expandvars(value))


def main() -> int:
    raw = os.environ.get('WRAPPER_CONFIG', '')
    if not raw:
        print('WRAPPER_CONFIG is required', file=sys.stderr)
        return 1
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        print(f'Wrapper config not found: {path}', file=sys.stderr)
        return 1
    cfg = load_config(path)
    print(shell_assignments(cfg))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
