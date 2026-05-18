from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value == '1'


def env_text(name: str, default: str = '') -> str:
    return os.environ.get(name, default)


@dataclass
class TaskCommandConfig:
    run_script: str
    instruction_path: str
    task_id: int
    result_dir: str
    test_config_base_dir: str
    provider: str
    model_path: str
    mode: str
    temperature: str
    top_p: str
    max_tokens: str
    stop_token: str
    prefix_validate_replay: bool
    prefix_validate_on_fail: str
    prefix_reference_dir: str
    prefix_allow_repairable: bool
    prefix_id_repair_enable: bool
    prefix_id_repair_model: str
    prefix_actions_json: str


def load_config() -> TaskCommandConfig:
    task_id_text = env_text('TASK_ID')
    if not task_id_text:
        raise ValueError('TASK_ID is required')
    return TaskCommandConfig(
        run_script=env_text('RUN_SCRIPT', 'run.py'),
        instruction_path=env_text('INSTRUCTION_PATH'),
        task_id=int(task_id_text),
        result_dir=env_text('RESULT_DIR'),
        test_config_base_dir=env_text('TEST_CONFIG_BASE_DIR'),
        provider=env_text('PROVIDER', 'openai'),
        model_path=env_text('MODEL_PATH'),
        mode=env_text('MODE', 'completion'),
        temperature=env_text('TEMPERATURE', '0.0'),
        top_p=env_text('TOP_P', '0.9'),
        max_tokens=env_text('MAX_TOKENS', '2048'),
        stop_token=env_text('STOP_TOKEN'),
        prefix_validate_replay=env_flag('PREFIX_VALIDATE_REPLAY', True),
        prefix_validate_on_fail=env_text('PREFIX_VALIDATE_ON_FAIL', 'restart'),
        prefix_reference_dir=env_text('PREFIX_REFERENCE_DIR'),
        prefix_allow_repairable=env_flag('PREFIX_ALLOW_REPAIRABLE', True),
        prefix_id_repair_enable=env_flag('PREFIX_ID_REPAIR_ENABLE', True),
        prefix_id_repair_model=env_text('PREFIX_ID_REPAIR_MODEL', 'deepseek-v3.2'),
        prefix_actions_json=env_text('PREFIX_ACTIONS_JSON'),
    )


def build_task_command(cfg: TaskCommandConfig) -> str:
    parts = [
        'python', cfg.run_script,
        '--instruction_path', cfg.instruction_path,
        '--test_start_idx', str(cfg.task_id),
        '--test_end_idx', str(cfg.task_id + 1),
        '--result_dir', cfg.result_dir,
        '--test_config_base_dir', cfg.test_config_base_dir,
        '--provider', cfg.provider,
        '--model', cfg.model_path,
        '--mode', cfg.mode,
        '--temperature', cfg.temperature,
        '--top_p', cfg.top_p,
        '--max_obs_length', '0',
        '--max_tokens', cfg.max_tokens,
        '--viewport_width', '1280',
        '--viewport_height', '720',
        '--parsing_failure_th', '5',
        '--repeating_action_failure_th', '5',
        '--action_set_tag', 'webrl_id',
        '--observation_type', 'webrl',
    ]
    if cfg.stop_token:
        parts.extend(['--stop_token', cfg.stop_token])
    if os.path.basename(cfg.run_script) == 'run_with_prefix.py':
        if cfg.prefix_validate_replay:
            parts.extend(['--prefix_validate_replay', '--prefix_validate_on_fail', cfg.prefix_validate_on_fail])
            if cfg.prefix_reference_dir:
                parts.extend(['--prefix_reference_dir', cfg.prefix_reference_dir])
            if not cfg.prefix_allow_repairable:
                parts.append('--prefix_disallow_repairable')
        if cfg.prefix_id_repair_enable:
            parts.extend(['--prefix_id_repair_enable', '--prefix_id_repair_model', cfg.prefix_id_repair_model])
        if cfg.prefix_actions_json:
            json.loads(cfg.prefix_actions_json)
            parts.extend(['--prefix_actions', cfg.prefix_actions_json])
    return ' '.join(shlex.quote(part) for part in parts)


def main() -> int:
    print(build_task_command(load_config()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
