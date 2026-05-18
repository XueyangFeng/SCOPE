from __future__ import annotations

import os
import sys
from pathlib import Path


def shell_quote(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def auto_result_dir(result_root: str, profile: str, model_path: str) -> str:
    model_path_obj = Path(model_path)
    parent = model_path_obj.parent.name
    leaf = model_path_obj.name
    model_name = f"{parent}__{leaf}" if parent not in {'.', '/', ''} else leaf
    model_name = ''.join(c if (c.isalnum() or c in '_.-') else '_' for c in model_name.replace(' ', '_'))
    model_name = model_name or 'unknown_model'
    return f"{result_root.rstrip('/')}/results_{profile}_{model_name}"


def main() -> int:
    profile = env('PROFILE', 'webrl_llama')
    dataset = env('DATASET', 'webarena')
    provider = env('PROVIDER', 'openai')
    mode = env('MODE', 'completion')
    temperature = env('TEMPERATURE', '0.0')
    top_p = env('TOP_P', '0.9')
    max_tokens = env('MAX_TOKENS', '2048')
    test_config_base_dir = env('TEST_CONFIG_BASE_DIR', 'config_files/wa/test_webarena_lite')
    llm_raw_log_dir = env('LLM_RAW_LOG_DIR', './debug_info/llm_raw')
    server = env('OPENAI_API_URL_OVERRIDE', 'http://localhost:8075/v1')
    openai_api_key = env('OPENAI_API_KEY', '')
    openai_organization = env('OPENAI_ORGANIZATION', '')
    openai_eval_api_url = env('OPENAI_EVAL_API_URL', '')
    openai_eval_api_key = env('OPENAI_EVAL_API_KEY', openai_api_key)
    conda_env_name = env('CONDA_ENV_NAME', 'vab310')
    full_ready_text = env('FULL_READY_TEXT', 'Ready for duty!')
    full_reset_timeout = env('FULL_RESET_TIMEOUT', '300')
    parallelism = env('PARALLELISM', '10')
    chunk_size = env('CHUNK_SIZE', '0')
    task_ids = env('TASK_IDS', '')
    sites = env('SITES', 'shopping,reddit,gitlab,shopping_admin')
    site_filter_mode = env('SITE_FILTER_MODE', 'any')
    retry_errors = env('RETRY_ERRORS', '0')
    tolerance = env('TOLERANCE', '2')
    proxy_url = env('PROXY_URL', '')
    task_timeout_seconds = env('TASK_TIMEOUT_SECONDS', '0')
    task_timeout_kill_after_seconds = env('TASK_TIMEOUT_KILL_AFTER_SECONDS', '30')
    run_script = env('RUN_SCRIPT', 'run.py')
    prefix_root = env('PREFIX_ROOT', '')
    only_selected_node_tasks = env('ONLY_SELECTED_NODE_TASKS', '0')
    prefix_validate_replay = env('PREFIX_VALIDATE_REPLAY', '1')
    prefix_validate_on_fail = env('PREFIX_VALIDATE_ON_FAIL', 'restart')
    prefix_reference_dir = env('PREFIX_REFERENCE_DIR', '')
    prefix_allow_repairable = '1'
    prefix_id_repair_enable = env('PREFIX_ID_REPAIR_ENABLE', '1')
    prefix_id_repair_model = env('PREFIX_ID_REPAIR_MODEL', env('DCRL_MODEL', 'deepseek-v3.2'))
    prestart_web_reset = env('PRESTART_WEB_RESET', '1')
    prestart_vllm_restart = env('PRESTART_VLLM_RESTART', '1')
    skip_web_reset = env('SKIP_WEB_RESET', '1')
    web_reset_timeout = env('WEB_RESET_TIMEOUT', '300')
    vllm_restart_base_url = env('VLLM_RESTART_BASE_URL', 'http://localhost:18075')
    vllm_restart_key = env('VLLM_RESTART_KEY', openai_api_key)
    model_path = env('MODEL_PATH', '')
    instruction_path = env('INSTRUCTION_PATH', '')
    profile_stop_token = env('PROFILE_STOP_TOKEN', '<|eot_id|>')
    vllm_restart_model = env('VLLM_RESTART_MODEL', model_path)
    vllm_restart_timeout = env('VLLM_RESTART_TIMEOUT', '900')
    stop_token = os.environ.get('STOP_TOKEN') or profile_stop_token or '<|eot_id|>'
    result_dir = os.environ.get('RESULT_DIR', '')
    if not result_dir:
        print('RESULT_DIR must be set explicitly', file=sys.stderr)
        return 1
    guideness_dir = env('GUIDENESS_DIR', f'{result_dir}/guideness')

    mapping = {
        'DATASET': dataset,
        'provider': provider,
        'MODE': mode,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'test_config_base_dir': test_config_base_dir,
        'LLM_RAW_LOG_DIR': llm_raw_log_dir,
        'SERVER': server,
        'OPENAI_API_KEY': openai_api_key,
        'OPENAI_ORGANIZATION': openai_organization,
        'OPENAI_EVAL_API_URL': openai_eval_api_url,
        'OPENAI_EVAL_API_KEY': openai_eval_api_key,
        'CONDA_ENV_NAME': conda_env_name,
        'FULL_READY_TEXT': full_ready_text,
        'FULL_RESET_TIMEOUT': full_reset_timeout,
        'PARALLELISM': parallelism,
        'CHUNK_SIZE': chunk_size,
        'TASK_IDS': task_ids,
        'SITES': sites,
        'SITE_FILTER_MODE': site_filter_mode,
        'RETRY_ERRORS': retry_errors,
        'TOLERANCE': tolerance,
        'PROXY_URL': proxy_url,
        'TASK_TIMEOUT_SECONDS': task_timeout_seconds,
        'TASK_TIMEOUT_KILL_AFTER_SECONDS': task_timeout_kill_after_seconds,
        'RUN_SCRIPT': run_script,
        'PREFIX_ROOT': prefix_root,
        'ONLY_SELECTED_NODE_TASKS': only_selected_node_tasks,
        'PREFIX_VALIDATE_REPLAY': prefix_validate_replay,
        'PREFIX_VALIDATE_ON_FAIL': prefix_validate_on_fail,
        'PREFIX_REFERENCE_DIR': prefix_reference_dir,
        'PREFIX_ALLOW_REPAIRABLE': prefix_allow_repairable,
        'PREFIX_ID_REPAIR_ENABLE': prefix_id_repair_enable,
        'PREFIX_ID_REPAIR_MODEL': prefix_id_repair_model,
        'PRESTART_WEB_RESET': prestart_web_reset,
        'PRESTART_VLLM_RESTART': prestart_vllm_restart,
        'SKIP_WEB_RESET': skip_web_reset,
        'WEB_RESET_TIMEOUT': web_reset_timeout,
        'VLLM_RESTART_BASE_URL': vllm_restart_base_url,
        'VLLM_RESTART_KEY': vllm_restart_key,
        'VLLM_RESTART_MODEL': vllm_restart_model,
        'VLLM_RESTART_TIMEOUT': vllm_restart_timeout,
        'STOP_TOKEN': stop_token,
        'RESULT_DIR': result_dir,
        'GUIDENESS_DIR': guideness_dir,
    }
    for key, value in mapping.items():
        print(f"{key}='{shell_quote(value)}'")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
