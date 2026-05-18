#!/bin/bash
set -euo pipefail

if [[ -z "${__WA_BG_ACTIVE:-}" && "${RUN_IN_BACKGROUND:-1}" -eq 1 && -t 1 ]]; then
    __WA_BG_ACTIVE=1
    export __WA_BG_ACTIVE
    BG_LOG_DIR="${BG_LOG_DIR:-${RESULT_DIR:-.}/logs}"
    mkdir -p "${BG_LOG_DIR}"
    BG_LOG_FILE="${BG_LOG_DIR}/wa_parallel_run_yi_$(date +%Y%m%d_%H%M%S).out"
    nohup bash "$0" "$@" > "${BG_LOG_FILE}" 2>&1 &
    echo "Running in background (log: ${BG_LOG_FILE})"
    exit 0
fi

WRAPPER_CONFIG="${WRAPPER_CONFIG:-}"
if [[ -z "${WRAPPER_CONFIG}" && -n "${1:-}" && "${1:-}" != "--profile" ]]; then
    WRAPPER_CONFIG="$1"
    shift
fi
if [[ -n "${WRAPPER_CONFIG}" ]]; then
    wrapper_assignments="$(WRAPPER_CONFIG="${WRAPPER_CONFIG}" python -m rollout_runtime.wrapper_config)"
    if [[ -z "${wrapper_assignments}" ]]; then
        echo "Failed to load wrapper config: ${WRAPPER_CONFIG}" >&2
        exit 1
    fi
    eval "${wrapper_assignments}"
fi

PROFILE="${PROFILE:-webrl_llama}"
if [[ "${1:-}" == "--profile" ]]; then
    PROFILE="${2:-}"
    shift 2
fi

profile_assignments="$(PROFILE="${PROFILE}" python -m rollout_runtime.profile_config)"
if [[ -z "${profile_assignments}" ]]; then
    echo "Failed to load profile defaults for ${PROFILE}" >&2
    exit 1
fi
eval "${profile_assignments}"

env_assignments="$(PROFILE="${PROFILE}" MODEL_PATH="${MODEL_PATH:-}" INSTRUCTION_PATH="${INSTRUCTION_PATH:-}" PROFILE_STOP_TOKEN="${PROFILE_STOP_TOKEN:-}" FULL_RESET_URL="${FULL_RESET_URL:-}" FULL_STATUS_URL="${FULL_STATUS_URL:-}" RESULT_DIR="${RESULT_DIR:-}" STOP_TOKEN="${STOP_TOKEN:-}" python -m rollout_runtime.env_config)"
if [[ -z "${env_assignments}" ]]; then
    echo "Failed to resolve runtime environment" >&2
    exit 1
fi
set -a
eval "${env_assignments}"
set +a

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
if [[ -z "${CONDA_BASE}" ]]; then
    CONDA_BASE="${HOME}/miniconda3"
fi

PROXY_EXPORTS=""
if [[ -n "${PROXY_URL}" ]]; then
    PROXY_EXPORTS="export HTTP_PROXY='${PROXY_URL}'; export HTTPS_PROXY='${PROXY_URL}'; export ALL_PROXY='${PROXY_URL}'; "
else
    PROXY_EXPORTS="unset HTTP_PROXY HTTPS_PROXY ALL_PROXY; "
fi

GUIDENESS_DIR="${GUIDENESS_DIR:-${RESULT_DIR}/guideness}"
ENV_VARIABLES="${PROXY_EXPORTS}export NO_PROXY='localhost,127.0.0.1,0.0.0.0'; export DATASET='${DATASET}'; export SHOPPING='http://localhost:8082'; export SHOPPING_ADMIN='http://localhost:8083/admin'; export REDDIT='http://localhost:8080'; export GITLAB='http://localhost:9001'; export WIKIPEDIA='http://localhost:8081'; export MAP='https://www.openstreetmap.org'; export HOMEPAGE='http://localhost:4399'; export OPENAI_API_URL='${SERVER}'; export OPENAI_API_KEY='${OPENAI_API_KEY}'; export OPENAI_EVAL_API_URL='${OPENAI_EVAL_API_URL}'; export OPENAI_EVAL_API_KEY='${OPENAI_EVAL_API_KEY}'; export LLM_RAW_LOG_DIR='${LLM_RAW_LOG_DIR}'; export GUIDENESS_DIR='${GUIDENESS_DIR}'; export FULL_RESET='${FULL_RESET_URL}'; export FULL_STATUS_URL='${FULL_STATUS_URL}'; export FULL_READY_TEXT='${FULL_READY_TEXT}'"

LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${GUIDENESS_DIR}"

urlencode() {
    python - "$1" <<'PY'
import sys
from urllib.parse import quote
print(quote(sys.argv[1], safe=""))
PY
}

reset_web_backend() {
    local reset_url=$1
    local status_url=$2
    local ready_text=$3
    local timeout_s=$4
    local log_file=$5
    python -m rollout_runtime.preflight reset-web         --reset-url "${reset_url}"         --status-url "${status_url}"         --ready-text "${ready_text}"         --timeout "${timeout_s}"         --log-file "${log_file}"
}

vllm_restart_and_wait() {
    local log_file=$1
    python -m rollout_runtime.preflight restart-vllm --log-file "${log_file}"
}

run_single_task() {
    local task_id=$1
    local log_file=$2
    local prefix_json=""
    if [[ -n "${PREFIX_ROOT}" && "$(basename "${RUN_SCRIPT}")" == "run_with_prefix.py" ]]; then
        if ! prefix_json="$(
            TASK_ID="${task_id}"             PREFIX_ROOT="${PREFIX_ROOT}"             python -m rollout_runtime.prefix_resolver 2>>"${log_file}"
        )"; then
            echo "[prefix] failed to resolve prefix for task ${task_id}" >> "${log_file}"
            return 1
        fi
    fi

    local command=""
    if ! command="$(
        TASK_ID="${task_id}"         RUN_SCRIPT="${RUN_SCRIPT}"         INSTRUCTION_PATH="${INSTRUCTION_PATH}"         RESULT_DIR="${RESULT_DIR}"         TEST_CONFIG_BASE_DIR="${test_config_base_dir}"         PROVIDER="${provider}"         MODEL_PATH="${MODEL_PATH}"         MODE="${MODE}"         TEMPERATURE="${temperature}"         TOP_P="${top_p}"         MAX_TOKENS="${max_tokens}"         STOP_TOKEN="${STOP_TOKEN}"         PREFIX_VALIDATE_REPLAY="${PREFIX_VALIDATE_REPLAY}"         PREFIX_VALIDATE_ON_FAIL="${PREFIX_VALIDATE_ON_FAIL}"         PREFIX_REFERENCE_DIR="${PREFIX_REFERENCE_DIR}"         PREFIX_ALLOW_REPAIRABLE="${PREFIX_ALLOW_REPAIRABLE}"         PREFIX_ID_REPAIR_ENABLE="${PREFIX_ID_REPAIR_ENABLE}"         PREFIX_ID_REPAIR_MODEL="${PREFIX_ID_REPAIR_MODEL}"         PREFIX_ACTIONS_JSON="${prefix_json}"         python -m rollout_runtime.command_builder 2>>"${log_file}"
    )"; then
        echo "[command] failed to build command for task ${task_id}" >> "${log_file}"
        return 1
    fi

    local activate_cmd="conda activate ${CONDA_ENV_NAME}"
    if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        activate_cmd="source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate ${CONDA_ENV_NAME}"
    fi
    if (( TASK_TIMEOUT_SECONDS > 0 )); then
        if timeout --foreground -k "${TASK_TIMEOUT_KILL_AFTER_SECONDS}s" "${TASK_TIMEOUT_SECONDS}s"             env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy             bash -lc "${activate_cmd}; ${ENV_VARIABLES}; ${command}" >> "${log_file}" 2>&1; then
            return 0
        fi
        local task_rc=$?
        if (( task_rc == 124 )); then
            echo "[task-timeout] task ${task_id} exceeded ${TASK_TIMEOUT_SECONDS}s (kill after ${TASK_TIMEOUT_KILL_AFTER_SECONDS}s)" >> "${log_file}"
        fi
        return "${task_rc}"
    fi

    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy         bash -lc "${activate_cmd}; ${ENV_VARIABLES}; ${command}" >> "${log_file}" 2>&1
}

run_chunk() {
    local id_list=$1
    local ts first_id last_id log_file
    ts=$(date +"%Y%m%d_%H%M%S")
    first_id=$(echo "${id_list}" | awk '{print $1}')
    last_id=$(echo "${id_list}" | awk '{print $NF}')
    log_file="${LOG_DIR}/job_${first_id}_${last_id}_${ts}.log"
    echo "job start: ${id_list}" >> "${log_file}"
    for task_id in ${id_list}; do
        run_single_task "${task_id}" "${log_file}"
    done
}

collect_task_ids() {
    if [[ -n "${TASK_IDS}" ]]; then
        echo "${TASK_IDS}" | tr "," " " | tr -s " " "\n" | sed "/^$/d"
        return 0
    fi
    TEST_CONFIG_BASE_DIR="${test_config_base_dir}" \
    SITES="${SITES}" \
    SITE_FILTER_MODE="${SITE_FILTER_MODE}" \
    TASK_RANGE_START="${TASK_RANGE_START:-}" \
    TASK_RANGE_END="${TASK_RANGE_END:-}" \
    PREFIX_ROOT="${PREFIX_ROOT}" \
    ONLY_SELECTED_NODE_TASKS="${ONLY_SELECTED_NODE_TASKS}" \
    python -m rollout_runtime.task_discovery
}

mapfile -t TASK_ID_LIST < <(collect_task_ids)
if [[ ${#TASK_ID_LIST[@]} -eq 0 ]]; then
    echo "No tasks matched. Check TASK_IDS, SITES, or TASK_RANGE_START/END." >&2
    exit 1
fi

TOTAL_TASKS=${#TASK_ID_LIST[@]}
if (( CHUNK_SIZE <= 0 )); then
    CHUNK_SIZE=$(( (TOTAL_TASKS + PARALLELISM - 1) / PARALLELISM ))
fi

chunks=()
for ((i=0; i<${#TASK_ID_LIST[@]}; i+=CHUNK_SIZE)); do
    chunk="${TASK_ID_LIST[*]:i:CHUNK_SIZE}"
    chunks+=("${chunk}")
done

SHOULD_RESET_WEB=1
if [[ "${PRESTART_WEB_RESET}" == "1" || "${PRESTART_VLLM_RESTART}" == "1" ]]; then
    PRESTART_LOG_FILE="${LOG_DIR}/prestart_$(date +%Y%m%d_%H%M%S).log"
    echo "prestart begin: ${PRESTART_LOG_FILE}"

    pids=()
    web_started=0
    if [[ "${PRESTART_WEB_RESET}" == "1" ]]; then
        reset_web_backend "${FULL_RESET_URL}" "${FULL_STATUS_URL}" "${FULL_READY_TEXT}" "${WEB_RESET_TIMEOUT}" "${PRESTART_LOG_FILE}" &
        pids+=("$!")
        web_started=1
    fi
    if [[ "${PRESTART_VLLM_RESTART}" == "1" ]]; then
        vllm_restart_and_wait "${PRESTART_LOG_FILE}" &
        pids+=("$!")
    fi

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if (( failed )); then
        echo "prestart failed, check log: ${PRESTART_LOG_FILE}" >&2
        exit 1
    fi
    echo "prestart done: ${PRESTART_LOG_FILE}"

    if (( web_started )); then
        SHOULD_RESET_WEB=0
    fi
fi

if (( SHOULD_RESET_WEB )) && [[ "${SKIP_WEB_RESET}" != "1" ]]; then
    RESET_LOG_FILE="${LOG_DIR}/reset_$(date +%Y%m%d_%H%M%S).log"
    reset_web_backend "${FULL_RESET_URL}" "${FULL_STATUS_URL}" "${FULL_READY_TEXT}" "${FULL_RESET_TIMEOUT}" "${RESET_LOG_FILE}"
fi

active=0
for chunk in "${chunks[@]}"; do
    run_chunk "${chunk}" &
    active=$((active + 1))
    if (( active >= PARALLELISM )); then
        wait -n
        active=$((active - 1))
    fi
done
wait

if (( RETRY_ERRORS )); then
    while ! python scripts/check_error_runs.py "${RESULT_DIR}" --delete_errors --tolerance "${TOLERANCE}"; do
        echo "Check failed, rerunning all matched tasks..."
        active=0
        for chunk in "${chunks[@]}"; do
            run_chunk "${chunk}" &
            active=$((active + 1))
            if (( active >= PARALLELISM )); then
                wait -n
                active=$((active - 1))
            fi
        done
        wait
    done
else
    python scripts/check_error_runs.py "${RESULT_DIR}" --delete_errors --tolerance "${TOLERANCE}"
fi

# Merge dcrl_debug into one jsonl (one record per task, final step only).
MERGE_DCRL_DEBUG="${MERGE_DCRL_DEBUG:-1}"
DCRL_DEBUG_DIR="${DCRL_DEBUG_DIR:-${RESULT_DIR}/dcrl_debug}"
DCRL_DEBUG_MERGED_DIR="${DCRL_DEBUG_MERGED_DIR:-${RESULT_DIR}/dcrl_debug_merged}"
DCRL_DEBUG_MERGED_FILE="${DCRL_DEBUG_MERGED_FILE:-${DCRL_DEBUG_MERGED_DIR}/merged_io.jsonl}"

if [[ "${MERGE_DCRL_DEBUG}" == "1" ]]; then
    if [[ -d "${DCRL_DEBUG_DIR}" ]] && ls "${DCRL_DEBUG_DIR}"/task_*.jsonl >/dev/null 2>&1; then
        mkdir -p "${DCRL_DEBUG_MERGED_DIR}"
        python scripts/merge_dcrl_debug_io.py "${DCRL_DEBUG_DIR}" --output "${DCRL_DEBUG_MERGED_FILE}"
        echo "merged dcrl_debug: ${DCRL_DEBUG_MERGED_FILE}"
    else
        echo "skip dcrl_debug merge (not found): ${DCRL_DEBUG_DIR}"
    fi
fi
