from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileDefaults:
    profile_result_dir: str
    model_path: str
    instruction_path: str
    profile_stop_token: str
    full_reset_url: str
    full_status_url: str


PROFILE_DEFAULTS = {
    "webrl_qwen": ProfileDefaults(
        profile_result_dir="./results_qwen_base_sft_bs64_1e5_ep1",
        model_path="${DCRL_MODEL_PATH}",
        instruction_path="agent/prompts/jsons/p_webrl_qwen.json",
        profile_stop_token="<|eot_id|>",
        full_reset_url="http://localhost:7565/reset",
        full_status_url="http://localhost:7565/status",
    ),
    "webrl_llama": ProfileDefaults(
        profile_result_dir="./results_llama31_sft_bs128_1e5_ep3",
        model_path="${DCRL_MODEL_PATH}",
        instruction_path="agent/prompts/jsons/p_webrl.json",
        profile_stop_token="<|eot_id|>",
        full_reset_url="http://localhost:7565/reset",
        full_status_url="http://localhost:7565/status",
    ),
    "webrl": ProfileDefaults(
        profile_result_dir="./results_llama31_sft_bs128_1e5_ep3",
        model_path="${DCRL_MODEL_PATH}",
        instruction_path="agent/prompts/jsons/p_webrl.json",
        profile_stop_token="<|eot_id|>",
        full_reset_url="http://localhost:7565/reset",
        full_status_url="http://localhost:7565/status",
    ),
}


def shell_quote(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def main() -> int:
    profile = os.environ.get("PROFILE", "webrl_llama")
    defaults = PROFILE_DEFAULTS.get(profile)
    if defaults is None:
        supported = ", ".join(sorted(PROFILE_DEFAULTS))
        print(f"Unknown PROFILE: {profile}", file=sys.stderr)
        print(f"Supported: {supported}", file=sys.stderr)
        return 1

    mapping = {
        "PROFILE_RESULT_DIR": defaults.profile_result_dir,
        "MODEL_PATH": defaults.model_path,
        "INSTRUCTION_PATH": defaults.instruction_path,
        "PROFILE_STOP_TOKEN": defaults.profile_stop_token,
        "FULL_RESET_URL": defaults.full_reset_url,
        "FULL_STATUS_URL": defaults.full_status_url,
    }
    for key, default in mapping.items():
        value = os.environ.get(key, default)
        print(f"{key}='{shell_quote(value)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
