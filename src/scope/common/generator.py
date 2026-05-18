from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scope.common.datasets import TaskRecord
from scope.common.io import read_text
from scope.common.json_tools import parse_json_object
from scope.common.llm import ChatClient, LLMConfig
from scope.common.prompts import PromptBundle
from scope.common.render import render_template
from scope.common.trajectory import load_task_payload, payload_to_prompt_text


def build_generator_prompt(task: TaskRecord, *, prompts: PromptBundle, extra: dict[str, Any]) -> tuple[Path, str]:
    payload = load_task_payload(task)
    template = read_text(prompts.generator)
    values = {
        "goal": extra.get("goal", payload.get("goal", "")),
        "privileged_info": json.dumps(extra.get("privileged_info", payload.get("privileged_info", {})), ensure_ascii=False),
        "task_success": str(task.success is True).lower(),
        "reward": task.score if task.score is not None else payload.get("reward", ""),
        "m_history": json.dumps(extra.get("m_history", []), ensure_ascii=False),
        "trajectory_data": payload_to_prompt_text(payload),
    }
    return prompts.generator, render_template(template, values)


def generate_memory_with_llm(
    task: TaskRecord,
    *,
    prompts: PromptBundle,
    llm_config: LLMConfig,
    extra: dict[str, Any],
) -> dict[str, Any]:
    prompt_path, prompt = build_generator_prompt(task, prompts=prompts, extra=extra)
    raw = ChatClient(llm_config).complete_json(prompt)
    parsed = parse_json_object(raw)
    return {
        "prompt_template": str(prompt_path),
        "raw_output": raw,
        "parsed_output": parsed,
    }
