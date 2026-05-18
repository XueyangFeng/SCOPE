from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scope.common.datasets import TaskRecord
from scope.common.io import load_jsonl, read_text
from scope.common.json_tools import parse_json_object
from scope.common.llm import ChatClient, LLMConfig
from scope.common.prompts import PromptBundle
from scope.common.render import render_template
from scope.common.store import MemoryStoreLayout
from scope.common.trajectory import load_task_payload, payload_to_prompt_text


def load_latest_selected_prefix(layout: MemoryStoreLayout, task: TaskRecord) -> dict[str, Any]:
    path = layout.selected_anchor_path(task.task_id)
    if not path.exists():
        return {"selected_prefix_id": "-1", "rationale": "No selected anchor exists; use root."}
    rows = load_jsonl(path)
    return rows[-1] if rows else {"selected_prefix_id": "-1", "rationale": "Empty selected anchor file; use root."}


def build_update_prompt(
    task: TaskRecord,
    *,
    prompts: PromptBundle,
    layout: MemoryStoreLayout,
    extra: dict[str, Any],
) -> tuple[Path, str]:
    payload = load_task_payload(task)
    template = read_text(prompts.update)
    values = {
        "goal": extra.get("goal", payload.get("goal", "")),
        "privileged_info": json.dumps(extra.get("privileged_info", payload.get("privileged_info", {})), ensure_ascii=False),
        "task_success": str(task.success is True).lower(),
        "selected_prefix": json.dumps(load_latest_selected_prefix(layout, task), ensure_ascii=False, indent=2),
        "new_suffix_trajectory": payload_to_prompt_text(payload),
    }
    return prompts.update, render_template(template, values)


def update_memory_with_llm(
    task: TaskRecord,
    *,
    prompts: PromptBundle,
    layout: MemoryStoreLayout,
    llm_config: LLMConfig,
    extra: dict[str, Any],
) -> dict[str, Any]:
    prompt_path, prompt = build_update_prompt(task, prompts=prompts, layout=layout, extra=extra)
    raw = ChatClient(llm_config).complete_json(prompt)
    parsed = parse_json_object(raw)
    return {
        "prompt_template": str(prompt_path),
        "raw_output": raw,
        "parsed_output": parsed,
    }
