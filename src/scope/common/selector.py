from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scope.common.datasets import TaskRecord
from scope.common.io import load_json, read_text
from scope.common.json_tools import parse_json_object
from scope.common.llm import ChatClient, LLMConfig
from scope.common.prompts import PromptBundle
from scope.common.render import render_template
from scope.common.store import MemoryStoreLayout


def load_prefix_candidates(layout: MemoryStoreLayout, task: TaskRecord) -> list[dict[str, Any]]:
    path = layout.prefix_memory_path(task.task_id)
    if not path.exists():
        return [{"prefix_id": "-1", "skill_summary": "Root empty prefix", "select_count": 0}]
    data = load_json(path)
    candidates = data.get("prefix_memory") or data.get("memories") or []
    if not isinstance(candidates, list) or not candidates:
        candidates = [{"prefix_id": "-1", "skill_summary": "Root empty prefix", "select_count": 0}]
    normalized = []
    for idx, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        out = dict(item)
        out.setdefault("prefix_id", out.get("id", f"p{idx}"))
        out.setdefault("select_count", 0)
        normalized.append(out)
    return normalized or [{"prefix_id": "-1", "skill_summary": "Root empty prefix", "select_count": 0}]


def build_selector_prompt(
    task: TaskRecord,
    *,
    prompts: PromptBundle,
    layout: MemoryStoreLayout,
    extra: dict[str, Any],
) -> tuple[Path, str]:
    template = read_text(prompts.selector)
    values = {
        "goal": extra.get("goal", ""),
        "privileged_info": json.dumps(extra.get("privileged_info", {}), ensure_ascii=False),
        "cmu_trajectory_json": json.dumps(load_prefix_candidates(layout, task), ensure_ascii=False, indent=2),
    }
    return prompts.selector, render_template(template, values)


def select_anchor_with_llm(
    task: TaskRecord,
    *,
    prompts: PromptBundle,
    layout: MemoryStoreLayout,
    llm_config: LLMConfig,
    extra: dict[str, Any],
) -> dict[str, Any]:
    prompt_path, prompt = build_selector_prompt(task, prompts=prompts, layout=layout, extra=extra)
    raw = ChatClient(llm_config).complete_json(prompt)
    parsed = parse_json_object(raw)
    return {
        "prompt_template": str(prompt_path),
        "raw_output": raw,
        "parsed_output": parsed,
    }
