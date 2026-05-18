from __future__ import annotations

import json
from pathlib import Path

from scope.common.datasets import TaskRecord
from scope.common.io import load_json


def load_task_payload(task: TaskRecord) -> dict:
    if task.source_path is None:
        return {}
    path = task.source_path
    if path.is_dir():
        return _load_directory_payload(path)
    if path.suffix == ".json":
        data = load_json(path)
        return data if isinstance(data, dict) else {"data": data}
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return {"rows": rows}
    return {"text": path.read_text(encoding="utf-8", errors="replace")}


def _load_directory_payload(path: Path) -> dict:
    payload: dict[str, object] = {}
    for child in sorted(path.iterdir()):
        if child.suffix != ".json":
            continue
        try:
            payload[child.name] = load_json(child)
        except Exception:
            continue
    return payload


def payload_to_prompt_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
