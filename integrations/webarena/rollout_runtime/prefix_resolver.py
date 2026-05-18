import json
import os
import sys
from pathlib import Path
from typing import Any


def _action_strings(items: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get('action')
        if action:
            values.append(str(action))
    return values


def _load_selected_prefix_id(task_dir: Path, current_iter: int | None, current_run: int | None) -> str | None:
    anchor_path = task_dir / 'selected_anchor.jsonl'
    if not anchor_path.exists():
        return None
    if current_iter is None or current_run is None:
        return None
    chosen: str | None = None
    for line in anchor_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if item.get('source_iter') != current_iter or item.get('source_run') != current_run:
            continue
        value = item.get('selected_prefix_id')
        if value:
            chosen = str(value)
    return chosen


def load_selected_prefix_actions(prefix_root: Path, task_id: int, current_iter: int | None, current_run: int | None) -> list[str]:
    task_dir = prefix_root / str(task_id)
    prefix_id = _load_selected_prefix_id(task_dir, current_iter, current_run)
    if not prefix_id:
        return []
    prefix_path = task_dir / 'prefix_memory.json'
    if not prefix_path.exists():
        return []
    data = json.loads(prefix_path.read_text())
    for prefix in data.get('prefixes', []):
        if isinstance(prefix, dict) and prefix.get('prefix_id') == prefix_id:
            return _action_strings(prefix.get('prefix_actions', [])) + _action_strings(prefix.get('actions', []))
    return []


def main() -> int:
    prefix_root = Path(os.environ['PREFIX_ROOT'])
    task_id = int(os.environ['TASK_ID'])
    current_iter = os.environ.get('CURRENT_ITER', '').strip()
    current_run = os.environ.get('CURRENT_RUN', '').strip()
    iter_value = int(current_iter) if current_iter else None
    run_value = int(current_run) if current_run else None
    actions = load_selected_prefix_actions(prefix_root, task_id, iter_value, run_value)
    sys.stdout.write(json.dumps(actions, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
