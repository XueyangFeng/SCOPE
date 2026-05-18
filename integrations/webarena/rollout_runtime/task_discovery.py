import json
import os
import sys
from pathlib import Path
from typing import Iterable


def _parse_sites(raw: str) -> list[str]:
    return [site.strip() for site in raw.split(',') if site.strip()]


def _load_task_sites(config_path: Path) -> list[str]:
    try:
        data = json.loads(config_path.read_text())
    except Exception:
        return []
    sites = data.get('sites')
    if isinstance(sites, list):
        return [str(x) for x in sites]
    if isinstance(sites, str):
        return [sites]
    return []


def _match_sites(task_sites: list[str], wanted_sites: list[str], mode: str) -> bool:
    if not wanted_sites:
        return True
    task_set = set(task_sites)
    wanted_set = set(wanted_sites)
    if mode == 'all':
        return wanted_set.issubset(task_set)
    return bool(task_set & wanted_set)


def _has_selected_node_for_run(prefix_root: Path, task_id: int, current_iter: int | None, current_run: int | None) -> bool:
    anchor_path = prefix_root / str(task_id) / 'selected_anchor.jsonl'
    if not anchor_path.exists():
        return False
    if current_iter is None or current_run is None:
        return False
    for line in anchor_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if item.get('source_iter') == current_iter and item.get('source_run') == current_run:
            return True
    return False


def collect_task_ids(
    base_dir: Path,
    sites: list[str],
    site_filter_mode: str,
    start: int,
    end: int,
    prefix_root: Path,
    only_selected_node_tasks: bool,
    current_iter: int | None,
    current_run: int | None,
) -> list[int]:
    task_ids: list[int] = []
    for path in base_dir.glob('*.json'):
        try:
            task_id = int(path.stem)
        except ValueError:
            continue
        if task_id < start or task_id >= end:
            continue
        if only_selected_node_tasks and not _has_selected_node_for_run(prefix_root, task_id, current_iter, current_run):
            continue
        task_sites = _load_task_sites(path)
        if not _match_sites(task_sites, sites, site_filter_mode):
            continue
        task_ids.append(task_id)
    task_ids.sort()
    return task_ids


def main() -> int:
    task_ids_raw = os.environ.get('TASK_IDS', '').strip()
    if task_ids_raw:
        for token in task_ids_raw.replace(',', ' ').split():
            print(int(token))
        return 0
    base_dir = Path(os.environ['TEST_CONFIG_BASE_DIR'])
    sites = _parse_sites(os.environ.get('SITES', ''))
    site_filter_mode = os.environ.get('SITE_FILTER_MODE', 'any').strip() or 'any'
    start = int(os.environ.get('TASK_RANGE_START', '0'))
    end = int(os.environ.get('TASK_RANGE_END', '1000000000'))
    prefix_root = Path(os.environ.get('PREFIX_ROOT', '.'))
    only_selected_node_tasks = os.environ.get('ONLY_SELECTED_NODE_TASKS', '0') == '1'
    current_iter = os.environ.get('CURRENT_ITER', '').strip()
    current_run = os.environ.get('CURRENT_RUN', '').strip()
    iter_value = int(current_iter) if current_iter else None
    run_value = int(current_run) if current_run else None
    for task_id in collect_task_ids(base_dir, sites, site_filter_mode, start, end, prefix_root, only_selected_node_tasks, iter_value, run_value):
        print(task_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
