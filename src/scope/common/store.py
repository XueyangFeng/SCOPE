from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryStoreLayout:
    root: Path

    def task_dir(self, task_id: str) -> Path:
        return self.root / str(task_id)

    def prefix_memory_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "prefix_memory.json"

    def selected_anchor_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "selected_anchor.jsonl"

    def memory_update_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "memory_update.jsonl"

    def has_prefix_memory(self, task_id: str) -> bool:
        return self.prefix_memory_path(task_id).exists()

    def has_selected_anchor(self, task_id: str) -> bool:
        return self.selected_anchor_path(task_id).exists()
