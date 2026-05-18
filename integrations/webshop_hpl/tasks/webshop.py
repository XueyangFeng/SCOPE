import json
import logging
import random
from typing import Iterable, Tuple, Any

logger = logging.getLogger("agent_eval")

from tasks.base import Task


class WebShopTask(Task):
    task_name = "webshop"

    def __init__(
        self,
        session_id: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_id = session_id
    
    @classmethod
    def load_tasks(cls, path: str, split: str, part_num: int, part_idx: int = -1) -> Tuple[Iterable[Task], int]:
        if split == 'train':
            idxs = json.load(open(f"{path}/train_indices.json"))
        else:
            idxs = json.load(open(f"{path}/test_indices.json"))
        if part_num == 1:
            idxs = idxs
        else:
            assert part_idx != -1
            part_len = len(idxs) // part_num + 1
            idxs = idxs[part_len * part_idx: part_len * (part_idx + 1)]
        N_TASKS = len(idxs)
        def generator():
            for idx in idxs:
                session_id = idx
                yield cls(
                    task_id=idx,
                    session_id=session_id,
                )

        return generator(), N_TASKS
    