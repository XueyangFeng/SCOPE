from copy import deepcopy
from typing import List, Dict, Any, Optional


class State:
    """Conversation state for environment interaction."""

    def __init__(
        self,
        reward: float = None,
        finished: bool = False,
        success: bool = False,
        terminate_reason: str = None,
        info: dict = None,
        goal: dict = None,
    ):
        self.history: List[Dict[str, Any]] = []
        self.reward: float = reward
        self.finished: bool = finished
        self.success: bool = success
        self.terminate_reason: str = terminate_reason
        self.error: Optional[str] = None
        self.steps = 0
        self.info = deepcopy(info) if info is not None else {}
        self.goal = deepcopy(goal) if goal is not None else {}
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.duration_seconds: Optional[float] = None

    @classmethod
    def load_json(cls, json_dict: Dict[str, Any]):
        state = cls()
        state.history = json_dict[:-1]
        info = json_dict[-1]
        state.reward = info.get("reward")
        state.finished = info.get("finished", False)
        state.success = info.get("success", False)
        state.terminate_reason = info.get("terminate_reason")
        state.error = info.get("error")
        state.steps = info.get("steps", 0)
        state.info = deepcopy(info.get("info") or {})
        state.goal = deepcopy(info.get("goal") or {})
        state.started_at = info.get("started_at")
        state.finished_at = info.get("finished_at")
        state.duration_seconds = info.get("duration_seconds")
        return state

    @property
    def empty(self):
        return len(self.history) == 0

    def to_dict(self) -> Dict[str, Any]:
        history = deepcopy(self.history)
        history.append({
            "steps": self.steps,
            "reward": self.reward,
            "finished": self.finished,
            "success": self.success,
            "terminate_reason": self.terminate_reason,
            "error": self.error,
            "info": deepcopy(self.info),
            "goal": deepcopy(self.goal),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
        })
        return history
