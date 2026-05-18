import json
import logging
import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import backoff
import openai
from openai import OpenAI

logger = logging.getLogger("evaluation")
GUIDED_RETRY_MAX_TRIES = 5
INSTRUCTION_RE = re.compile(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]", re.DOTALL)
WEBSHOP_ICL_PATH = Path(__file__).resolve().parents[1] / "prompt" / "icl_examples" / "webshop_icl.json"


GUIDED_PROMPT_TEMPLATE = """You are a WebShop action guide generator.
You are NOT starting from scratch. A reusable prefix has already been replayed, and you are now at that frontier state.
Your job is to use the matched failure memories to suggest a better immediate next step.

Return JSON only with exactly these three keys:
{{
  "rationale": "...",
  "thought": "...",
  "next_action": "..."
}}

Field roles:
- `rationale`: internal analysis for why the recommended next step is better. It may use matched failure memories, historical failure patterns, and hidden target-level signals.
- `thought`: a short natural next-step thought that the policy model itself could plausibly produce from the current page.
- `next_action`: exactly one valid WebShop action, either `search[...]` or `click[...]`.

Hard constraints for `thought`:
- Write it as ordinary WebShop reasoning grounded only in the current visible page state and options.
- Do NOT mention historical failures, matched memories, oracle knowledge, privileged information, hidden targets, or previous mistakes.
- Do NOT use control phrases such as `avoid`, `instead`, `previous failure`, `matched failure`, `historical`, `privileged`, `target product`, `go back`, `I accidentally`, or `I made an invalid action`.
- Do NOT reveal information that would not be directly inferable from the current page.
- Keep it short, natural, and suitable for later policy distillation.

Action preference constraints:
- Prefer the minimal change that stays close to the original intent while avoiding the known failed continuation.
- Use the current trajectory history, the matched failures, and the original failed behavior together.
- The `next_action` should be consistent with the `thought`.

Task Goal:
{task_goal}

Current Trajectory History After Prefix Replay:
{history_text}

Matched Failure Memories:
{matched_failures_text}

Original Failed Behavior Continuation For This Prefix:
{original_failed_continuation}
"""


def _read_json(path: Path) -> Any:
    return json.load(open(path))


def _read_jsonl_last(path: Path) -> Dict[str, Any]:
    lines = [line.strip() for line in open(path) if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def extract_instruction(text: str) -> str:
    match = INSTRUCTION_RE.search(str(text or ""))
    if match:
        return match.group(1).strip()
    return str(text or "").strip()


@lru_cache(maxsize=1)
def load_webshop_icl_examples() -> List[List[Dict[str, Any]]]:
    if not WEBSHOP_ICL_PATH.exists():
        return []
    try:
        data = json.loads(WEBSHOP_ICL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[List[Dict[str, Any]]] = []
    for item in data:
        if isinstance(item, list):
            out.append([msg for msg in item if isinstance(msg, dict)])
    return out


def _message_matches(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> bool:
    return (
        str(lhs.get("role") or "") == str(rhs.get("role") or "")
        and str(lhs.get("content") or "") == str(rhs.get("content") or "")
    )


def detect_known_icl_end_index(messages: List[Dict[str, str]]) -> int:
    idx = 2
    examples = load_webshop_icl_examples()
    if len(messages) <= idx or not examples:
        return idx
    for example in examples:
        if not example:
            continue
        end = idx + len(example)
        if end > len(messages):
            break
        segment = messages[idx:end]
        if all(
            isinstance(msg, dict) and _message_matches(msg, ref)
            for msg, ref in zip(segment, example)
        ):
            idx = end
        else:
            break
    return idx


def trim_guided_history(messages: List[Dict[str, str]], task_goal: str) -> List[Dict[str, str]]:
    if not messages:
        return []
    default_start = min(detect_known_icl_end_index(messages), len(messages) - 1)
    goal = str(task_goal or "").strip()
    if not goal:
        return list(messages[default_start:])

    matched_indices: List[int] = []
    for idx, msg in enumerate(messages):
        if str(msg.get("role") or "") != "user":
            continue
        if extract_instruction(msg.get("content", "")) == goal:
            matched_indices.append(idx)

    if not matched_indices:
        return list(messages[default_start:])
    for idx in matched_indices:
        if idx >= default_start:
            return list(messages[idx:])
    return list(messages[matched_indices[0]:])


def stringify_history(messages: List[Dict[str, str]]) -> str:
    rows = []
    for idx, msg in enumerate(messages):
        rows.append(f"[{idx}] {msg.get('role', '').upper()}:\n{msg.get('content', '')}")
    return "\n\n".join(rows)


def get_last_user_message_content(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def load_dcrl_task_artifacts(prefix_root: str, task_id: Any) -> Dict[str, Any]:
    task_dir = Path(str(prefix_root)) / str(task_id)
    selected_anchor = _read_jsonl_last(task_dir / "selected_anchor.jsonl") if (task_dir / "selected_anchor.jsonl").exists() else {}
    prefix_memory = _read_json(task_dir / "prefix_memory.json") if (task_dir / "prefix_memory.json").exists() else {}
    failure_memory = _read_json(task_dir / "failure_memory.json") if (task_dir / "failure_memory.json").exists() else {}
    parsed_trace = _read_json(task_dir / "parsed_trace.json") if (task_dir / "parsed_trace.json").exists() else {}
    return {
        "task_dir": str(task_dir),
        "selected_anchor": selected_anchor,
        "prefix_memory": prefix_memory,
        "failure_memory": failure_memory,
        "parsed_trace": parsed_trace,
    }


def get_selected_prefix(prefix_memory: Dict[str, Any], selected_prefix_id: str) -> Dict[str, Any]:
    for prefix in list(prefix_memory.get("prefixes") or []):
        if str(prefix.get("prefix_id")) == str(selected_prefix_id):
            return deepcopy(prefix)
    return {}


def collect_prefix_actions(selected_prefix: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in list(selected_prefix.get("prefix_actions") or []):
        action = str(item.get("action") or "").strip()
        if action:
            out.append(action)
    for item in list(selected_prefix.get("actions") or []):
        action = str(item.get("action") or "").strip()
        if action:
            out.append(action)
    return out


def build_replay_output(action: str, thought: str = "Replay the selected prefix action.") -> str:
    return f"Thought: {thought}\nAction: {action}"


def replay_prefix_actions(env, selected_prefix: Dict[str, Any]) -> int:
    actions = collect_prefix_actions(selected_prefix)
    for action in actions:
        env.step(build_replay_output(action))
    return len(actions)


def collect_failure_candidates(failure_memory: Dict[str, Any], selected_prefix: Dict[str, Any]) -> List[Dict[str, Any]]:
    prefix_signature = collect_prefix_actions(selected_prefix)
    matches = []
    for item in list(failure_memory.get("failures") or []):
        sig = [str(x) for x in list(item.get("prefix_signature") or [])]
        if sig == prefix_signature:
            matches.append(deepcopy(item))
    if matches:
        return matches
    return [deepcopy(item) for item in list(failure_memory.get("failures") or [])[:3]]


def build_matched_failures_text(failure_candidates: List[Dict[str, Any]]) -> str:
    if not failure_candidates:
        return "None"
    chunks = []
    for idx, item in enumerate(failure_candidates, start=1):
        summary = str(item.get("failure_summary") or "").strip()
        chunks.append(f"[Failure {idx}]\nFailure summary: {summary}")
    return "\n\n".join(chunks)


def get_original_failed_behavior_continuation(failure_candidates: List[Dict[str, Any]]) -> str:
    for item in failure_candidates:
        for cmu in list(item.get("failure_cmus") or []):
            actions = list(cmu.get("actions") or [])
            if actions:
                act = str(actions[0].get("action") or "").strip()
                if act:
                    return act
    return "N/A"


def build_guided_prompt(task_goal: str, history_text: str, matched_failures_text: str, original_failed_continuation: str) -> str:
    return GUIDED_PROMPT_TEMPLATE.format(
        task_goal=task_goal,
        history_text=history_text,
        matched_failures_text=matched_failures_text,
        original_failed_continuation=original_failed_continuation,
    )


def parse_guided_response(raw_response: str) -> Dict[str, str]:
    text = str(raw_response or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    obj = json.loads(text)
    return {
        "rationale": str(obj.get("rationale", "")).strip(),
        "thought": str(obj.get("thought", "")).strip(),
        "next_action": str(obj.get("next_action", "")).strip(),
    }


def call_guided_think(
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    task_goal: str,
    history_messages: List[Dict[str, str]],
    failure_candidates: List[Dict[str, Any]],
    original_failed_continuation: str,
    timeout: float | None = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    trimmed_history_messages = trim_guided_history(history_messages, task_goal)
    matched_failures_text = build_matched_failures_text(failure_candidates)
    prompt = build_guided_prompt(
        task_goal=task_goal,
        history_text=stringify_history(trimmed_history_messages),
        matched_failures_text=matched_failures_text,
        original_failed_continuation=original_failed_continuation,
    )
    client = OpenAI(base_url=api_base or None, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=timeout)
    response = _request_guided_completion(client, model_name=model_name, prompt=prompt)
    raw_response = response.choices[0].message.content or ""
    parsed = parse_guided_response(raw_response)
    trace = {
        "prompt": prompt,
        "raw_response": raw_response,
        "matched_failures_text": matched_failures_text,
    }
    return parsed, trace


@backoff.on_exception(
    backoff.fibo,
    (
        openai.APIError,
        openai.Timeout,
        openai.RateLimitError,
        openai.APIConnectionError,
    ),
    max_tries=GUIDED_RETRY_MAX_TRIES,
)
def _request_guided_completion(client: OpenAI, *, model_name: str, prompt: str):
    try:
        return client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )
    except openai.BadRequestError:
        return client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=1024,
        )


def format_guided_replay_output(guided_result: Dict[str, Any]) -> str:
    thought = str(guided_result.get("thought") or guided_result.get("rationale") or "").strip()
    next_action = str(guided_result.get("next_action") or "").strip()
    return f"Thought: {thought}\nAction: {next_action}"


def extract_action_from_output(llm_output: str) -> str:
    text = str(llm_output or "").strip()
    marker = "Action:"
    idx = text.rfind(marker)
    if idx == -1:
        return ""
    return text[idx + len(marker):].strip()


def _normalize_clickable(value: str) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def is_action_executable_for_current_page(env, llm_output: str) -> bool:
    action = extract_action_from_output(llm_output)
    if not action:
        return False
    if action.startswith("search[") and action.endswith("]"):
        return bool(action[len("search["):-1].strip())
    if action.startswith("click[") and action.endswith("]"):
        value = action[len("click["):-1].strip()
        if not value:
            return False
        try:
            available = env.env.get_available_actions()
        except Exception:
            return True
        clickables = []
        if isinstance(available, dict):
            clickables = list(available.get("clickables") or [])
        elif isinstance(available, (list, tuple)):
            clickables = list(available)
        normalized = _normalize_clickable(value)
        return normalized in {_normalize_clickable(item) for item in clickables}
    return False
