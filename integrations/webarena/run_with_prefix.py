"""Run evaluation with optional prefix replay support."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

import cv2
import openai
import requests
import torch
from PIL import Image

import run as base_run
from agent import PromptAgent, construct_agent
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_id_based_action,
    create_playwright_action,
    create_stop_action,
)
from browser_env.actions import create_webrl_id_based_action
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import RenderHelper, get_action_description
from evaluation_harness import evaluator_router, image_utils

_PREFIX_REPAIR_CLIENT = None


def config_with_prefix() -> argparse.Namespace:
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument(
        "--prefix_actions",
        type=str,
        default="",
        help="Action prefix to replay after reset. Accepts a JSON list or newline-separated actions.",
    )
    prefix_parser.add_argument(
        "--prefix_state_path",
        type=str,
        default="",
        help="Path to a saved state_snapshots/<task_id>/prefix.json file.",
    )
    prefix_parser.add_argument(
        "--prefix_state_step",
        type=int,
        default=None,
        help="Step index to load from prefix_state_path. Defaults to the last record.",
    )
    prefix_parser.add_argument(
        "--prefix_reference_dir",
        type=str,
        default="",
        help="Reference rollout dir containing state_snapshots/<task_id>/prefix.json for replay validation.",
    )
    prefix_parser.add_argument(
        "--prefix_validate_replay",
        action="store_true",
        help="Validate replay state after prefix execution using object+URL consistency rules.",
    )
    prefix_parser.add_argument(
        "--prefix_validate_on_fail",
        type=str,
        default="restart",
        choices=["skip", "raise", "restart"],
        help="Behavior when replay validation fails: skip task, raise exception, or restart task from scratch.",
    )
    prefix_parser.add_argument(
        "--prefix_allow_repairable",
        action="store_true",
        default=True,
        help="If set, repairable URL drift is treated as pass; otherwise treated as fail (no auto-fix yet).",
    )
    prefix_parser.add_argument(
        "--prefix_disallow_repairable",
        action="store_false",
        dest="prefix_allow_repairable",
        help="Disable allowing repairable URL drift; treat repairable as fail.",
    )
    prefix_parser.add_argument(
        "--prefix_id_repair_enable",
        action="store_true",
        help="If set, attempt model-based element-id repair during prefix replay when semantic check fails.",
    )
    prefix_parser.add_argument(
        "--prefix_id_repair_model",
        type=str,
        default="gpt-5-mini-ca",
        help="Model name for prefix element-id repair.",
    )
    prefix_args, remaining = prefix_parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = base_run.config()
    finally:
        sys.argv = original_argv

    args.prefix_actions = prefix_args.prefix_actions
    args.prefix_state_path = prefix_args.prefix_state_path
    args.prefix_state_step = prefix_args.prefix_state_step
    args.prefix_reference_dir = prefix_args.prefix_reference_dir
    args.prefix_validate_replay = prefix_args.prefix_validate_replay
    args.prefix_validate_on_fail = prefix_args.prefix_validate_on_fail
    args.prefix_allow_repairable = prefix_args.prefix_allow_repairable
    args.prefix_id_repair_enable = prefix_args.prefix_id_repair_enable
    args.prefix_id_repair_model = prefix_args.prefix_id_repair_model
    return args


def _parse_prefix_actions(raw_value: str) -> list[str]:
    raw_value = raw_value.strip()
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        raw_value = parsed

    actions: list[str] = []
    pending_comment_lines: list[str] = []
    for raw_line in raw_value.splitlines():
        line = raw_line.strip()
        if not line:
            pending_comment_lines = []
            continue
        if line.startswith("#"):
            pending_comment_lines.append(line)
            continue
        if pending_comment_lines:
            actions.append("\n".join(pending_comment_lines + [line]))
            pending_comment_lines = []
        else:
            actions.append(line)
    if pending_comment_lines:
        actions.extend(pending_comment_lines)
    return actions


def _load_prefix_actions_from_state(
    prefix_state_path: str, prefix_state_step: int | None, task_id: int
) -> list[str]:
    with open(prefix_state_path, "r") as f:
        records = json.load(f)
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Invalid prefix state file: {prefix_state_path}")

    selected_record = None
    if prefix_state_step is None:
        selected_record = records[-1]
    else:
        for record in records:
            if (
                isinstance(record, dict)
                and int(record.get("step", -999999)) == prefix_state_step
            ):
                selected_record = record
                break
        if selected_record is None:
            raise ValueError(
                f"Step {prefix_state_step} not found in prefix state file {prefix_state_path}"
            )

    if not isinstance(selected_record, dict):
        raise ValueError(f"Invalid prefix record in {prefix_state_path}")

    record_task_id = selected_record.get("task_id")
    if record_task_id is not None and int(record_task_id) != task_id:
        base_run.logger.warning(
            "Prefix state task_id %s does not match current task_id %s",
            record_task_id,
            task_id,
        )

    prefix_actions = selected_record.get("prefix_actions", [])
    if not isinstance(prefix_actions, list):
        raise ValueError(f"Invalid prefix_actions in {prefix_state_path}")
    return [str(item).strip() for item in prefix_actions if str(item).strip()]


def resolve_prefix_actions(args: argparse.Namespace, task_id: int) -> list[str]:
    if args.prefix_actions and args.prefix_state_path:
        raise ValueError("Use either --prefix_actions or --prefix_state_path, not both.")
    if args.prefix_actions:
        return _parse_prefix_actions(args.prefix_actions)
    if args.prefix_state_path:
        return _load_prefix_actions_from_state(
            args.prefix_state_path, args.prefix_state_step, task_id
        )
    return []


def _action_strings(items: list[dict]) -> list[str]:
    values: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if action:
            values.append(str(action).strip())
    return [value for value in values if value]


def _load_selected_anchor_record(
    task_dir: Path, current_iter: int | None, current_run: int | None
) -> dict:
    anchor_path = task_dir / "selected_anchor.jsonl"
    if not anchor_path.exists():
        return {}
    if current_iter is None or current_run is None:
        return {}
    matched_record: dict = {}
    for raw_line in anchor_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        selected_prefix_id = str(item.get("selected_prefix_id", "")).strip()
        if not selected_prefix_id:
            continue
        if item.get("source_iter") == current_iter and item.get("source_run") == current_run:
            matched_record = item

    return matched_record


def load_prefix_runtime_context(task_id: int) -> dict:
    prefix_root = os.environ.get("PREFIX_ROOT", "").strip()
    if not prefix_root:
        return {}

    task_dir = Path(prefix_root) / str(task_id)
    if not task_dir.exists():
        return {}

    current_iter_text = os.environ.get("CURRENT_ITER", "").strip()
    current_run_text = os.environ.get("CURRENT_RUN", "").strip()
    current_iter = int(current_iter_text) if current_iter_text else None
    current_run = int(current_run_text) if current_run_text else None

    selected_anchor = _load_selected_anchor_record(task_dir, current_iter, current_run)
    selected_prefix_id = str(selected_anchor.get("selected_prefix_id", "")).strip()
    if not selected_prefix_id:
        return {}

    prefix_path = task_dir / "prefix_memory.json"
    if not prefix_path.exists():
        return {}
    try:
        prefix_memory = json.loads(prefix_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(prefix_memory, dict):
        return {}

    selected_prefix: dict = {}
    for item in prefix_memory.get("prefixes", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("prefix_id", "")).strip() == selected_prefix_id:
            selected_prefix = item
            break
    if not selected_prefix:
        return {}

    selected_prefix_signature = _action_strings(
        selected_prefix.get("prefix_actions", [])
    ) + _action_strings(selected_prefix.get("actions", []))

    failure_candidates: list[dict] = []
    failure_path = task_dir / "failure_memory.json"
    if failure_path.exists():
        try:
            failure_memory = json.loads(failure_path.read_text(encoding="utf-8"))
        except Exception:
            failure_memory = {}
        if isinstance(failure_memory, dict):
            for item in failure_memory.get("failures", []):
                if not isinstance(item, dict):
                    continue
                prefix_signature_raw = item.get("prefix_signature", [])
                if not isinstance(prefix_signature_raw, list):
                    continue
                prefix_signature = [
                    str(action).strip() for action in prefix_signature_raw if str(action).strip()
                ]
                current_len = len(selected_prefix_signature)
                if current_len > len(prefix_signature):
                    continue
                if prefix_signature[:current_len] != selected_prefix_signature:
                    continue

                failure_path_suffix = []
                if len(prefix_signature) > current_len:
                    failure_path_suffix = prefix_signature[current_len:]
                first_failed_action = ""
                failure_cmus = item.get("failure_cmus", [])
                reduced_cmus: list[dict] = []
                if isinstance(failure_cmus, list):
                    for cmu in failure_cmus:
                        if not isinstance(cmu, dict):
                            continue
                        actions = _action_strings(cmu.get("actions", []))
                        reduced_cmus.append(
                            {
                                "skill_summary": str(cmu.get("skill_summary", "")).strip(),
                                "status_summary": str(cmu.get("status_summary", "")).strip(),
                                "actions": actions,
                            }
                        )
                        if not first_failed_action and actions:
                            first_failed_action = actions[0]

                failure_candidates.append(
                    {
                        "source_iter": item.get("source_iter"),
                        "source_run": item.get("source_run"),
                        "prefix_signature": prefix_signature,
                        "failure_path_suffix": failure_path_suffix,
                        "failure_summary": str(item.get("failure_summary", "")).strip(),
                        "first_failed_action": first_failed_action.strip(),
                        "failure_cmus": reduced_cmus,
                    }
                )

    def _candidate_sort_key(candidate: dict) -> tuple[int, int, int, int]:
        source_iter = candidate.get("source_iter")
        source_run = candidate.get("source_run")
        exact_match = int(source_iter == current_iter and source_run == current_run)
        prefix_signature = candidate.get("prefix_signature", [])
        delta = len(prefix_signature) - len(selected_prefix_signature)
        safe_delta = delta if delta >= 0 else 10**9
        source_iter_num = int(source_iter) if isinstance(source_iter, int) else -1
        source_run_num = int(source_run) if isinstance(source_run, int) else -1
        return (exact_match, -safe_delta, source_iter_num, source_run_num)

    failure_candidates.sort(key=_candidate_sort_key, reverse=True)

    return {
        "task_id": task_id,
        "selected_prefix_id": selected_prefix_id,
        "selected_anchor_rationale": str(selected_anchor.get("rationale", "")).strip(),
        "selected_anchor_source_iter": selected_anchor.get("source_iter"),
        "selected_anchor_source_run": selected_anchor.get("source_run"),
        "selected_prefix_parent_id": selected_prefix.get("parent_prefix_id"),
        "selected_prefix_skill_summary": str(
            selected_prefix.get("skill_summary", "")
        ).strip(),
        "selected_prefix_status_summary": str(
            selected_prefix.get("status_summary", "")
        ).strip(),
        "selected_prefix_actions": list(selected_prefix_signature),
        "selected_prefix_signature": list(selected_prefix_signature),
        "failure_candidates": failure_candidates[:3],
    }


def parse_action_for_replay(action_str: str, action_set_tag: str) -> Action:
    if action_set_tag == "webrl_id":
        action = create_webrl_id_based_action(action_str)
    elif action_set_tag == "id_accessibility_tree":
        action = create_id_based_action(action_str)
    else:
        action = create_playwright_action(action_str)
    action["raw_prediction"] = action_str
    action["parsed_action"] = action_str
    return action


def _extract_action_element_id(action_str: str) -> str:
    m = re.search(r'element\s*=\s*"([^"]+)"', action_str)
    if m:
        return m.group(1).strip()
    m = re.search(r"element\s*=\s*'([^']+)'", action_str)
    if m:
        return m.group(1).strip()
    m = re.search(r"element\s*=\s*([0-9]+)", action_str)
    if m:
        return m.group(1).strip()
    return ""


def _extract_expected_element_hint(action_str: str) -> str:
    for line in action_str.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        m = re.search(r"#\s*Element\s*:\s*(.+)", s, flags=re.I)
        if m:
            return m.group(1).strip()
    return ""


def _keyword_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "on",
        "in",
        "of",
        "to",
        "for",
        "with",
        "and",
        "or",
        "under",
        "left",
        "right",
        "top",
        "bottom",
        "middle",
        "center",
        "field",
        "button",
        "option",
        "section",
        "sidebar",
        "input",
        "dropdown",
    }
    return {
        t
        for t in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(t) >= 2 and t not in stop
    }


def _prefix_action_semantic_ok(
    action_str: str, observation_text: str, expected_hint: str = ""
) -> bool:
    element_id = _extract_action_element_id(action_str)
    if not element_id:
        return True
    current_sem = _normalize_semantic(
        _extract_element_semantic(observation_text, element_id)
    )
    if not current_sem:
        return False

    if not expected_hint:
        expected_hint = _extract_expected_element_hint(action_str)
    expected_hint = _normalize_semantic(expected_hint)
    if not expected_hint:
        return True
    if expected_hint in current_sem or current_sem in expected_hint:
        return True

    expected_tokens = _keyword_tokens(expected_hint)
    current_tokens = _keyword_tokens(current_sem)
    if not expected_tokens:
        return True
    overlap = len(expected_tokens & current_tokens)
    return overlap > 0


def _replace_action_element_id(action_str: str, new_element_id: str) -> str:
    if not new_element_id:
        return action_str
    return re.sub(
        r'(element\s*=\s*)(?:"[^"]*"|\'[^\']*\'|[0-9]+)',
        rf'\1"{new_element_id}"',
        action_str,
        count=1,
    )


def _collect_candidate_elements(observation_text: str, max_candidates: int = 300) -> list[dict]:
    ids: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'id="([^"]+)"', observation_text):
        eid = m.group(1).strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        ids.append(eid)
        if len(ids) >= max_candidates:
            break
    candidates: list[dict] = []
    for eid in ids:
        sem = _extract_element_semantic(observation_text, eid)
        sem = re.sub(r"\s+", " ", sem).strip()
        if not sem:
            continue
        candidates.append({"id": eid, "semantic": sem[:180]})
    return candidates


def _get_prefix_repair_client():
    global _PREFIX_REPAIR_CLIENT
    if _PREFIX_REPAIR_CLIENT is not None:
        return _PREFIX_REPAIR_CLIENT
    kwargs = {}
    api_key = (
        os.environ.get("OPENAI_EVAL_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    base_url = (
        os.environ.get("OPENAI_EVAL_API_URL", "").strip()
        or os.environ.get("OPENAI_API_URL", "").strip()
        or os.environ.get("OPENAI_API_URL_OVERRIDE", "").strip()
    )
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    _PREFIX_REPAIR_CLIENT = openai.OpenAI(**kwargs)
    return _PREFIX_REPAIR_CLIENT


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _repair_prefix_action_element(
    observation_text: str,
    action_str: str,
    model: str,
    expected_hint: str = "",
) -> tuple[str, dict]:
    trace: dict = {
        "repair_model": model,
        "status": "",
        "old_element_id": "",
        "new_element_id": "",
        "candidate_count": 0,
        "expected_hint": expected_hint,
        "raw_response": "",
        "parsed_json": None,
        "response_full": {},
        "choice0_full": {},
        "message_full": {},
        "finish_reason": "",
        "latency_ms": 0,
    }
    old_id = _extract_action_element_id(action_str)
    trace["old_element_id"] = old_id
    if not old_id:
        trace["status"] = "no_old_element_id"
        return action_str, trace
    if not expected_hint:
        expected_hint = _extract_expected_element_hint(action_str)
    trace["expected_hint"] = expected_hint
    old_sem = _extract_element_semantic(observation_text, old_id)
    candidates = _collect_candidate_elements(observation_text)
    trace["candidate_count"] = len(candidates)
    if not candidates:
        trace["status"] = "no_candidates"
        return action_str, trace

    client = _get_prefix_repair_client()
    prompt = {
        "task": "Choose the best element_id for this action from candidates. Change only element.",
        "raw_action": action_str,
        "old_element_id": old_id,
        "old_element_semantic": old_sem,
        "expected_element_hint": expected_hint,
        "candidates": candidates,
        "output_schema": {"element_id": "string from candidates.id"},
    }
    trace["prompt"] = prompt
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a web action repairer. "
                    "Only output JSON with key element_id. "
                    "Do not change action type or argument."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        temperature=0.0,
        max_tokens=4096,
        top_p=1.0,
    )
    trace["latency_ms"] = int((time.time() - t0) * 1000)
    try:
        if hasattr(response, "model_dump"):
            trace["response_full"] = response.model_dump()
        else:
            trace["response_full"] = dict(response)  # type: ignore[arg-type]
    except Exception:
        trace["response_full"] = {"_dump_error": "failed_to_dump_response"}
    try:
        choice0 = response.choices[0]
        if hasattr(choice0, "model_dump"):
            trace["choice0_full"] = choice0.model_dump()
        else:
            trace["choice0_full"] = dict(choice0)  # type: ignore[arg-type]
        trace["finish_reason"] = str(getattr(choice0, "finish_reason", "") or "")
    except Exception:
        trace["choice0_full"] = {"_dump_error": "failed_to_dump_choice0"}
    try:
        message = response.choices[0].message
        if hasattr(message, "model_dump"):
            trace["message_full"] = message.model_dump()
        else:
            trace["message_full"] = dict(message)  # type: ignore[arg-type]
    except Exception:
        trace["message_full"] = {"_dump_error": "failed_to_dump_message"}
    content = response.choices[0].message.content or ""
    trace["raw_response"] = content
    if not content.strip():
        trace["status"] = "empty_response"
        return action_str, trace
    obj = _extract_json_object(content)
    if not obj:
        trace["status"] = "json_not_found"
        return action_str, trace
    trace["parsed_json"] = obj
    new_id = str(obj.get("element_id", "")).strip()
    trace["new_element_id"] = new_id
    if not new_id or new_id == old_id:
        trace["status"] = "empty_or_same_new_id"
        return action_str, trace
    if _extract_element_semantic(observation_text, new_id).strip() == "":
        trace["status"] = "new_id_not_found_in_observation"
        return action_str, trace
    trace["status"] = "candidate_returned"
    return _replace_action_element_id(action_str, new_id), trace


def trim_terminal_prefix_action(
    prefix_actions: list[str], action_set_tag: str
) -> list[str]:
    if not prefix_actions:
        return prefix_actions
    try:
        last_action = parse_action_for_replay(prefix_actions[-1], action_set_tag)
    except Exception:
        return prefix_actions
    if last_action["action_type"] == ActionTypes.STOP:
        base_run.logger.info("[Prefix Replay] Dropping trailing STOP action from prefix")
        return prefix_actions[:-1]
    return prefix_actions


def replay_prefix_actions(
    env: ScriptBrowserEnv,
    trajectory: Trajectory,
    state_info: StateInfo,
    task_id: int,
    action_set_tag: str,
    prefix_actions: list[str],
    prefix_records: list[dict],
    task_snapshot_dir: str,
    prefix_id_repair_enable: bool = False,
    prefix_id_repair_model: str = "gpt-5-mini-ca",
    screenshot_dir: str | None = None,
    replay_debug_path: str | None = None,
    source_by_step: dict[int, dict] | None = None,
) -> StateInfo:
    if replay_debug_path:
        os.makedirs(os.path.dirname(replay_debug_path), exist_ok=True)
    for idx, action_str in enumerate(prefix_actions):
        observation_text = base_run._extract_observation_text(state_info["observation"])
        replay_action_str = action_str
        repaired = False
        repair_attempted = False
        repair_trace: dict = {}
        source_expected_hint = ""
        eid = _extract_action_element_id(action_str)
        if eid and source_by_step:
            pre_step = idx - 1
            src_pre = source_by_step.get(pre_step)
            if isinstance(src_pre, dict):
                source_expected_hint = _extract_element_semantic(
                    str(src_pre.get("observation_text", "")), eid
                )
        effective_hint = source_expected_hint or _extract_expected_element_hint(action_str)
        if prefix_id_repair_enable and not _prefix_action_semantic_ok(
            action_str, observation_text, expected_hint=effective_hint
        ):
            repair_attempted = True
            try:
                candidate, repair_trace = _repair_prefix_action_element(
                    observation_text=observation_text,
                    action_str=action_str,
                    model=prefix_id_repair_model,
                    expected_hint=effective_hint,
                )
                if candidate != action_str:
                    # strict: only allow changing element id
                    old_clean = _replace_action_element_id(action_str, "__ELEMENT__")
                    new_clean = _replace_action_element_id(candidate, "__ELEMENT__")
                    if old_clean == new_clean:
                        replay_action_str = candidate
                        repaired = True
                        repair_trace["status"] = "repaired"
                    else:
                        repair_trace["status"] = "candidate_rejected_non_element_change"
                else:
                    if "status" not in repair_trace:
                        repair_trace["status"] = "candidate_same_as_original"
            except Exception as e:
                repair_trace = {
                    "repair_model": prefix_id_repair_model,
                    "status": "exception",
                    "exception": repr(e),
                }
                base_run.logger.warning(
                    "[Prefix Repair] task=%s step=%s failed: %s",
                    task_id,
                    idx,
                    repr(e),
                )
        if repaired:
            base_run.logger.info(
                "[Prefix Repair] task=%s step=%s repaired action: %s -> %s",
                task_id,
                idx,
                action_str.replace("\n", " "),
                replay_action_str.replace("\n", " "),
            )

        action = parse_action_for_replay(replay_action_str, action_set_tag)
        trajectory.append(action)
        obs, _, _, _, info = env.step(action)
        state_info = {"observation": obs, "info": info}
        trajectory.append(state_info)
        screenshot_path = ""
        if screenshot_dir:
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_path = os.path.join(screenshot_dir, f"{idx}.png")
            try:
                env.page.screenshot(path=screenshot_path)
            except Exception:
                screenshot_path = ""
        if replay_debug_path:
            try:
                with open(replay_debug_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "step": idx,
                                "original_action": action_str,
                                "executed_action": replay_action_str,
                                "repaired": repaired,
                                "repair_attempted": repair_attempted,
                                "repair_trace": repair_trace if repair_attempted else {},
                                "expected_hint": effective_hint,
                                "url": env.page.url,
                                "screenshot": screenshot_path,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as e:
                base_run.logger.warning(
                    "[Prefix Replay] task=%s step=%s debug write failed: %s",
                    task_id,
                    idx,
                    repr(e),
                )
        base_run.append_prefix_state(
            prefix_records=prefix_records,
            task_id=task_id,
            step_idx=idx,
            url=env.page.url,
            actions=prefix_actions[: idx + 1],
            observation_text=base_run._extract_observation_text(
                state_info["observation"]
            ),
            llm_output=replay_action_str,
        )
        base_run.flush_prefix_states(task_snapshot_dir, prefix_records)
    return state_info


def _normalize_semantic(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _extract_element_semantic(observation_text: str, element_id: str) -> str:
    m = re.search(rf'id="{re.escape(element_id)}"[^>]*data-text="([^"]+)"', observation_text)
    if m:
        return m.group(1).strip()
    m = re.search(rf'id="{re.escape(element_id)}"[^>]*title="([^"]+)"', observation_text)
    if m:
        return m.group(1).strip()
    m = re.search(rf'<[^>]*id="{re.escape(element_id)}"[^>]*>(.*?)</[^>]+>', observation_text, re.S)
    if m:
        txt = re.sub(r"\s+", " ", m.group(1)).strip()
        if txt:
            return txt[:160]
    m = re.search(rf'.{{0,120}}id="{re.escape(element_id)}".{{0,180}}', observation_text, re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip()[:180]
    return ""


def _canonical_url_with_noise(url: str) -> str:
    if not url:
        return ""
    s = urlsplit(url)
    path = re.sub(r"/form_key/[^/]+/?", "/form_key/<MASK>/", s.path)
    volatile = {"form_key", "_", "token", "csrf", "nonce", "session", "sid", "key"}
    q = [(k, v) for k, v in parse_qsl(s.query, keep_blank_values=True) if k.lower() not in volatile]
    q.sort()
    if q:
        return f"{s.scheme}://{s.netloc}{path}?{urlencode(q)}"
    return f"{s.scheme}://{s.netloc}{path}"


def _site_from_url(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    if ":8080" in host:
        return "reddit"
    if ":8083" in host:
        return "shopping_admin"
    if ":9001" in host:
        return "gitlab"
    return "other"


def _is_reddit_slug_match(target_url: str, source_url: str) -> bool:
    t = urlsplit(target_url)
    s = urlsplit(source_url)
    if t.netloc != s.netloc:
        return False
    pt = [x for x in t.path.split("/") if x]
    ps = [x for x in s.path.split("/") if x]
    if len(pt) < 4 or len(ps) < 4:
        return False
    if pt[0] != "f" or ps[0] != "f":
        return False
    return pt[1] == ps[1] and pt[3] == ps[3]


def _is_gitlab_repairable(target_url: str, source_url: str) -> tuple[bool, str]:
    t = urlsplit(target_url)
    s = urlsplit(source_url)
    if t.netloc != s.netloc or t.path != s.path:
        return False, ""
    tq = parse_qs(t.query, keep_blank_values=True)
    sq = parse_qs(s.query, keep_blank_values=True)

    sort_t = (tq.get("sort") or [""])[0]
    sort_s = (sq.get("sort") or [""])[0]
    if {sort_t, sort_s} == {"created_date", "created_asc"}:
        tq2 = dict(tq)
        sq2 = dict(sq)
        tq2["sort"] = ["__SORT_ALIAS__"]
        sq2["sort"] = ["__SORT_ALIAS__"]
        if tq2 == sq2:
            return True, "gitlab_sort_alias"

    labels_t = tq.get("label_name[]", [])
    labels_s = sq.get("label_name[]", [])
    if len(labels_t) == 1 and len(labels_s) == 1 and {labels_t[0], labels_s[0]} == {"e", "one"}:
        tq2 = dict(tq)
        sq2 = dict(sq)
        tq2["label_name[]"] = ["__LABEL_ALIAS__"]
        sq2["label_name[]"] = ["__LABEL_ALIAS__"]
        if tq2 == sq2:
            return True, "gitlab_label_alias_e_one"
    return False, ""


def _load_reference_prefix_record(
    reference_dir: str, task_id: int, replayed_actions: list[str]
) -> dict | None:
    ref_path = Path(reference_dir) / "state_snapshots" / str(task_id) / "prefix.json"
    if not ref_path.exists():
        return None
    with ref_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        return None
    for rec in records:
        if isinstance(rec, dict) and rec.get("prefix_actions", []) == replayed_actions:
            return rec
    return None


def _load_reference_source_by_step(
    reference_dir: str, task_id: int, replayed_actions: list[str]
) -> dict[int, dict]:
    if not reference_dir:
        return {}
    ref_path = Path(reference_dir) / "state_snapshots" / str(task_id) / "prefix.json"
    if not ref_path.exists():
        return {}
    try:
        with ref_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return {}
    if not isinstance(records, list):
        return {}
    matched = any(
        isinstance(rec, dict) and rec.get("prefix_actions", []) == replayed_actions
        for rec in records
    )
    if not matched:
        return {}
    return {
        int(r.get("step")): r
        for r in records
        if isinstance(r, dict) and isinstance(r.get("step"), int)
    }


def _validate_prefix_replay(
    task_id: int,
    replayed_actions: list[str],
    target_prefix_records: list[dict],
    reference_dir: str,
    allow_repairable: bool,
) -> dict:
    result = {
        "task_id": task_id,
        "ok": False,
        "decision": "fail",
        "rule": "unknown",
        "reason": "",
        "target_url": "",
        "source_url": "",
        "object_checked": 0,
        "object_drift": 0,
    }
    if not replayed_actions:
        result.update({"ok": True, "decision": "pass", "rule": "empty_prefix", "reason": "No replay actions"})
        return result
    if not reference_dir:
        result.update({"rule": "missing_reference_dir", "reason": "--prefix_reference_dir is required for validation"})
        return result

    target_end = None
    for rec in target_prefix_records:
        if isinstance(rec, dict) and rec.get("prefix_actions", []) == replayed_actions:
            target_end = rec
            break
    source_end = _load_reference_prefix_record(reference_dir, task_id, replayed_actions)
    if target_end is None or source_end is None:
        result.update({"rule": "missing_prefix_match", "reason": "Could not find matched prefix record in target/source"})
        return result

    target_url = str(target_end.get("url", ""))
    source_url = str(source_end.get("url", ""))
    result["target_url"] = target_url
    result["source_url"] = source_url

    target_by_step = {int(r.get("step")): r for r in target_prefix_records if isinstance(r, dict) and isinstance(r.get("step"), int)}
    ref_path = Path(reference_dir) / "state_snapshots" / str(task_id) / "prefix.json"
    with ref_path.open("r", encoding="utf-8") as f:
        source_records = json.load(f)
    source_by_step = {int(r.get("step")): r for r in source_records if isinstance(r, dict) and isinstance(r.get("step"), int)}

    def _is_gitlab_sidebar_counter_noise(action: str, src_sem: str, tgt_sem: str) -> bool:
        """Whitelist GitLab sidebar counter jitter on same navigation icon/entry."""
        a = action.lower()
        s = src_sem.lower()
        t = tgt_sem.lower()
        if 'do(action="click"' not in a:
            return False
        # Current observed stable pattern: same merge icon entry, only count badge changes.
        if "git-merge-icon" in s and "git-merge-icon" in t:
            s_norm = re.sub(r"\b\d+\b", "<NUM>", s)
            t_norm = re.sub(r"\b\d+\b", "<NUM>", t)
            return s_norm == t_norm
        return False

    checked = 0
    drift = 0
    whitelisted_drift = 0
    elem_re = re.compile(r'element\s*=\s*"(\d+)"')
    gitlab_sort_values = {"name", "most stars", "updated date"}
    for i, action_str in enumerate(replayed_actions):
        m = elem_re.search(action_str)
        if not m:
            continue
        eid = m.group(1)
        pre_step = i - 1
        src_pre = source_by_step.get(pre_step)
        tgt_pre = target_by_step.get(pre_step)
        if src_pre is None or tgt_pre is None:
            continue
        src_sem = _extract_element_semantic(str(src_pre.get("observation_text", "")), eid)
        tgt_sem = _extract_element_semantic(str(tgt_pre.get("observation_text", "")), eid)
        checked += 1
        src_norm = _normalize_semantic(src_sem)
        tgt_norm = _normalize_semantic(tgt_sem)
        if src_norm != tgt_norm:
            site = _site_from_url(target_url or source_url)
            is_gitlab_sort_default_drift = (
                site == "gitlab"
                and 'do(action="click"' in action_str.lower()
                and src_norm in gitlab_sort_values
                and tgt_norm in gitlab_sort_values
            )
            is_gitlab_counter_noise = (
                site == "gitlab"
                and _is_gitlab_sidebar_counter_noise(action_str, src_sem, tgt_sem)
            )
            if is_gitlab_sort_default_drift or is_gitlab_counter_noise:
                whitelisted_drift += 1
            else:
                drift += 1
    result["object_checked"] = checked
    result["object_drift"] = drift
    result["object_whitelisted_drift"] = whitelisted_drift

    if drift > 0:
        result.update(
            {
                "decision": "fail",
                "rule": "prefix_object_mismatch",
                "reason": f"Object semantic drift detected: {drift}/{checked}",
            }
        )
        return result

    if target_url == source_url:
        result.update({"ok": True, "decision": "pass", "rule": "url_exact", "reason": "URL exact match"})
        return result

    site = _site_from_url(target_url or source_url)
    if site == "shopping_admin" and _canonical_url_with_noise(target_url) == _canonical_url_with_noise(source_url):
        result.update(
            {
                "ok": True,
                "decision": "pass",
                "rule": "shopping_admin_form_key_noise",
                "reason": "Only shopping_admin form/session noise",
            }
        )
        return result
    if site == "reddit" and _is_reddit_slug_match(target_url, source_url):
        result.update(
            {
                "ok": True,
                "decision": "pass",
                "rule": "reddit_same_slug_different_post_id",
                "reason": "Reddit same subreddit+slug",
            }
        )
        return result
    if site == "gitlab":
        rep_ok, rep_rule = _is_gitlab_repairable(target_url, source_url)
        if rep_ok:
            if allow_repairable:
                result.update(
                    {
                        "ok": True,
                        "decision": "repairable",
                        "rule": rep_rule,
                        "reason": "GitLab repairable URL drift (allowed)",
                    }
                )
            else:
                result.update(
                    {
                        "decision": "fail",
                        "rule": rep_rule,
                        "reason": "GitLab repairable URL drift (auto-fix not enabled)",
                    }
                )
            return result

    result.update({"decision": "fail", "rule": "url_semantic_mismatch", "reason": "URL mismatch outside exemption/repair rules"})
    return result


def test_with_prefix(
    args: argparse.Namespace, config_file_list: list[str]
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in ["accessibility_tree_with_captioner"]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    if base_run.DATASET == "visualwebarena":
        if caption_image_fn and args.eval_captioning_model == args.captioning_model:
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        captioning_fn=caption_image_fn,
    )

    for config_file in config_file_list:
        render_helper = None
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                images = []

                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as fw:
                        json.dump(_c, fw)

                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        if image_path.startswith("http"):
                            input_image = Image.open(
                                requests.get(image_path, stream=True).raw
                            )
                        else:
                            input_image = Image.open(image_path)
                        images.append(input_image)

            base_run.logger.info(f"[Config file]: {config_file}")
            base_run.logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)
            actions = trim_terminal_prefix_action(
                resolve_prefix_actions(args, task_id), args.action_set_tag
            )
            prefix_runtime = load_prefix_runtime_context(task_id)
            meta_data = {
                "action_history": ["None"],
                "task_id": task_id,
                "step": 0,
                # Prefix integration: guided-think is only consumed once after
                # successful prefix replay/validation.
                "guided_think_budget": 0,
            }
            if prefix_runtime:
                meta_data["prefix_runtime"] = prefix_runtime
            out_path = os.path.join(args.result_dir, "actions", f"{task_id}.json")
            replay_effect: dict = {
                "guided_records": [],
            }
            guided_record_steps: set[int] = set()
            snapshot_root = os.path.join(args.result_dir, "state_snapshots")
            task_snapshot_dir = os.path.join(snapshot_root, str(task_id))
            prefix_records: list[dict] = []

            os.makedirs(os.path.join(args.result_dir, "screehshots"), exist_ok=True)
            if os.path.exists(os.path.join(args.result_dir, "screehshots", f"{task_id}")):
                shutil.rmtree(os.path.join(args.result_dir, "screehshots", f"{task_id}"))
            os.makedirs(os.path.join(args.result_dir, "screehshots", f"{task_id}"))
            replay_debug_dir = os.path.join(args.result_dir, "prefix_replay")
            replay_debug_path = os.path.join(replay_debug_dir, f"{task_id}.jsonl")
            if os.path.exists(replay_debug_path):
                os.remove(replay_debug_path)
            if os.path.exists(task_snapshot_dir):
                shutil.rmtree(task_snapshot_dir)
            os.makedirs(task_snapshot_dir, exist_ok=True)
            base_run.append_prefix_state(
                prefix_records=prefix_records,
                task_id=task_id,
                step_idx=-1,
                url=env.page.url,
                actions=[],
                observation_text=base_run._extract_observation_text(
                    state_info["observation"]
                ),
                llm_output="",
            )
            base_run.flush_prefix_states(task_snapshot_dir, prefix_records)
            skip_task_due_prefix_validation = False
            restart_task_from_scratch = False
            allow_guided_after_prefix = False
            if actions:
                base_run.logger.info(
                    f"[Prefix Replay] Replaying {len(actions)} actions"
                )
                source_by_step = {}
                if args.prefix_id_repair_enable:
                    source_by_step = _load_reference_source_by_step(
                        reference_dir=args.prefix_reference_dir,
                        task_id=task_id,
                        replayed_actions=actions,
                    )
                state_info = replay_prefix_actions(
                    env=env,
                    trajectory=trajectory,
                    state_info=state_info,
                    task_id=task_id,
                    action_set_tag=args.action_set_tag,
                    prefix_actions=actions,
                    prefix_records=prefix_records,
                    task_snapshot_dir=task_snapshot_dir,
                    prefix_id_repair_enable=args.prefix_id_repair_enable,
                    prefix_id_repair_model=args.prefix_id_repair_model,
                    screenshot_dir=os.path.join(
                        args.result_dir, "screehshots", f"{task_id}"
                    ),
                    replay_debug_path=replay_debug_path,
                    source_by_step=source_by_step,
                )
                meta_data["action_history"].extend(actions)
                allow_guided_after_prefix = True
                if args.prefix_validate_replay:
                    check = _validate_prefix_replay(
                        task_id=task_id,
                        replayed_actions=actions,
                        target_prefix_records=prefix_records,
                        reference_dir=args.prefix_reference_dir,
                        allow_repairable=args.prefix_allow_repairable,
                    )
                    validate_dir = os.path.join(args.result_dir, "prefix_validation")
                    os.makedirs(validate_dir, exist_ok=True)
                    with open(os.path.join(validate_dir, f"{task_id}.json"), "w", encoding="utf-8") as vf:
                        json.dump(check, vf, ensure_ascii=False, indent=2)
                    base_run.logger.info(
                        "[Prefix Validation] task=%s decision=%s rule=%s reason=%s",
                        task_id,
                        check["decision"],
                        check["rule"],
                        check["reason"],
                    )
                    if not check.get("ok", False):
                        allow_guided_after_prefix = False
                        if args.prefix_validate_on_fail == "raise":
                            raise RuntimeError(
                                f"Prefix validation failed for task {task_id}: "
                                f"{check['rule']} | {check['reason']} | "
                                f"target={check['target_url']} source={check['source_url']}"
                            )
                        if args.prefix_validate_on_fail == "skip":
                            base_run.update_action_history(
                                out_path,
                                task_id,
                                actions=actions,
                                score=0,
                                extra_fields={"replay_effect": replay_effect},
                            )
                            base_run.logger.info(
                                "[Prefix Validation] Skip task %s due to failed replay validation",
                                task_id,
                            )
                            skip_task_due_prefix_validation = True
                        if args.prefix_validate_on_fail == "restart":
                            base_run.logger.info(
                                "[Prefix Validation] Restart task %s from scratch due to failed replay validation",
                                task_id,
                            )
                            restart_task_from_scratch = True
            if restart_task_from_scratch:
                trajectory = []
                obs, info = env.reset(options={"config_file": config_file})
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                meta_data = {
                    "action_history": ["None"],
                    "task_id": task_id,
                    "step": 0,
                    "guided_think_budget": 0,
                }
                if prefix_runtime:
                    meta_data["prefix_runtime"] = prefix_runtime
                actions = []
                prefix_records = []
                if os.path.exists(task_snapshot_dir):
                    shutil.rmtree(task_snapshot_dir)
                os.makedirs(task_snapshot_dir, exist_ok=True)
                base_run.append_prefix_state(
                    prefix_records=prefix_records,
                    task_id=task_id,
                    step_idx=-1,
                    url=env.page.url,
                    actions=[],
                    observation_text=base_run._extract_observation_text(
                        state_info["observation"]
                    ),
                    llm_output="",
                )
                base_run.flush_prefix_states(task_snapshot_dir, prefix_records)
            elif allow_guided_after_prefix and not skip_task_due_prefix_validation:
                # Prefix recovery succeeded: allow exactly one guided-think call.
                meta_data["guided_think_budget"] = 1

            if not skip_task_due_prefix_validation:
                while True:
                    base_run.update_action_history(
                        out_path,
                        task_id,
                        actions=actions,
                        extra_fields={"replay_effect": replay_effect},
                    )
                    early_stop_flag, stop_info = base_run.early_stop(
                        trajectory, max_steps, early_stop_thresholds, actions
                    )

                    if early_stop_flag:
                        action = create_stop_action(f"Early stop: {stop_info}")
                    else:
                        try:
                            meta_data["step"] = len(actions)
                            action = agent.next_action(
                                trajectory,
                                intent,
                                images=images,
                                meta_data=meta_data,
                            )
                            guided_last = meta_data.get("guided_last", {})
                            if isinstance(guided_last, dict):
                                try:
                                    step_idx = int(meta_data.get("step", len(actions)))
                                except Exception:
                                    step_idx = len(actions)
                                guided_called = bool(guided_last.get("guided_called", False))
                                if guided_called and step_idx not in guided_record_steps:
                                    guided_record_steps.add(step_idx)
                                    replay_effect["guided_records"].append(
                                        {
                                            "step": step_idx,
                                            "guided_apply_mode": guided_last.get("guided_apply_mode", ""),
                                            "next_action_guideness": guided_last.get("guided_think", ""),
                                            "guided_next_action": guided_last.get("guided_next_action", ""),
                                            "guided_replay_applied": bool(
                                                guided_last.get("guided_replay_applied", False)
                                            ),
                                        }
                                    )
                        except ValueError as e:
                            action = create_stop_action(f"ERROR: {str(e)}")

                    trajectory.append(action)

                    action_str = get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None,
                    )
                    render_helper.render(
                        action, state_info, meta_data, args.render_screenshot
                    )

                    current_screenshot = os.path.join(
                        args.result_dir, "screehshots", f"{task_id}", f"{len(actions)}.png"
                    )
                    _ = env.page.viewport_size
                    env.page.screenshot(path="/dev/null")
                    env.page.screenshot(path=current_screenshot)
                    element_id = action["element_id"]
                    if element_id != "":
                        element = env.page.query_selector(
                            f"[data-label-id='{element_id}']"
                        )
                        if element:
                            bbox = element.bounding_box()
                            bbox = [
                                int(bbox["x"]),
                                int(bbox["y"]),
                                int(bbox["width"]),
                                int(bbox["height"]),
                            ]
                            image = cv2.imread(current_screenshot)
                            cv2.rectangle(
                                image,
                                (bbox[0], bbox[1]),
                                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                (0, 255, 0),
                                2,
                            )
                            cv2.circle(
                                image,
                                (
                                    int(bbox[0] + bbox[2] / 2),
                                    int(bbox[1] + bbox[3] / 2),
                                ),
                                radius=0,
                                color=(0, 255, 0),
                                thickness=2,
                            )
                            cv2.imwrite(current_screenshot, image)

                    meta_data["action_history"].append(action_str)
                    actions.append(action_str)
                    print("Action String: ", action_str)

                    if action["action_type"] == ActionTypes.STOP:
                        base_run.append_prefix_state(
                            prefix_records=prefix_records,
                            task_id=task_id,
                            step_idx=len(actions) - 1,
                            url=env.page.url,
                            actions=actions,
                            observation_text=base_run._extract_observation_text(
                                state_info["observation"]
                            ),
                            llm_output=str(action.get("raw_prediction", "")),
                        )
                        base_run.flush_prefix_states(task_snapshot_dir, prefix_records)
                        break

                    obs, _, terminated, _, info = env.step(action)
                    state_info = {"observation": obs, "info": info}
                    trajectory.append(state_info)
                    base_run.append_prefix_state(
                        prefix_records=prefix_records,
                        task_id=task_id,
                        step_idx=len(actions) - 1,
                        url=env.page.url,
                        actions=actions,
                        observation_text=base_run._extract_observation_text(
                            state_info["observation"]
                        ),
                        llm_output=str(action.get("raw_prediction", "")),
                    )
                    base_run.flush_prefix_states(task_snapshot_dir, prefix_records)

                    if terminated:
                        trajectory.append(create_stop_action(""))
                        break

            if args.observation_type == "webrl" and not skip_task_due_prefix_validation:
                current_path = os.path.join(args.result_dir, "traces", f"{task_id}.jsonl")
                traces = []
                for i in range(1, len(trajectory), 2):
                    action = trajectory[i]
                    prev_state = trajectory[i - 1]
                    obs = prev_state["observation"]["text"]
                    action_str = action["raw_prediction"]
                    item = {
                        "trace_id": task_id,
                        "index": i // 2,
                        "prompt": intent if i == 1 else "** Simplified html **",
                        "html": obs,
                        "response": action_str,
                        "target": intent,
                    }
                    traces.append(item)
                with open(current_path, "w") as f:
                    for item in traces:
                        f.write(json.dumps(item) + "\n")

            if not skip_task_due_prefix_validation:
                evaluator = evaluator_router(
                    config_file, captioning_fn=eval_caption_image_fn
                )
                score = evaluator(
                    trajectory=trajectory,
                    config_file=config_file,
                    page=env.page,
                )

                base_run.update_action_history(
                    out_path,
                    task_id,
                    actions=actions,
                    score=score,
                    extra_fields={"replay_effect": replay_effect},
                )
                scores.append(score)

                if score == 1:
                    base_run.logger.info(f"[Result] (PASS) {config_file}")
                else:
                    base_run.logger.info(f"[Result] (FAIL) {config_file}")

                if args.save_trace_enabled:
                    env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")
        except openai.OpenAIError as e:
            base_run.logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            base_run.logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())

        finally:
            if render_helper is not None:
                render_helper.close()

    env.close()
    if len(scores):
        base_run.logger.info(f"Average score: {sum(scores) / len(scores)}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config_with_prefix()
    args.sleep_after_execution = 3.0
    base_run.prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = base_run.get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True
    args.current_viewport_only = True
    base_run.dump_config(args)

    test_with_prefix(args, test_file_list)
