import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import pathlib
import queue
import time
from typing import Any, Dict, Iterable, List

from colorama import Fore
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from webshop.web_agent_site.envs import WebAgentTextEnv

import agents as agents
import envs as envs
import tasks as tasks
from utils.datatypes import State
from utils.dcrl_runtime import (
    call_guided_think,
    collect_failure_candidates,
    format_guided_replay_output,
    get_last_user_message_content,
    get_original_failed_behavior_continuation,
    get_selected_prefix,
    is_action_executable_for_current_page,
    load_dcrl_task_artifacts,
    replay_prefix_actions,
)

logger = logging.getLogger("evaluation")
args = None
WEBSHOP_ROOT = pathlib.Path(os.environ.get("WEBSHOP_ROOT", "webshop"))
WEBSHOP_DATA_ROOT = WEBSHOP_ROOT / "data"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_webshop_env(exp_config_name: str) -> WebAgentTextEnv:
    kwargs: dict[str, Any] = {
        "observation_mode": "text",
        "human_goals": True,
    }
    if exp_config_name == "webshop_1000":
        kwargs.update(
            {
                "file_path": str(WEBSHOP_DATA_ROOT / "items_shuffle_1000.json"),
                "attr_path": str(WEBSHOP_DATA_ROOT / "items_ins_v2_1000.json"),
                "num_products": 1000,
                "human_goals": False,
            }
        )
    return WebAgentTextEnv(**kwargs)


def _task_ids_filter(task_ids_arg: str) -> set[int]:
    raw = str(task_ids_arg or "").strip()
    if not raw:
        return set()
    out = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.add(int(piece))
    return out


def _snapshot_state(state: State) -> dict[str, Any]:
    return {
        "history": copy.deepcopy(state.history),
        "reward": state.reward,
        "finished": state.finished,
        "success": state.success,
        "terminate_reason": state.terminate_reason,
        "error": state.error,
        "steps": state.steps,
        "info": copy.deepcopy(state.info),
    }


def _restore_state(state: State, snap: dict[str, Any]) -> None:
    state.history = copy.deepcopy(snap["history"])
    state.reward = snap["reward"]
    state.finished = snap["finished"]
    state.success = snap["success"]
    state.terminate_reason = snap["terminate_reason"]
    state.error = snap["error"]
    state.steps = snap["steps"]
    state.info = copy.deepcopy(snap["info"])


def _looks_invalid_observation(observation: str) -> bool:
    text = str(observation or "")
    return (
        text.startswith("Observation: Invalid action!")
        or text.startswith("Observation: Invalid format. The input must contains 'Action: '")
    )


def _to_json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item, depth + 1) for item in value]
    return str(value)


def _normalize_instruction_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("instruction:"):
        return text.split(":", 1)[1].strip()
    return text


def _normalize_goal_payload(obj: dict[str, Any]) -> dict[str, Any]:
    goal_keys = (
        "asin",
        "instruction_text",
        "attributes",
        "goal_options",
        "price_upper",
        "query",
        "category",
        "product_category",
        "name",
    )
    normalized: dict[str, Any] = {}
    for key in goal_keys:
        if key not in obj:
            continue
        value = _to_json_safe(obj.get(key))
        if key == "instruction_text":
            value = _normalize_instruction_text(value)
        normalized[key] = value
    return normalized


def _extract_webshop_goal_snapshot(env: envs.BaseEnv, task: tasks.Task) -> dict[str, Any]:
    wrapped_env = getattr(env, "env", None)
    server = getattr(wrapped_env, "server", None)
    session_id = getattr(env, "session_id", None) or getattr(wrapped_env, "session", None)
    normalized_session_id = str(session_id).strip() if session_id is not None else ""
    user_sessions = getattr(server, "user_sessions", None)

    if isinstance(user_sessions, dict) and normalized_session_id in user_sessions:
        session = user_sessions.get(normalized_session_id) or {}
        goal = session.get("goal")
        if isinstance(goal, dict):
            normalized = _normalize_goal_payload(goal)
            if normalized:
                return normalized

    instruction = _normalize_instruction_text(getattr(task, "instruction", None))
    if instruction:
        return {"instruction_text": instruction}
    return {}


def _find_goal_payload(obj: Any, depth: int = 0, seen: set[int] | None = None) -> dict[str, Any]:
    if seen is None:
        seen = set()
    if obj is None or depth > 4:
        return {}
    oid = id(obj)
    if oid in seen:
        return {}
    seen.add(oid)

    if isinstance(obj, dict):
        keys = set(obj.keys())
        if {"instruction_text", "price_upper"} & keys or {"asin", "goal_options", "attributes"} & keys:
            return _normalize_goal_payload(obj)
        for value in obj.values():
            found = _find_goal_payload(value, depth + 1, seen)
            if found:
                return found
        return {}

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_goal_payload(value, depth + 1, seen)
            if found:
                return found
        return {}

    if hasattr(obj, "__dict__"):
        return _find_goal_payload(vars(obj), depth + 1, seen)

    return {}


def _extract_goal_snapshot(env: envs.BaseEnv, task: tasks.Task, dcrl_artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(task, tasks.WebShopTask):
        goal = _extract_webshop_goal_snapshot(env, task)
        if goal:
            return goal
    if dcrl_artifacts:
        goal = _normalize_goal_payload(dict(dcrl_artifacts.get("parsed_trace", {}).get("goal") or {}))
        if goal:
            return goal
    goal = _find_goal_payload(getattr(env, "env", None))
    if goal:
        return goal
    goal = _find_goal_payload(task)
    if goal:
        return goal
    if isinstance(task, tasks.WebShopTask):
        return _extract_webshop_goal_snapshot(env, task)
    return {}


def _prepare_dcrl_context(task: tasks.Task, env: envs.BaseEnv, state: State) -> dict[str, Any]:
    if str(args.exp_config or "").strip().lower() not in {"webshop", "webshop_1000", "webshop_full"}:
        return {}
    if not str(args.prefix_root or "").strip():
        return {}
    if int(args.run or 0) <= 0:
        return {}
    guided_mode = str(args.guided_think_mode or "none").strip().lower()
    if guided_mode == "none":
        return {}

    artifacts = load_dcrl_task_artifacts(args.prefix_root, task.task_id)
    selected_anchor = dict(artifacts.get("selected_anchor") or {})
    selected_prefix_id = str(selected_anchor.get("selected_prefix_id") or "").strip()
    if not selected_prefix_id:
        return {}
    selected_prefix = get_selected_prefix(dict(artifacts.get("prefix_memory") or {}), selected_prefix_id)
    if not selected_prefix:
        return {}

    prefix_action_count = replay_prefix_actions(env, selected_prefix)
    state = env.state
    failure_candidates = collect_failure_candidates(dict(artifacts.get("failure_memory") or {}), selected_prefix)
    return {
        "artifacts": artifacts,
        "selected_anchor": selected_anchor,
        "selected_prefix": selected_prefix,
        "selected_prefix_id": selected_prefix_id,
        "selected_prefix_rationale": str(selected_anchor.get("rationale") or "").strip(),
        "failure_candidates": failure_candidates,
        "prefix_action_count": prefix_action_count,
        "guided_pending": guided_mode not in {"replay_only", "prefix_only", "no_guidance"},
    }


def _merge_info(state: State, payload: dict[str, Any]) -> None:
    if state.info is None:
        state.info = {}
    state.info.update(payload)


def _append_management_step(task_id: Any, step: str, input_text: str, output_text: str) -> None:
    prefix_root = str(args.prefix_root or "").strip()
    if not prefix_root:
        return
    try:
        path = pathlib.Path(prefix_root) / str(task_id) / "_artifacts" / "management_steps.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "step": str(step),
                        "iter": int(args.iter or 0),
                        "run": int(args.run or 0),
                        "input": str(input_text or ""),
                        "output": str(output_text or ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception as exc:
        logger.warning(f"failed to append management step for task={task_id}: {exc}")


def _run_guided_step(
    task: tasks.Task,
    agent: agents.BaseAgent,
    env: envs.BaseEnv,
    state: State,
    dcrl_context: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    task_goal = str((state.goal or {}).get("instruction_text") or getattr(task, "instruction", "")).strip()
    failure_candidates = list(dcrl_context.get("failure_candidates") or [])
    original_failed_continuation = get_original_failed_behavior_continuation(failure_candidates)
    guided_result, trace = call_guided_think(
        model_name=args.guided_think_model or args.model_name,
        api_base=args.guided_api_base or args.api_base,
        api_key=args.guided_api_key or args.api_key,
        task_goal=task_goal,
        history_messages=state.history,
        failure_candidates=failure_candidates,
        original_failed_continuation=original_failed_continuation,
        timeout=_safe_float(args.guided_think_timeout, 60.0),
    )
    _append_management_step(
        task.task_id,
        "guided_think",
        trace.get("prompt", ""),
        trace.get("raw_response", ""),
    )

    info_payload = {
        "dcrl_selected_prefix_id": dcrl_context.get("selected_prefix_id"),
        "dcrl_prefix_action_count": dcrl_context.get("prefix_action_count", 0),
        "dcrl_failure_candidate_count": len(failure_candidates),
        "dcrl_selected_prefix_rationale": dcrl_context.get("selected_prefix_rationale", ""),
        "dcrl_guided_called": True,
        "dcrl_guided_reason": "ok",
        "dcrl_guided_prompt": trace.get("prompt", ""),
        "dcrl_guided_raw_response": trace.get("raw_response", ""),
        "dcrl_guided_next_action": guided_result.get("next_action", ""),
        "dcrl_guided_thought": guided_result.get("thought", ""),
    }

    apply_mode = str(args.guided_think_apply_mode or "guideness").strip().lower()
    if apply_mode == "replay":
        llm_output = format_guided_replay_output(guided_result)
        info_payload["dcrl_guided_policy_output"] = llm_output
        return llm_output, info_payload

    prefill = agent.build_prefilled_thought(guided_result.get("thought", ""), guided_result.get("next_action", ""))
    policy_messages = list(state.history) + [{"role": "assistant", "content": prefill}]
    llm_output = agent.call_with_prefilled_thought(
        state.history,
        guided_result.get("thought", ""),
        guided_result.get("next_action", ""),
    )

    info_payload.update(
        {
            "dcrl_guided_policy_input_last_user": get_last_user_message_content(state.history),
            "dcrl_guided_policy_prefill": prefill,
            "dcrl_guided_policy_messages": json.dumps(policy_messages, ensure_ascii=False),
            "dcrl_guided_fallback_triggered": False,
            "dcrl_guided_fallback_reason": "",
        }
    )

    if not is_action_executable_for_current_page(env, llm_output):
        info_payload["dcrl_guided_policy_output_before_fallback"] = llm_output
        info_payload["dcrl_guided_fallback_triggered"] = True
        info_payload["dcrl_guided_fallback_reason"] = "non_executable_for_current_page"
        llm_output = format_guided_replay_output(guided_result)

    info_payload["dcrl_guided_policy_output"] = llm_output
    return llm_output, info_payload


def interactive_loop(
    task: tasks.Task,
    agent: agents.BaseAgent,
    env_config: Dict[str, Any],
) -> State:
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, incorporation_type=args.incorporation_type, **env_config)
    observation, state = env.reset()
    state.started_at = time.time()

    init_msg = observation
    logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    dcrl_context = _prepare_dcrl_context(task, env, state)
    state = env.state
    state.goal = _extract_goal_snapshot(env, task, dcrl_context.get("artifacts"))
    if dcrl_context:
        _merge_info(
            state,
            {
                "dcrl_selected_prefix_id": dcrl_context.get("selected_prefix_id"),
                "dcrl_prefix_action_count": dcrl_context.get("prefix_action_count", 0),
                "dcrl_failure_candidate_count": len(dcrl_context.get("failure_candidates") or []),
                "dcrl_selected_prefix_rationale": dcrl_context.get("selected_prefix_rationale", ""),
            },
        )

    while not state.finished:
        guided_step_info: dict[str, Any] | None = None
        try:
            if dcrl_context.get("guided_pending"):
                llm_output, guided_step_info = _run_guided_step(task, agent, env, state, dcrl_context)
                dcrl_context["guided_pending"] = False
            elif args.incorporation_type == "thought" and getattr(task, "workflow", None):
                llm_output = agent.call_with_workflow(state.history)
            else:
                llm_output = agent(state.history)
            logger.info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n")
        except Exception as e:
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception value: {e}")
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            state.error = str(e)
            break

        snap = _snapshot_state(state)
        observation, state = env.step(llm_output)

        if guided_step_info is not None:
            if _looks_invalid_observation(observation) and guided_step_info.get("dcrl_guided_next_action"):
                _restore_state(state, snap)
                fallback_output = format_guided_replay_output(
                    {
                        "thought": guided_step_info.get("dcrl_guided_thought", ""),
                        "next_action": guided_step_info.get("dcrl_guided_next_action", ""),
                    }
                )
                guided_step_info["dcrl_guided_policy_output_before_fallback"] = llm_output
                guided_step_info["dcrl_guided_fallback_triggered"] = True
                guided_step_info["dcrl_guided_fallback_reason"] = "env_invalid_after_step"
                logger.info(f"\n{Fore.GREEN}{fallback_output}{Fore.RESET}\n")
                observation, state = env.step(fallback_output)
                llm_output = fallback_output
            guided_step_info["dcrl_guided_policy_output"] = llm_output
            _merge_info(state, guided_step_info)

        if not state.finished:
            logger.info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")

    state.finished_at = time.time()
    if state.started_at is not None and state.finished_at is not None:
        state.duration_seconds = state.finished_at - state.started_at

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
        if state.info is not None:
            logger.info(f"Info: {state.info}")
    else:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}")

    return state


def _build_worker_env_config(exp_config_name: str, env_config: Dict[str, Any]) -> Dict[str, Any]:
    worker_env_config = copy.deepcopy(env_config)
    if worker_env_config.get("env_class") == "WebShopEnv":
        worker_env_config["env"] = _build_webshop_env(exp_config_name)
    return worker_env_config


def _save_state_json(output_path: str, task_id: Any, state: State) -> None:
    with open(os.path.join(output_path, f"{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=4, ensure_ascii=False)


def _run_task_worker(
    *,
    worker_id: int,
    task_queue: "queue.Queue[tasks.Task]",
    exp_config_name: str,
    env_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    output_path: str,
    progress_callback: Any = None,
) -> List[State]:
    worker_env_config = _build_worker_env_config(exp_config_name, env_config)
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(copy.deepcopy(agent_config["config"]))
    logger.info(f"worker={worker_id} ready")
    worker_states: List[State] = []

    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        try:
            state = interactive_loop(task, agent, worker_env_config)
            worker_states.append(state)
            _save_state_json(output_path, task.task_id, state)
            if progress_callback is not None:
                progress_callback()
        finally:
            task_queue.task_done()

    logger.info(f"worker={worker_id} done")
    return worker_states


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name
    if args.api_base is not None:
        agent_config['config']['api_base'] = args.api_base
    if args.api_key is not None:
        agent_config['config']['api_key'] = args.api_key
    if args.temperature is not None:
        agent_config['config']['temperature'] = args.temperature

    exp_name = args.exp_name or args.exp_config

    if args.output_dir is not None:
        output_path = args.output_dir
    else:
        model_leaf = agent_config['config']['model_name'].split('/')[-1]
        if args.split == "test":
            output_path = os.path.join(exp_config["output_dir"], "unseen", model_leaf, exp_name, args.metaplan_type)
        elif args.split == "dev":
            output_path = os.path.join(exp_config["output_dir"], "seen", model_leaf, exp_name, args.metaplan_type)
        else:
            output_path = os.path.join(exp_config["output_dir"], "train", model_leaf, exp_name, args.metaplan_type)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler(), file_handler])

    env_config = exp_config["env_config"]
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    task_config: Dict[str, Any] = exp_config["task"]
    if args.metaplan_path is not None:
        task_config['workflow_path'] = args.metaplan_path

    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks_iter, n_tasks = task_class.load_tasks(
        path=task_config.get("filepath", ""),
        split=args.split,
        part_num=args.part_num,
        part_idx=args.part_idx,
    )
    all_tasks = list(all_tasks_iter)
    selected_task_ids = _task_ids_filter(args.task_ids)
    if selected_task_ids:
        all_tasks = [task for task in all_tasks if int(task.task_id) in selected_task_ids]
        n_tasks = len(all_tasks)

    state_list = []
    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(int(file.split('.')[0]))
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks and n_tasks > 0:
        reward_list = [state.reward for state in state_list if state.reward is not None]
        success_list = [state.success for state in state_list]
        if reward_list:
            logger.info(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.info(f"Success rate: {sum(success_list)/len(success_list):.4f}")
        logger.info("All tasks done. Exiting.")
        return

    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    pending_tasks = [task for task in all_tasks if task.task_id not in done_task_id]
    n_todo_tasks = len(pending_tasks)

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        worker_count = max(1, int(args.workers))
        if worker_count == 1:
            single_env_config = _build_worker_env_config(args.exp_config, env_config)
            agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(copy.deepcopy(agent_config["config"]))
            for i, task in enumerate(pending_tasks):
                if args.debug and i == 5:
                    break

                state = interactive_loop(task, agent, single_env_config)
                state_list.append(state)
                _save_state_json(output_path, task.task_id, state)

                pbar.update(1)
        else:
            task_queue: "queue.Queue[tasks.Task]" = queue.Queue()
            queued_tasks = pending_tasks[:5] if args.debug else pending_tasks
            for task in queued_tasks:
                task_queue.put(task)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _run_task_worker,
                        worker_id=worker_id,
                        task_queue=task_queue,
                            exp_config_name=args.exp_config,
                            env_config=env_config,
                            agent_config=agent_config,
                            output_path=output_path,
                            progress_callback=lambda: pbar.update(1),
                        )
                    for worker_id in range(worker_count)
                ]
                for future in as_completed(futures):
                    worker_states = future.result()
                    state_list.extend(worker_states)
        pbar.close()

        logger.info("All tasks done.")
        logger.info(f"Output saved to {output_path}")

        reward_list = []
        success_list = []
        att_list, option_list, type_list, price_list = [], [], [], []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            if args.exp_config == 'webshop':
                success_list.append(state.reward == 1)
                if state.info is not None:
                    att_list.append(state.info.get('r_att', 0))
                    option_list.append(state.info.get('r_option', 0))
                    type_list.append(state.info.get('r_type', 0))
                    price_list.append(state.info.get('w_price', 0))
                else:
                    att_list.extend([0])
                    option_list.extend([0])
                    type_list.extend([0])
                    price_list.extend([0])
            else:
                success_list.append(state.success)

        if reward_list:
            logger.info(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        if att_list:
            logger.info(f"Average attribute score: {sum(att_list)/len(att_list):.4f}")
            logger.info(f"Average option score: {sum(option_list)/len(option_list):.4f}")
            logger.info(f"Average type score: {sum(type_list)/len(type_list):.4f}")
            logger.info(f"Average price score: {sum(price_list)/len(price_list):.4f}")
        if success_list:
            logger.info(f"Success rate: {sum(success_list)/len(success_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument("--exp_path", type=str, default="./configs/task", help="Config path of experiment.")
    parser.add_argument("--exp_config", type=str, default="math", help="Config of experiment.")
    parser.add_argument("--exp_name", type=str, help="Config of experiment.")
    parser.add_argument("--split", type=str, default="test", help="Evaluation split.")
    parser.add_argument("--part_num", type=int, default=1, help="Evaluation part.")
    parser.add_argument("--part_idx", type=int, default=-1, help="Evaluation part.")
    parser.add_argument("--agent_path", type=str, default="./configs/model", help="Config path of model.")
    parser.add_argument("--agent_config", type=str, default="openai", help="Config of model.")
    parser.add_argument("--model_name", type=str, required=False, help="Model name. It will override the 'model_name' in agent_config")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode (10 ex per task).")
    parser.add_argument("--override", action="store_true", help="Whether to ignore done tasks.")
    parser.add_argument("--incorporation_type", type=str, choices=["query", "observation", "thought", "no_icl"], default="no_icl")
    parser.add_argument("--api_base", type=str, required=False, help="Agent base url. It will override the 'api_base' in agent_config")
    parser.add_argument("--api_key", type=str, required=False, help="Agent api key. It will override the 'api_key' in agent_config")
    parser.add_argument("--temperature", type=float, default=None, help="Agent sampling temperature. It will override the 'temperature' in agent_config")
    parser.add_argument("--metaplan_path", type=str, required=False, help="Metaplan path.")
    parser.add_argument("--metaplan_type", type=str, choices=["none", "sft", "rft", "mpo"], default="none", help="Metaplan type.")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to save the output.")
    parser.add_argument("--prefix_root", type=str, default="", help="DCRL memory root.")
    parser.add_argument("--iter", type=int, default=0, help="DCRL iter index.")
    parser.add_argument("--run", type=int, default=0, help="DCRL run index.")
    parser.add_argument("--task_ids", type=str, default="", help="Comma-separated task ids.")
    parser.add_argument("--guided_think_mode", type=str, default="none", help="Guided-think mode.")
    parser.add_argument("--guided_think_apply_mode", type=str, default="guideness", help="Guided-think application mode.")
    parser.add_argument("--guided_think_model", type=str, default="", help="Guide model name.")
    parser.add_argument("--guided_api_base", type=str, default="", help="Guide api base.")
    parser.add_argument("--guided_api_key", type=str, default="", help="Guide api key.")
    parser.add_argument("--guided_think_timeout", type=float, default=60.0, help="Guide request timeout seconds.")
    parser.add_argument("--workers", type=int, default=1, help="Number of in-process rollout workers.")

    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    main(args)
