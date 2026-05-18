import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import tiktoken
from beartype import beartype
from openai import OpenAI
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_webrl_id_based_action
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
        planner_ip = None
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.planner_ip = planner_ip
        self.dcrl_episode_memory = {"goal": "", "milestones_done": [], "failed_attempts": []}
        self.dcrl_semantic_memory = ""
        self.dcrl_think_history: list[str] = []
        meta_cfg = {}
        try:
            meta_cfg = dict(prompt_constructor.instruction.get("meta_data", {}))
        except Exception:
            meta_cfg = {}
        self.guided_think_mode = (
            os.environ.get("GUIDED_THINK_MODE", "")
            or str(meta_cfg.get("guided_think_mode", "")).strip()
            or "off"
        ).strip().lower()
        self.guided_think_model = (
            os.environ.get("GUIDED_THINK_MODEL", "")
            or os.environ.get("DCRL_MODEL", "")
            or str(meta_cfg.get("guided_think_model", "deepseek-v3.2")).strip()
        ).strip()
        self.guided_think_apply_mode = (
            os.environ.get("GUIDED_THINK_APPLY_MODE", "")
            or str(meta_cfg.get("guided_think_apply_mode", "replay")).strip()
            or "replay"
        ).strip().lower()
        if self.guided_think_apply_mode not in {"guideness", "replay"}:
            self.guided_think_apply_mode = "replay"
        timeout_str = (
            os.environ.get("GUIDED_THINK_TIMEOUT", "")
            or str(meta_cfg.get("guided_think_timeout", "8")).strip()
        )
        try:
            self.guided_think_timeout = max(0.1, float(timeout_str))
        except Exception:
            self.guided_think_timeout = 8.0
        self._guided_openai_client: Any | None = None
        self.guideness_dir = os.environ.get("GUIDENESS_DIR", "").strip()
        
        # Check if the model is multimodal.
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model or lm_config.provider in ["api", "finetune"]) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

    def _dcrl_debug_enabled(self) -> bool:
        return bool(os.environ.get("DCRL_DEBUG_DUMP_DIR"))

    def _dcrl_debug_dump(self, payload: dict[str, Any]) -> None:
        dump_dir = os.environ.get("DCRL_DEBUG_DUMP_DIR", "").strip()
        if not dump_dir:
            return
        try:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            task_id = payload.get("task_id", "unknown")
            path = Path(dump_dir) / f"task_{task_id}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _guideness_dump(self, payload: dict[str, Any]) -> None:
        if not self.guideness_dir:
            return
        try:
            Path(self.guideness_dir).mkdir(parents=True, exist_ok=True)
            task_id = payload.get("task_id", "unknown")
            path = Path(self.guideness_dir) / f"task_{task_id}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def _build_guided_think(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        current_step_input_text: str = "",
    ) -> tuple[str, str, dict[str, Any]]:
        """Generate guided-think and optional guided next-action using internal auto-llm only."""
        if self.guided_think_mode != "auto":
            return "", "", {
                "guided_called": False,
                "guided_reason": f"mode={self.guided_think_mode}",
            }
        try:
            return self._build_guided_think_auto_llm(
                trajectory, intent, meta_data, current_step_input_text
            )
        except Exception as e:
            print(f"[Guided Think] auto mode exception: {repr(e)}", flush=True)
            return "", "", {
                "guided_called": True,
                "guided_exception": repr(e),
                "guided_traceback": traceback.format_exc(),
            }

    def _get_guided_openai_client(self) -> OpenAI:
        if self._guided_openai_client is not None:
            return self._guided_openai_client
        api_key = (
            os.environ.get("OPENAI_EVAL_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        base_url = (
            os.environ.get("OPENAI_EVAL_API_URL", "").strip()
            or os.environ.get("OPENAI_API_URL", "").strip()
            or os.environ.get("OPENAI_API_URL_OVERRIDE", "").strip()
        )
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._guided_openai_client = OpenAI(**kwargs)
        return self._guided_openai_client

    def _load_failure_context_from_prefix_runtime(
        self, meta_data: dict[str, Any]
    ) -> tuple[str, str]:
        prefix_runtime = meta_data.get("prefix_runtime")
        if not isinstance(prefix_runtime, dict):
            return "", ""

        failure_candidates = prefix_runtime.get("failure_candidates", [])
        if not isinstance(failure_candidates, list):
            return "", ""

        summaries: list[str] = []
        original_failed_behavior = ""
        for idx, candidate in enumerate(failure_candidates):
            if not isinstance(candidate, dict):
                continue
            failure_path_suffix = candidate.get("failure_path_suffix", [])
            if not isinstance(failure_path_suffix, list):
                failure_path_suffix = []
            failure_path_suffix = [
                str(action).strip()
                for action in failure_path_suffix
                if str(action).strip()
            ]
            failure_summary = str(candidate.get("failure_summary", "")).strip()
            first_failed_action = str(
                candidate.get("first_failed_action", "")
            ).strip()

            parts = [f"[Failure {idx + 1}]"]
            if failure_path_suffix:
                path_lines = "\n".join(
                    f"{step_idx + 1}. {action}"
                    for step_idx, action in enumerate(failure_path_suffix)
                )
                parts.append(
                    "Historical continuation after the current prefix:\n"
                    f"{path_lines}"
                )
                if first_failed_action:
                    parts.append(
                        "First failed action after that continuation:\n"
                        f"{first_failed_action}"
                    )
            elif first_failed_action:
                parts.append(
                    "First failed action from this prefix:\n"
                    f"{first_failed_action}"
                )
            if failure_summary:
                parts.append(f"Failure summary:\n{failure_summary}")
            if len(parts) > 1:
                summaries.append("\n".join(parts))

            if not original_failed_behavior:
                if failure_path_suffix:
                    original_failed_behavior = "\n".join(failure_path_suffix)
                    if first_failed_action:
                        original_failed_behavior = (
                            f"{original_failed_behavior}\n{first_failed_action}"
                        ).strip()
                elif first_failed_action:
                    original_failed_behavior = first_failed_action
                else:
                    failure_cmus = candidate.get("failure_cmus", [])
                    if isinstance(failure_cmus, list):
                        for failure_cmu in failure_cmus:
                            if not isinstance(failure_cmu, dict):
                                continue
                            actions = failure_cmu.get("actions", [])
                            if not isinstance(actions, list) or not actions:
                                continue
                            first_action = str(actions[0]).strip()
                            if first_action:
                                original_failed_behavior = first_action
                                break

        return "\n\n".join(summaries).strip(), original_failed_behavior

    def _load_guided_failure_context(
        self, meta_data: dict[str, Any], task_id: Any
    ) -> tuple[str, str, str]:
        failure_memories, original_failed_behavior = (
            self._load_failure_context_from_prefix_runtime(meta_data)
        )
        if failure_memories or original_failed_behavior:
            return failure_memories, original_failed_behavior, "prefix_runtime"
        return "", "", "prefix_runtime_missing"

    def _build_guided_think_auto_llm(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        current_step_input_text: str = "",
    ) -> tuple[str, str, dict[str, Any]]:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = str(state_info["observation"].get("text", ""))[:6000]
        task_id = meta_data.get("task_id")
        (
            failure_memories,
            original_failed_behavior,
            failure_context_source,
        ) = self._load_guided_failure_context(meta_data, task_id)
        current_step_input = (current_step_input_text or "").strip()[:12000]
        if not current_step_input:
            current_step_input = obs if obs else "(empty)"

        prompt = f"""You are a long-horizon environment interaction agent.
You are NOT starting from scratch. A reusable skill prefix has already been replayed, and you are now at that prefix frontier state.
Your task is to generate alternative action guidance and the next action based on prior failed trajectory reflections and the current step input.

Your output format must strictly follow:

next_action_guideness:

#Note: for key intermediate reasoning/results

#Element: for locating the target UI element to operate on (button, input field, link).

next_action must be selected from the predefined functions:

def do(action, argument, element):
    action must be one of ["Click", "Right Click", "Type", "Search", "Hover", "Scroll Up", "Scroll Down", "Press Enter", "Switch Tab", "Select Dropdown Option", "Wait"].
    The call must include keyword action="...".
    argument is required for Type, Search, Select Dropdown Option, and Switch Tab; otherwise omit argument.
    element is required for Click, Right Click, Hover, Type, Search, and Select Dropdown Option, and it must be a numeric element id string (e.g., "16"), not natural-language text.
    Omit element for Scroll Up, Scroll Down, Press Enter, Wait, and Switch Tab.

def exit(message):
    End the browsing process when the task is fulfilled.

def go_backward():
    Go back to the previous page.

def go_forward():
    Go forward to the next page.

Only output JSON in the following format:
{{
  "rational": "...",
  "next_action_guideness": "...",
  "next_action": "..."
}}

Constraints:
1. Output exactly one JSON object. Do not output markdown/code fences or any extra explanation text.
2. `next_action_guideness` may include `#Note`, `#Element`, either one, or both.
3. Output `#Element` only when the next action operates on an element.
4. `next_action` must be one valid call from the predefined functions above.
5. Adhering to the Minimal Edit Principle, the Rationale must first predict the intended effect of the "Original Failed Behavior." Identify $n$ alternative operations closest to the original intent, then evaluate their semantic distance. Prioritize configurations affecting the same element to ensure the final execution remains as logically aligned with the original attempt as possible.

Task:
{intent}

Current step input (merged):
{current_step_input}

Matched failure memories for this prefix:
{failure_memories if failure_memories else "(empty)"}

Original failed behavior continuation for this prefix:
{original_failed_behavior if original_failed_behavior else "(empty)"}
"""
        trace: dict[str, Any] = {
            "guided_called": True,
            "guided_model": self.guided_think_model,
            "guided_system_prompt": "You are a web action guide generator. Return JSON only.",
            "guided_user_prompt": prompt,
            "guided_response_full": {},
            "guided_choice0_full": {},
            "guided_message_full": {},
            "guided_finish_reason": "",
            "guided_raw_response": "",
            "guided_json_candidate": "",
            "guided_json_parsed": False,
            "guided_error": "",
            "guided_failure_context_source": failure_context_source,
        }
        client = self._get_guided_openai_client()
        t0 = time.time()
        resp = client.chat.completions.create(
            model=self.guided_think_model,
            messages=[
                {"role": "system", "content": "You are a web action guide generator. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=8096,
            top_p=1.0,
        )
        trace["guided_latency_ms"] = int((time.time() - t0) * 1000)
        try:
            if hasattr(resp, "model_dump"):
                trace["guided_response_full"] = resp.model_dump()
            else:
                trace["guided_response_full"] = dict(resp)  # type: ignore[arg-type]
        except Exception:
            trace["guided_response_full"] = {"_dump_error": "failed_to_dump_response"}
        try:
            choice0 = resp.choices[0]
            if hasattr(choice0, "model_dump"):
                trace["guided_choice0_full"] = choice0.model_dump()
            else:
                trace["guided_choice0_full"] = dict(choice0)  # type: ignore[arg-type]
            trace["guided_finish_reason"] = str(getattr(choice0, "finish_reason", "") or "")
        except Exception:
            trace["guided_choice0_full"] = {"_dump_error": "failed_to_dump_choice0"}
        try:
            message = resp.choices[0].message
            if hasattr(message, "model_dump"):
                trace["guided_message_full"] = message.model_dump()
            else:
                trace["guided_message_full"] = dict(message)  # type: ignore[arg-type]
        except Exception:
            trace["guided_message_full"] = {"_dump_error": "failed_to_dump_message"}
        content = (resp.choices[0].message.content or "").strip()
        trace["guided_raw_response"] = content
        if not content:
            trace["guided_error"] = "empty_response"
            return "", "", trace
        obj = None
        try:
            obj = json.loads(content)
            trace["guided_json_candidate"] = content
        except Exception:
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                try:
                    candidate = m.group(0)
                    trace["guided_json_candidate"] = candidate
                    obj = json.loads(candidate)
                except Exception:
                    obj = None
                    trace["guided_error"] = "json_extract_parse_failed"
            else:
                trace["guided_error"] = "json_not_found"
        if not isinstance(obj, dict):
            return "", "", trace
        trace["guided_json_parsed"] = True
        g = str(obj.get("next_action_guideness", "")).strip()
        a = str(obj.get("next_action", "")).strip()
        trace["guided_next_action_guideness"] = g
        trace["guided_next_action"] = a
        return g, a, trace

    def _inject_guided_think_into_prompt(self, prompt: Any, guided_think: str) -> Any:
        """Inject guided note right after assistant header for WebRL text prompts."""
        if not guided_think:
            return prompt
        if self.action_set_tag != "webrl_id":
            return prompt
        if not isinstance(prompt, str):
            return prompt
        guide = guided_think.strip()
        if not guide:
            return prompt
        # Keep current schema by inserting one comment line as assistant prefix hint.
        if not guide.startswith("#"):
            guide = f"# Note: {guide}"
        return f"{prompt}{guide}\n"

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print(
                    "WARNING: Input image provided but no image captioner available."
                )

        dcrl_meta_data = meta_data
        if isinstance(self.prompt_constructor, DCRLPromptConstructor):
            dcrl_meta_data = dict(meta_data)
            dcrl_meta_data["dcrl_episode_memory"] = self.dcrl_episode_memory
            dcrl_meta_data["dcrl_semantic_memory"] = self.dcrl_semantic_memory
            dcrl_meta_data["dcrl_think_history"] = list(self.dcrl_think_history)

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, dcrl_meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, dcrl_meta_data
            )
        prompt_before_guided = str(prompt) if isinstance(prompt, str) else ""
        guided_think = ""
        guided_next_action = ""
        guided_trace: dict[str, Any] = {
            "guided_called": False,
            "guided_reason": "budget_or_mode_not_triggered",
        }
        guided_budget = dcrl_meta_data.get("guided_think_budget")
        should_call_guided = True
        if guided_budget is not None:
            try:
                should_call_guided = int(guided_budget) > 0
            except Exception:
                should_call_guided = False
        if should_call_guided:
            guided_think, guided_next_action, guided_trace = self._build_guided_think(
                trajectory,
                intent,
                dcrl_meta_data,
                current_step_input_text=str(prompt) if isinstance(prompt, str) else "",
            )
            if guided_budget is not None:
                try:
                    dcrl_meta_data["guided_think_budget"] = max(0, int(guided_budget) - 1)
                    meta_data["guided_think_budget"] = dcrl_meta_data["guided_think_budget"]
                except Exception:
                    dcrl_meta_data["guided_think_budget"] = 0
                    meta_data["guided_think_budget"] = 0
        if self.guided_think_apply_mode == "guideness":
            prompt = self._inject_guided_think_into_prompt(prompt, guided_think)
        prompt_after_guided = str(prompt) if isinstance(prompt, str) else ""
        prev_episode_for_debug = json.loads(json.dumps(self.dcrl_episode_memory, ensure_ascii=False))
        prev_semantic_for_debug = json.loads(json.dumps(self.dcrl_semantic_memory, ensure_ascii=False))
        lm_config = self.lm_config
        n = 0
        use_guided_replay_once = (
            self.guided_think_apply_mode == "replay"
            and bool(guided_next_action)
        )
        guided_replay_applied = False
        while True:
            if use_guided_replay_once:
                response = (
                    f"{guided_think}\n{guided_next_action}".strip()
                    if guided_think
                    else guided_next_action
                )
                use_guided_replay_once = False
                guided_replay_applied = True
            else:
                if self.planner_ip is not None and self.planner_ip != "":
                    response = call_llm(lm_config, prompt, 'EMPTY', self.planner_ip)
                else:
                    response = call_llm(lm_config, prompt)
                force_prefix = self.prompt_constructor.instruction[
                    "meta_data"
                ].get("force_prefix", "")
                response = f"{force_prefix}{response}"
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == 'webrl_id':
                    action = create_webrl_id_based_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                action["parsed_action"] = parsed_response
                if bool(guided_trace.get("guided_called", False)):
                    self._guideness_dump({
                        "ts": time.time(),
                        "task_id": dcrl_meta_data.get("task_id"),
                        "step": dcrl_meta_data.get("step"),
                        "guided_apply_mode": self.guided_think_apply_mode,
                        "guided_think": guided_think,
                        "guided_next_action": guided_next_action,
                        "guided_replay_applied": guided_replay_applied,
                        "model_response": response,
                        "guided_trace": guided_trace,
                    })
                try:
                    meta_data["guided_last"] = {
                        "step": dcrl_meta_data.get("step"),
                        "guided_called": bool(guided_trace.get("guided_called", False)),
                        "guided_apply_mode": self.guided_think_apply_mode,
                        "guided_think": guided_think,
                        "guided_next_action": guided_next_action,
                        "guided_replay_applied": guided_replay_applied,
                        "guided_trace": guided_trace,
                        "model_response": response,
                        "parsed_action": parsed_response,
                        "parse_failed": False,
                    }
                except Exception:
                    pass
                if isinstance(self.prompt_constructor, DCRLPromptConstructor):
                    m_think = re.search(r"<think>\s*(.*?)\s*</think>", response, re.S | re.I)
                    self.dcrl_think_history.append(m_think.group(1).strip() if m_think else "")
                    episode, semantic = self.prompt_constructor.consume_latest_memories()
                    if isinstance(episode, dict):
                        self.dcrl_episode_memory = episode
                    if isinstance(semantic, str):
                        self.dcrl_semantic_memory = semantic
                    elif isinstance(semantic, dict):
                        # backward compatibility for old schema
                        facts = semantic.get("facts")
                        if isinstance(facts, list):
                            facts = [str(x).strip() for x in facts if str(x).strip()]
                            self.dcrl_semantic_memory = " ".join(facts[:3]).strip()
                    if self._dcrl_debug_enabled():
                        self._dcrl_debug_dump({
                            "ts": time.time(),
                            "task_id": dcrl_meta_data.get("task_id"),
                            "step": dcrl_meta_data.get("step"),
                            "intent": intent,
                            "prompt": prompt,
                            "raw_response": response,
                            "parsed_action": parsed_response,
                            "previous_episode_memory": prev_episode_for_debug,
                            "previous_semantic_memory": prev_semantic_for_debug,
                            "new_episode_memory": self.dcrl_episode_memory,
                            "new_semantic_memory": self.dcrl_semantic_memory,
                            "guided_think": guided_think,
                            "guided_next_action": guided_next_action,
                            "guided_apply_mode": self.guided_think_apply_mode,
                        })
                elif self._dcrl_debug_enabled():
                    # Keep a unified per-step prompt/response dump for non-DCRL
                    # constructors (e.g., WebRL) so rollout data can be reused
                    # for training/debugging.
                    self._dcrl_debug_dump({
                        "ts": time.time(),
                        "task_id": dcrl_meta_data.get("task_id"),
                        "step": dcrl_meta_data.get("step"),
                        "intent": intent,
                        "prompt_constructor": type(self.prompt_constructor).__name__,
                        "prompt": prompt,
                        "raw_response": response,
                        "parsed_action": parsed_response,
                        "guided_think": guided_think,
                        "guided_next_action": guided_next_action,
                        "guided_apply_mode": self.guided_think_apply_mode,
                    })
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    action["parsed_action"] = "none"
                    if bool(guided_trace.get("guided_called", False)):
                        self._guideness_dump({
                            "ts": time.time(),
                            "task_id": dcrl_meta_data.get("task_id"),
                            "step": dcrl_meta_data.get("step"),
                            "guided_apply_mode": self.guided_think_apply_mode,
                            "guided_think": guided_think,
                            "guided_next_action": guided_next_action,
                            "guided_replay_applied": guided_replay_applied,
                            "model_response": response,
                            "guided_trace": guided_trace,
                        })
                    try:
                        meta_data["guided_last"] = {
                            "step": dcrl_meta_data.get("step"),
                            "guided_called": bool(guided_trace.get("guided_called", False)),
                            "guided_apply_mode": self.guided_think_apply_mode,
                            "guided_think": guided_think,
                            "guided_next_action": guided_next_action,
                            "guided_replay_applied": guided_replay_applied,
                            "guided_trace": guided_trace,
                            "model_response": response,
                            "parsed_action": None,
                            "parse_failed": True,
                        }
                    except Exception:
                        pass
                    if isinstance(self.prompt_constructor, DCRLPromptConstructor) and self._dcrl_debug_enabled():
                        self._dcrl_debug_dump({
                            "ts": time.time(),
                            "task_id": dcrl_meta_data.get("task_id"),
                            "step": dcrl_meta_data.get("step"),
                            "intent": intent,
                            "prompt": prompt,
                            "raw_response": response,
                            "parsed_action": None,
                            "parse_failed": True,
                            "previous_episode_memory": prev_episode_for_debug,
                            "previous_semantic_memory": prev_semantic_for_debug,
                            "guided_think": guided_think,
                            "guided_next_action": guided_next_action,
                            "guided_apply_mode": self.guided_think_apply_mode,
                        })
                    elif self._dcrl_debug_enabled():
                        self._dcrl_debug_dump({
                            "ts": time.time(),
                            "task_id": dcrl_meta_data.get("task_id"),
                            "step": dcrl_meta_data.get("step"),
                            "intent": intent,
                            "prompt_constructor": type(self.prompt_constructor).__name__,
                            "prompt": prompt,
                            "raw_response": response,
                            "parsed_action": None,
                            "parse_failed": True,
                            "guided_think": guided_think,
                            "guided_next_action": guided_next_action,
                            "guided_apply_mode": self.guided_think_apply_mode,
                        })
                    break
        try:
            if "guided_last" not in meta_data:
                meta_data["guided_last"] = {
                    "step": dcrl_meta_data.get("step"),
                    "guided_called": bool(guided_trace.get("guided_called", False)),
                    "guided_apply_mode": self.guided_think_apply_mode,
                    "guided_think": guided_think,
                    "guided_next_action": guided_next_action,
                    "guided_replay_applied": guided_replay_applied,
                    "guided_trace": guided_trace,
                    "model_response": "",
                    "parsed_action": "",
                    "parse_failed": False,
                }
        except Exception:
            pass

        return action

    def reset(self, test_config_file: str) -> None:
        self.dcrl_episode_memory = {"goal": "", "milestones_done": [], "failed_attempts": []}
        self.dcrl_semantic_memory = ""
        self.dcrl_think_history = []


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
            planner_ip=args.planner_ip
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent
