import json
import re
from pathlib import Path
from typing import Any, TypedDict
from PIL import Image

from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput


class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def get_lm_api_input(
        self, intro: str, examples: list[tuple[str, str]], current: str
    ) -> APIInput:

        """Return the require format for an API"""
        message: list[dict[str, str]] | str
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [{"role": "system", "content": intro}]
                for (x, y) in examples:
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": x,
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": y,
                        }
                    )
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {self.lm_config.mode}"
                )
        elif "huggingface" in self.lm_config.provider:
            # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
            if "Llama-2" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                        (
                            B_SYS + intro + E_SYS + examples[0][0],
                            examples[0][1],
                        )
                    ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")
            else:
                raise ValueError(
                    f"Huggingface models do not support model_tag {self.lm_config.gen_config['model_tag']}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f"Cannot parse action from response {response}"
            )


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )


class MultimodalCoTPromptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images
        )
        return prompt

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(
                        {
                            "role": "user" if "gpt-4o" in self.lm_config.model else "system",
                            "name": "example_user",
                            "content": [
                                {"type": "text", "text": x},
                                {
                                    "type": "text",
                                    "text": "IMAGES: (1) current page screenshot",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": pil_to_b64(example_img)
                                    },
                                },
                            ],
                        }
                    )
                    message.append(
                        {
                            "role": "user" if "gpt-4o" in self.lm_config.model else "system",
                            "name": "example_assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )

                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                for image_i, image in enumerate(images):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"({image_i+2}) input image {image_i+1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(image)},
                            },
                        ]
                    )
                content = [{"type": "text", "text": current_prompt}] + content

                message.append({"role": "user", "content": content})
                return message
            else:
                raise ValueError(
                    f"GPT-4V models do not support mode {self.lm_config.mode}"
                )
        elif "google" in self.lm_config.provider:
            if self.lm_config.mode == "completion":
                message = [
                    intro,
                    "Here are a few examples:",
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(f"Observation\n:{x}\n")
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) current page screenshot:",
                            pil_to_vertex(example_img),
                        ]
                    )
                    message.append(f"Action: {y}")
                message.append("Now make prediction given the observation")
                message.append(f"Observation\n:{current}\n")
                message.extend(
                    [
                        "IMAGES:",
                        "(1) current page screenshot:",
                        pil_to_vertex(page_screenshot_img),
                    ]
                )
                for image_i, image in enumerate(images):
                    message.extend(
                        [
                            f"({image_i+2}) input image {image_i+1}",
                            pil_to_vertex(image),
                        ]
                    )
                message.append("Action:")
                return message
            else:
                raise ValueError(
                    f"Gemini models do not support mode {self.lm_config.mode}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )


class WebRLPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                try:
                    obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]
                except:
                    print("NOTE: There is no available tokenizer, so we use characters instead of tokens for max_obs_length.")
                    obs = obs[:max_obs_length]

        turn_num = len(meta_data["action_history"])
        if turn_num == 1:
            previous_action_str = []
        else:
            previous_action_str = meta_data["action_history"][1:]

        index = turn_num - 1
        history = ""
        for i in range(index - 1, -1, -1):
            if i == 0:
                history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{intent}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{previous_action_str[i]}\n\n" + history
            else:
                history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n** Simplified html **\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{previous_action_str[i]}\n\n" + history
        if len(history) + len(obs) > (16384 - 512):
            obs = obs[:(16384 - 512)-len(history)]
        current_turn = f"Round {index}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{obs}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        prompt = f"Task Instruction: {intent}\n\n{history}{current_turn}"

        return prompt

    def extract_action(self, response: str) -> str:
        return response


class QwenWebRLPromptConstructor(PromptConstructor):
    """WebRL prompt constructor with Qwen-style ChatML packing."""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                try:
                    obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]
                except Exception:
                    print("NOTE: There is no available tokenizer, so we use characters instead of tokens for max_obs_length.")
                    obs = obs[:max_obs_length]

        turn_num = len(meta_data["action_history"])
        previous_action_str = [] if turn_num == 1 else meta_data["action_history"][1:]
        index = turn_num - 1
        parts: list[str] = [f"Task Instruction: {intent}", ""]

        for i in range(index):
            parts.extend([f"Round {i}", "", "<|im_start|>user"])
            if i == 0:
                parts.extend([intent, ""])
            else:
                parts.extend(["** Simplified html **", ""])
            parts.extend(
                [
                    "<|im_end|>",
                    "<|im_start|>assistant",
                    f"{previous_action_str[i]}",
                    "",
                ]
            )

        history = "\n".join(parts)
        if len(history) + len(obs) > (16384 - 512):
            obs = obs[: (16384 - 512) - len(history)]

        parts.extend(
            [
                f"Round {index}",
                "",
                "<|im_start|>user",
                obs,
                "",
                "<|im_end|>",
                "<|im_start|>assistant",
            ]
        )
        return "\n".join(parts)

    def extract_action(self, response: str) -> str:
        return response


class DCRLPromptConstructor(PromptConstructor):
    """DCRL prompt constructor: model outputs memory + action (v12 supports <answer> parsing)."""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.latest_episode_memory = None
        self.latest_semantic_memory = None

    def _extract_json_obj(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            try:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]
            except Exception:
                obs = obs[:max_obs_length]

        turn_num = len(meta_data["action_history"])
        index = turn_num - 1
        previous_action = "" if turn_num == 1 else meta_data["action_history"][-1]

        prev_episode = meta_data.get(
            "dcrl_episode_memory",
            {"goal": intent, "milestones_done": [], "failed_attempts": []},
        )
        prev_semantic = meta_data.get("dcrl_semantic_memory", "")

        prompt = (
            f"Task Instruction: {intent}\n"
            f"Round {index}\n\n"
            "Folded Memory:\n"
            f"episode_memory: {json.dumps(prev_episode, ensure_ascii=False)}\n"
            f"semantic_memory: {json.dumps(prev_semantic, ensure_ascii=False)}\n\n"
            f"Previous Action:\n{previous_action}\n\n"
            f"Current observation:\n{obs}"
        )
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        return self.get_lm_api_input(intro, examples, prompt)

    def extract_action(self, response: str) -> str:
        def pick_action(s: str) -> str | None:
            m = re.search(
                r"(do\s*\(.*?\)|exit\s*\(.*?\)|go_backward\s*\(\s*\)|go_forward\s*\(\s*\))",
                s,
                re.S,
            )
            return m.group(1).strip() if m else None

        # v12 preferred path: parse action from <answer>...</answer>.
        m_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.S | re.I)
        if m_answer:
            answer_text = m_answer.group(1).strip()
            picked = pick_action(answer_text)
            if picked:
                return picked
            if answer_text:
                return answer_text

        # Backward compatibility: old JSON DCRL output.
        obj = self._extract_json_obj(response)
        if isinstance(obj, dict):
            episode = obj.get("episode_memory")
            semantic = obj.get("semantic_memory")
            action = obj.get("action", "")
            if isinstance(episode, dict):
                self.latest_episode_memory = episode
            if isinstance(semantic, str):
                self.latest_semantic_memory = semantic
            elif isinstance(semantic, dict):
                facts = semantic.get("facts")
                if isinstance(facts, list):
                    facts = [str(x).strip() for x in facts if str(x).strip()]
                    self.latest_semantic_memory = " ".join(facts[:3]).strip()
            if isinstance(action, str) and action.strip():
                picked = pick_action(action.strip())
                return picked if picked else action.strip()

        # Final fallback: search full response.
        picked = pick_action(response)
        if picked:
            return picked
        raise ActionParsingError(f"Cannot parse DCRL action from response: {response[:500]}")

    def consume_latest_memories(self) -> tuple[dict[str, Any] | None, str | None]:
        episode = self.latest_episode_memory
        semantic = self.latest_semantic_memory
        self.latest_episode_memory = None
        self.latest_semantic_memory = None
        return episode, semantic


class DCRLV12PromptConstructor(DCRLPromptConstructor):
    """v12 legacy prompt packing: matches the old wrongly-trained v12 human format."""

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            try:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]
            except Exception:
                obs = obs[:max_obs_length]

        turn_num = len(meta_data["action_history"])
        index = turn_num - 1
        previous_action = "" if turn_num == 1 else meta_data["action_history"][-1]

        prompt = (
            f"Task Instruction: {intent}\n"
            f"Round {index}\n\n"
            f"Previous Action:\n{previous_action}\n\n"
            f"Current Step Input:\n{obs}"
        )
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        return self.get_lm_api_input(intro, examples, prompt)


class DCRLV13PromptConstructor(DCRLPromptConstructor):
    """v13 prompt packing: round-tokenized history string aligned with v13 rebuilt training data."""

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            try:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]
            except Exception:
                obs = obs[:max_obs_length]

        turn_num = len(meta_data["action_history"])
        index = turn_num - 1
        history_actions = [] if turn_num <= 1 else [str(x) for x in meta_data["action_history"][1:]]
        history_thinks = meta_data.get("dcrl_think_history", [])
        if not isinstance(history_thinks, list):
            history_thinks = []
        parts: list[str] = [f"Task Instruction: {intent}", ""]

        for i, act in enumerate(history_actions):
            think_text = ""
            if i < len(history_thinks):
                raw_think = str(history_thinks[i] or "").strip()
                m = re.search(r"<think>\s*(.*?)\s*</think>", raw_think, re.S | re.I)
                think_text = (m.group(1) if m else raw_think).strip()
            if not think_text:
                think_text = (
                    "[Goal] Continue the task based on instruction and current progress.\n"
                    "[Progress] Previous step completed and action is recorded in history.\n"
                    "[Plan] Execute the recorded action for that round."
                )

            parts.extend([f"Round {i}", ""])
            parts.extend(
                [
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>",
                    intent if i == 0 else "** Simplified html **",
                    "",
                ]
            )
            parts.extend(
                [
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                    f"<think>{think_text}</think>",
                    f"<answer>{act}</answer>",
                    "",
                ]
            )

        parts.extend([f"Round {index}", ""])
        parts.extend(
            [
                "<|eot_id|><|start_header_id|>user<|end_header_id|>",
                obs,
                "",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                "",
            ]
        )
        prompt = "\n".join(parts)
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        if self.lm_config.mode == "completion":
            # For v13 completion mode, keep manual packing unchanged and inline intro.
            # This avoids the generic Observation/Action wrapper used by base constructor.
            return f"{intro}\n\n{prompt}"
        return self.get_lm_api_input(intro, examples, prompt)


class DCRLQwenV13PromptConstructor(DCRLPromptConstructor):
    """v13 prompt packing for Qwen-style ChatML tokens."""

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            try:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]
            except Exception:
                obs = obs[:max_obs_length]

        turn_num = len(meta_data["action_history"])
        index = turn_num - 1
        history_actions = [] if turn_num <= 1 else [str(x) for x in meta_data["action_history"][1:]]
        history_thinks = meta_data.get("dcrl_think_history", [])
        if not isinstance(history_thinks, list):
            history_thinks = []

        parts: list[str] = [f"Task Instruction: {intent}", ""]
        for i, act in enumerate(history_actions):
            think_text = ""
            if i < len(history_thinks):
                raw_think = str(history_thinks[i] or "").strip()
                m = re.search(r"<think>\s*(.*?)\s*</think>", raw_think, re.S | re.I)
                think_text = (m.group(1) if m else raw_think).strip()
            if not think_text:
                think_text = (
                    "[Goal] Continue the task based on instruction and current progress.\n"
                    "[Progress] Previous step completed and action is recorded in history.\n"
                    "[Plan] Execute the recorded action for that round."
                )

            parts.extend([f"Round {i}", "", "<|im_start|>user"])
            parts.extend([intent if i == 0 else "** Simplified html **", "", "<|im_end|>"])
            parts.extend(
                [
                    "<|im_start|>assistant",
                    f"<think>{think_text}</think>",
                    f"<answer>{act}</answer>",
                    "",
                    "<|im_end|>",
                ]
            )

        parts.extend(
            [
                f"Round {index}",
                "",
                "<|im_start|>user",
                obs,
                "",
                "<|im_end|>",
                "<|im_start|>assistant",
                "",
            ]
        )
        prompt = "\n".join(parts)
        intro = self.instruction["intro"]
        if self.lm_config.mode == "completion":
            return f"{intro}\n\n{prompt}"
        examples = self.instruction["examples"]
        return self.get_lm_api_input(intro, examples, prompt)


class WebRLChatPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                try:
                    obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]
                except:
                    print("NOTE: There is no available tokenizer, so we use characters instead of tokens for max_obs_length.")
                    obs = obs[:max_obs_length]

        turn_num = len(meta_data["action_history"])
        if turn_num == 1:
            previous_action_str = []
        else:
            previous_action_str = meta_data["action_history"][1:]

        index = turn_num - 1
        conversations = []
        for i in range(index - 1, -1, -1):
            if i == 0:
                content_user = f"Task Instruction: {intent}\n\nRound {i}\n{intent}"
                content_assistant = f"{previous_action_str[i]}"
            else:
                content_user = f"Round {i}\n** Simplified html **"
                content_assistant = f"{previous_action_str[i]}"
            conversation = [{'role': 'user', 'content': content_user}, {'role': 'assistant', 'content': content_assistant}]
            conversations = conversation + conversations

        system_turn = [{'role': 'system', 'content': self.instruction['intro']}]
        current_turn = [{'role': 'user', 'content': f'Round {index}\n\n{obs}'}]
        conversations = system_turn + conversations + current_turn

        return conversations

    def extract_action(self, response: str) -> str:
        return response
