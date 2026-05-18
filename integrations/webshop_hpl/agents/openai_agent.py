import logging
import os

import backoff
import openai
from openai import OpenAI

from .base import BaseAgent

logger = logging.getLogger("agent_eval")


class OpenAIAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        self.client = OpenAI(
            base_url=config.get("api_base", None),
            api_key=config.get("api_key", os.environ.get("OPENAI_API_KEY")),
        )

    def _max_completion_tokens(self) -> int:
        return int(self.config.get("max_completion_tokens", self.config.get("max_tokens", 512)))

    @backoff.on_exception(
        backoff.fibo,
        (
            openai.APIError,
            openai.Timeout,
            openai.RateLimitError,
            openai.APIConnectionError,
        ),
    )
    def __call__(self, messages) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=messages,
                max_completion_tokens=self._max_completion_tokens(),
                temperature=self.config.get("temperature", 0),
                stop=self.stop_words,
            )
            return response.choices[0].message.content
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                logger.warning(f"Prompt too long, skipping: {e}")
                return "[PROMPT TOO LONG]"
            logger.error(f"BadRequestError: {e}")
            raise

    def call_with_workflow(self, messages) -> str:
        new_messages = messages + [{"role": "assistant", "content": self.workflow + "Thought: "}]
        response = self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=new_messages,
            max_completion_tokens=self._max_completion_tokens(),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
        )
        return response.choices[0].message.content

    def build_prefilled_thought(self, thought: str, next_action: str = "") -> str:
        thought = str(thought or "").strip()
        next_action = str(next_action or "").strip().lower()
        if next_action.startswith("click["):
            action_prefix = "click["
        elif next_action.startswith("search["):
            action_prefix = "search["
        else:
            action_prefix = ""
        return f"Thought: {thought}\nAction: {action_prefix}"

    def call_with_prefilled_thought(self, messages, thought: str, next_action: str = "") -> str:
        prefix = self.build_prefilled_thought(thought, next_action)
        new_messages = messages + [{"role": "assistant", "content": prefix}]
        response = self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=new_messages,
            max_completion_tokens=self._max_completion_tokens(),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
        )
        completion = response.choices[0].message.content or ""
        return prefix + completion
