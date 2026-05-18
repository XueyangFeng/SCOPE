import re
import json
import logging
from typing import Tuple

from envs import BaseEnv
from tasks import WebShopTask
from prompt import prompt_with_icl
from utils.datatypes import State
from webshop.web_agent_site.envs import WebAgentTextEnv


logger = logging.getLogger("agent_eval")


class WebShopEnv(BaseEnv):
    def __init__(
        self,
        task: WebShopTask,
        env: WebAgentTextEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: WebShopTask = task
        self.session_id = self.task.session_id
        self.session = {}
        self.env = env
        
        self.state = State()
        self.incorporation_type = kwargs.get("incorporation_type", "no_icl")
        self.icl_num = int(kwargs.get("icl_num", 0) or 0)
        raw_without_icl = kwargs.get("without_icl")
        if raw_without_icl is None:
            self.without_icl = self.icl_num <= 0
        else:
            self.without_icl = bool(raw_without_icl)
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        matches = pattern.findall(llm_output)
        if matches:
            action = matches[0]
        else:
            action = "no action"
        assert action is not None
        return action
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
        except:
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = 0
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state
        try:
            observation, reward, done, info = self.env.step(action=action)
            observation = f"Observation: {observation}"
            # available_actions = self.env.get_available_actions()
            # observation = f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
        except AssertionError:
            observation = 'Observation: Invalid action!'
            done = False

        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        # Persist env diagnostics on every step so failed / truncated runs
        # still retain search statistics for downstream memory processing.
        merged_info = dict(self.state.info or {})
        if info:
            merged_info.update(info)
        self.state.info = merged_info

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = 0

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward
        else:
            self.state.reward = reward

        return observation, self.state
    
    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.env.reset(self.session_id)
        cur_task = self.env.observation
        observation, messages = prompt_with_icl(
            instruction=self.instruction, 
            raw_icl=self.raw_icl, 
            cur_task=cur_task, 
            icl_num=self.icl_num,
            workflow=None,
            without_icl=self.without_icl
        )
        if self.incorporation_type == "no_icl":
            self.state.history = messages
        else:
            if self.icl_format == 'first':
                self.state.history.append({
                    "role": "user",
                    "content": observation,
                })
            elif self.icl_format == 'conversation':
                self.state.history = messages
        return observation, self.state
