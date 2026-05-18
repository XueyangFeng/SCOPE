from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0


def load_llm_config(config: dict[str, Any]) -> LLMConfig:
    raw = dict(config.get("llm") or {})
    enabled = bool(raw.get("enabled", False))
    api_key = str(raw.get("api_key") or os.environ.get("OPENAI_API_KEY") or "")
    base_url = str(raw.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    model = str(raw.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
    return LLMConfig(
        enabled=enabled,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 4096)),
        timeout=float(raw.get("timeout", 60.0)),
    )


class ChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete_json(self, prompt: str) -> str:
        if not self.config.api_key:
            raise RuntimeError("LLM is enabled but no API key was provided. Set OPENAI_API_KEY or llm.api_key.")
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        req = Request(
            self.config.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        return str(data["choices"][0]["message"]["content"])
