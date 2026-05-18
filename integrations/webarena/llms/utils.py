import argparse
import json
import os
import time
from typing import Any

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    generate_with_api,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]

def _log_raw_response(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    response: str,
) -> None:
    raw_log_path = os.environ.get("LLM_RAW_LOG_PATH")
    raw_log_dir = os.environ.get("LLM_RAW_LOG_DIR")
    if not raw_log_path and not raw_log_dir:
        return
    if not raw_log_path:
        raw_log_path = os.path.join(
            raw_log_dir, f"llm_raw_{os.getpid()}.jsonl"
        )
    try:
        dir_name = os.path.dirname(raw_log_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        payload: dict[str, Any] = {
            "ts": time.time(),
            "provider": lm_config.provider,
            "model": lm_config.model,
            "mode": lm_config.mode,
            "response": response,
        }
        if os.environ.get("LLM_RAW_LOG_INCLUDE_PROMPT") == "1":
            payload["prompt"] = prompt
        with open(raw_log_path, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    api_key = None,
    base_url = None
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
        )
    elif lm_config.provider in ["api", "finetune"]:
        args = {
            "temperature": lm_config.gen_config["temperature"],   # openai, gemini, claude
            "max_tokens": lm_config.gen_config["max_tokens"],     # openai, gemini, claude
            "top_k": lm_config.gen_config["top_p"],               # qwen
        }
        response = generate_with_api(prompt, lm_config.model, args)

    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    _log_raw_response(lm_config, prompt, response)
    return response
