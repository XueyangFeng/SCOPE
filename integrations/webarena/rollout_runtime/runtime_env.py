from __future__ import annotations

import json
import os
import shlex


def env_text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or default)


def build_runtime_env() -> dict[str, str]:
    env = {
        "DATASET": env_text("DATASET", "webarena"),
        "REDDIT": env_text("REDDIT", "http://localhost:8080"),
        "SHOPPING": env_text("SHOPPING", "http://localhost:8082"),
        "SHOPPING_ADMIN": env_text("SHOPPING_ADMIN", "http://localhost:8083/admin"),
        "GITLAB": env_text("GITLAB", "http://localhost:9001"),
        "WIKIPEDIA": env_text("WIKIPEDIA", "http://localhost:8081"),
        "MAP": env_text("MAP", "https://www.openstreetmap.org"),
        "HOMEPAGE": env_text("HOMEPAGE", "http://localhost:4399"),
        "OPENAI_API_URL": env_text("OPENAI_API_URL"),
        "OPENAI_API_KEY": env_text("OPENAI_API_KEY"),
        "OPENAI_EVAL_API_URL": env_text("OPENAI_EVAL_API_URL"),
        "OPENAI_EVAL_API_KEY": env_text("OPENAI_EVAL_API_KEY"),
        "LLM_RAW_LOG_DIR": env_text("LLM_RAW_LOG_DIR"),
        "GUIDENESS_DIR": env_text("GUIDENESS_DIR"),
        "FULL_RESET": env_text("FULL_RESET", env_text("FULL_RESET_URL")),
        "FULL_STATUS_URL": env_text("FULL_STATUS_URL"),
        "FULL_READY_TEXT": env_text("FULL_READY_TEXT", "Ready for duty!"),
        "NO_PROXY": env_text("NO_PROXY", "localhost,127.0.0.1,0.0.0.0"),
    }
    proxy_url = env_text("PROXY_URL")
    if proxy_url:
        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
        env["ALL_PROXY"] = proxy_url
    else:
        env["HTTP_PROXY"] = ""
        env["HTTPS_PROXY"] = ""
        env["ALL_PROXY"] = ""
    return env


def build_export_string() -> str:
    env = build_runtime_env()
    parts: list[str] = []
    for key, value in env.items():
        if value == "" and key in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
            parts.append(f"unset {key}")
        else:
            parts.append(f"export {key}={shlex.quote(value)}")
    return "; ".join(parts)


def main() -> int:
    mode = env_text("RUNTIME_ENV_OUTPUT", "shell")
    if mode == "json":
        print(json.dumps(build_runtime_env(), ensure_ascii=False))
    else:
        print(build_export_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
