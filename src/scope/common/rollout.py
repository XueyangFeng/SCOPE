from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import subprocess
from typing import Any

_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
_SENSITIVE_NAMES = ("api_key", "apikey", "token", "secret", "password")


@dataclass(frozen=True)
class CommandSpec:
    command: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def format(self) -> str:
        env_prefix = " ".join(
            f"{key}={redact_value(key, expand_env(value))}" for key, value in sorted(self.env.items())
        )
        env_prefix = f"{env_prefix} " if env_prefix else ""
        prefix = f"(cd {expand_path(self.cwd)} && " if self.cwd else ""
        suffix = ")" if self.cwd else ""
        return prefix + env_prefix + " ".join(redact_command(self.command)) + suffix


def expand_env(value: Any) -> str:
    text = str(value)

    def replace_default(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or default

    text = _ENV_DEFAULT_RE.sub(replace_default, text)
    return os.path.expanduser(os.path.expandvars(text))


def redact_value(name: str, value: str) -> str:
    lowered = name.lower().lstrip("-").replace("-", "_")
    if any(marker in lowered for marker in _SENSITIVE_NAMES) and value:
        return "<redacted>"
    return value


def redact_command(command: list[str]) -> list[str]:
    rendered: list[str] = []
    redact_next = False
    for item in command:
        text = expand_env(item)
        if redact_next:
            rendered.append("<redacted>")
            redact_next = False
            continue
        lowered = text.lower().lstrip("-").replace("-", "_")
        if text.startswith("--") and "=" in text and any(marker in lowered.split("=", 1)[0] for marker in _SENSITIVE_NAMES):
            rendered.append(f"{text.split('=', 1)[0]}=<redacted>")
            continue
        rendered.append(text)
        if text.startswith("--") and any(marker in lowered for marker in _SENSITIVE_NAMES):
            redact_next = True
    return rendered


def expand_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return Path(expand_env(value))


def append_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    text = expand_env(value).strip()
    if text:
        command.extend([name, text])


def append_flag(command: list[str], name: str, enabled: Any) -> None:
    if bool(enabled):
        command.append(name)


def run_command(spec: CommandSpec) -> int:
    env = os.environ.copy()
    env.update({key: expand_env(value) for key, value in spec.env.items()})
    cwd = expand_path(spec.cwd)
    completed = subprocess.run(
        [expand_env(item) for item in spec.command],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )
    return int(completed.returncode)
