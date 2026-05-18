#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/train_policy.sh <llamafactory-yaml>" >&2
  exit 2
fi

if [ -z "${LLAMA_FACTORY_ROOT:-}" ]; then
  echo "LLAMA_FACTORY_ROOT is required." >&2
  exit 1
fi

yaml_path="$1"
rendered_yaml="$(mktemp /tmp/scope_train_XXXXXX.yaml)"
trap 'rm -f "${rendered_yaml}"' EXIT

python - "$yaml_path" "$rendered_yaml" <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def replace(match: re.Match[str]) -> str:
    name = match.group(1)
    if name not in os.environ:
        raise SystemExit(f"Missing environment variable: {name}")
    return os.environ[name]

dst.write_text(pattern.sub(replace, src.read_text(encoding="utf-8")), encoding="utf-8")
PY

cd "${LLAMA_FACTORY_ROOT}"
llamafactory-cli train "${rendered_yaml}"
