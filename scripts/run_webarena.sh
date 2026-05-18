#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m scope.webarena.run_round --stage "${1:-all}" --dry-run
