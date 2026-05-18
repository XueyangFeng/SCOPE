#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash integrations/webarena/apply_overlay.sh /path/to/VAB-WebArena-Lite" >&2
    exit 2
fi

target_root="$1"
overlay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${target_root}" ]]; then
    echo "Target directory not found: ${target_root}" >&2
    exit 1
fi
if [[ ! -f "${target_root}/run.py" || ! -d "${target_root}/agent" ]]; then
    echo "Target does not look like VAB-WebArena-Lite: ${target_root}" >&2
    exit 1
fi

while IFS= read -r relpath; do
    [[ -z "${relpath}" ]] && continue
    mkdir -p "${target_root}/$(dirname "${relpath}")"
    cp -f "${overlay_root}/${relpath}" "${target_root}/${relpath}"
done < "${overlay_root}/OVERLAY_FILES.txt"

echo "Applied Scope WebArena overlay to ${target_root}"
