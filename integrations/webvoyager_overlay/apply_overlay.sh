#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash integrations/webvoyager_overlay/apply_overlay.sh <WebVoyager-root>" >&2
  exit 2
fi

target_root="$1"
overlay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/overlay" && pwd)"
file_list="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/OVERLAY_FILES.txt"

if [ ! -d "$target_root" ]; then
  echo "WebVoyager root not found: $target_root" >&2
  exit 1
fi

while IFS= read -r relpath; do
  [ -z "$relpath" ] && continue
  mkdir -p "$target_root/$(dirname "$relpath")"
  cp "$overlay_root/$relpath" "$target_root/$relpath"
done < "$file_list"

echo "Applied Scope WebVoyager overlay to $target_root"
