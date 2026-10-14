#!/usr/bin/env bash
# Overnight local-model batch. Needs a ComfyUI server with the local video model on :8199.
# Resumable: re-run to continue; one Ctrl-C finishes the current clip, then stops.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
curl -s -m3 localhost:8199/system_stats >/dev/null || { echo "local model server on :8199 is not up"; exit 1; }
cd "$H/../.."
exec python3 tools/h3_queue.py "${1:-$H/queue.example.json}" --log-dir "$H/logs"
