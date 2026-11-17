#!/bin/sh
set -eu
# The supervisor remains PID1, owns the lifetime shared lock and reaps the child
# before removing its durable marker. Hard death leaves a blocking marker.
lock_path=${OCR_INSTALLATION_LOCK_PATH:-/coord/installation.lock}
exec python3 "$(dirname "$0")/supervision.py" --role ollama --lock "$lock_path" -- /bin/ollama "$@"
