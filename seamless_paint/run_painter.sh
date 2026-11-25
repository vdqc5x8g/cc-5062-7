#!/usr/bin/env bash
# Convenience launcher for the seamless pattern painting window.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8000}"
exec python3 -m seamless_paint.webapp --host 127.0.0.1 --port "$PORT" --output output
