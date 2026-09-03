#!/usr/bin/env bash
# launcher for the detached HALIM serve (see halim_start.sh)
set -uo pipefail
# Halim self-contained in the HANOON PRIME repo — see halim_start.sh.
HALIM_REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd "$HALIM_REPO" || exit 1
# shellcheck source=/dev/null
source "$HALIM_REPO/scripts/halim_env.sh"
exec bash "$HALIM_REPO/scripts/halim_serve.sh" --host 127.0.0.1 --port "${HALIM_SERVE_PORT:-8765}"
