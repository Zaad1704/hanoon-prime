#!/usr/bin/env bash
# M. A. Halim local server — active runtime (status + learn + write). Not inference-only.
# Reflex (PPO/proxy) always stays inline in HANOON; this never replaces fast path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/halim_env.sh"

# Use venv python explicitly (exec python may resolve to system python without mlx)
_PYTHON="python3"
if [[ -f "$ROOT/.venv/bin/python" ]]; then
    _PYTHON="$ROOT/.venv/bin/python"
elif [[ -d "$ROOT/venv" ]]; then
    source "$ROOT/venv/bin/activate"
fi

# Verify mlx is available in the chosen python
if ! "$_PYTHON" -c "import mlx; import mlx_lm" 2>/dev/null; then
    echo "⚠️  mlx/mlx-lm not installed in $_PYTHON — installing..."
    "$_PYTHON" -m pip install mlx mlx-lm "transformers<5.1.0" 2>/dev/null || true
fi

export HALIM_LM_BACKEND="${HALIM_LM_BACKEND:-mlx}"

# ── STRICT single-instance rule ──────────────────────────────────────────────
# Only ONE Halim serve may run — two 4B MoE processes crash the device (OOM).
# If a serve is already healthy, do NOT start a second one; just exit 0.
if curl -sf --max-time 2 http://127.0.0.1:${HALIM_SERVE_PORT:-8765}/health >/dev/null 2>&1; then
    echo "✅ Halim serve already running on :${HALIM_SERVE_PORT:-8765} — single-instance, skipping."
    exit 0
fi
# A stale serve that isn't healthy? Kill it before starting fresh.
if pgrep -f "halim/halim/serve.py" >/dev/null 2>&1; then
    echo "⚠️  Halim serve process exists but not healthy — killing stale instance."
    pkill -9 -f "halim/halim/serve.py" 2>/dev/null || true
    sleep 2
fi

# 🧠 Low-priority CPU scheduling — Halim inference must not starve the trading loop.
# nice -n 19 tells macOS: "only give Halim CPU cycles when the trading system is idle."
exec nice -n 19 "$_PYTHON" "$ROOT/halim/halim/serve.py" "$@"
