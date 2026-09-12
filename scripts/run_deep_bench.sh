#!/usr/bin/env bash
# scripts/run_deep_bench.sh — Phase 10 deep-panel bake: fetch + bench all three
# sandbox stacks on multi-year IBKR data (the Phase-9 blocker was PANEL DEPTH).
#
# Idempotent: every fetch step uses --resume (skip existing CSVs), so restarting
# after a gateway blip continues where it stopped. Benches only run when the
# panel looks complete enough (RTH sessions > 400 per ticker on average).
#
# Usage:
#   bash scripts/run_deep_bench.sh [YEARS]
#   YEARS defaults to 5.
set -euo pipefail
cd "$(dirname "$0")/.."

YEARS="${1:-5}"
RTH="data/research/ibkr_${YEARS}y_rth"
PRE="data/research/ibkr_${YEARS}y_premarket"
EARN="data/research/earnings"

echo "==> [1/4] 5-year RTH panel (resume-safe)"
.venv/bin/python scripts/fetch_ibkr.py \
    --years "$YEARS" --window rth --out-dir "$RTH" --resume --client-id 31

echo "==> [2/4] 5-year pre-market panel (resume-safe)"
.venv/bin/python scripts/fetch_ibkr.py \
    --years "$YEARS" --window pre-market --out-dir "$PRE" --resume --client-id 32

echo "==> [3/4] 5-year earnings dates via yfinance"
.venv/bin/python scripts/fetch_earnings.py \
    --out-dir "$EARN" --days-back $((YEARS * 365 + 60))

echo "==> [4/4] benches"
.venv/bin/python scripts/phase7_bench.py \
    --data-dir "$RTH" --output "reports/phase7_bench_ibkr_${YEARS}y.json"
.venv/bin/python scripts/phase8_bench.py \
    --data-dir "$RTH" --output "reports/phase8_bench_ibkr_${YEARS}y.json"
.venv/bin/python scripts/phase9_bench.py \
    --rth-dir "$RTH" --premkt-dir "$PRE" --earnings-dir "$EARN" \
    --output "reports/phase9_bench_ibkr_${YEARS}y.json"

echo "==> DONE. Reports in reports/phase{7,8,9}_bench_ibkr_${YEARS}y.*"