#!/usr/bin/env bash
# scripts/run_deep_bench.sh — Phase 10 deep-panel bake: fetch + bench all three
# sandbox stacks on multi-year IBKR data (the Phase-9 blocker was PANEL DEPTH).
#
# Idempotent: every fetch worker uses --resume (skip tickers with existing CSVs),
# so restarting after a gateway blip continues where it stopped. Benches only run
# when the panel looks complete enough.
#
# Parallelization: each fetch worker is a separate IB client with its own pacing
# budget, so wall-clock scales ~linearly with WORKERS (default 6). --all-windows
# buckets rth/pre-market/post-market from the SAME monthly pages (one pass per
# ticker instead of two), halving request count vs the old two-window runs.
#
# Usage:
#   bash scripts/run_deep_bench.sh [YEARS] [WORKERS]
#   YEARS defaults to 5, WORKERS defaults to 6.
set -euo pipefail
cd "$(dirname "$0")/.."

YEARS="${1:-5}"
WORKERS="${2:-6}"
RTH="data/research/ibkr_${YEARS}y_rth"
PRE="data/research/ibkr_${YEARS}y_premarket"
EARN="data/research/earnings"

TICKERS=$(ls data/fixtures/*_1min.csv | sed 's|.*/||; s|_1min.csv||' | sort | tr '\n' ' ')
read -r -a TARR <<< "$TICKERS"
N=${#TARR[@]}

echo "==> [1/3] ${YEARS}-year fetch (${N} tickers / ${WORKERS} workers, all-windows single-pass, resume-safe)"

workers_pids=()
BASE_ID=60
for ((w = 0; w < WORKERS; w++)); do
    subset=()
    for ((i = w; i < N; i += WORKERS)); do
        subset+=("${TARR[i]}")
    done
    if [ ${#subset[@]} -eq 0 ]; then
        continue
    fi
    LIST=$(IFS=,; echo "${subset[*]}")
    CID=$((BASE_ID + w))
    .venv/bin/python -u scripts/fetch_ibkr.py \
        --years "$YEARS" --all-windows --tickers "$LIST" \
        --resume --client-id "$CID" \
        > "logs/fetch_worker_${w}.log" 2>&1 &
    workers_pids+=("$!")
    echo "    worker $w pid=${workers_pids[${#workers_pids[@]}-1]} clientId=$CID tickers=${#subset[@]}"
done

rc=0
for pid in "${workers_pids[@]}"; do
    wait "$pid" || rc=1
done
if [ $rc -ne 0 ]; then
    echo "!! at least one fetch worker failed; tail logs/fetch_worker_*"
    exit 1
fi

echo "==> [2/3] ${YEARS}-year earnings dates via yfinance"
.venv/bin/python -u scripts/fetch_earnings.py \
    --out-dir "$EARN" --days-back $((YEARS * 365 + 60))

echo "==> [3/3] benches"
.venv/bin/python -u scripts/phase7_bench.py \
    --data-dir "$RTH" --output "reports/phase7_bench_ibkr_${YEARS}y.json"
.venv/bin/python -u scripts/phase8_bench.py \
    --data-dir "$RTH" --output "reports/phase8_bench_ibkr_${YEARS}y.json"
.venv/bin/python -u scripts/phase9_bench.py \
    --rth-dir "$RTH" --premkt-dir "$PRE" --earnings-dir "$EARN" \
    --output "reports/phase9_bench_ibkr_${YEARS}y.json"

echo "==> DONE. Reports in reports/phase{7,8,9}_bench_ibkr_${YEARS}y.*"