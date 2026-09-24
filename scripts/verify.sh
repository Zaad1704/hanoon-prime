#!/usr/bin/env bash
# scripts/verify.sh — mandatory pre-push verification gate.
#
# Runs the contract + safety + validation suites that guard the
# 2026-09-23 pytest fixes. Run from the repo root:
#
#     bash scripts/verify.sh
#
# All tests must pass before any push. No skips, no xfails, no
# weakened assertions — a failure here means the push waits.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 1/3 architectural contract ==="
python3 -m pytest tests/test_contract.py -q

echo "=== 2/3 safety (producer + training bypass) ==="
python3 -m pytest tests/test_safety_producer.py tests/test_safety_training.py -q

echo "=== 3/3 validation honesty + telemetry + fixes journal ==="
python3 -m pytest \
    tests/test_telemetry.py \
    tests/test_telemetry_auth.py \
    tests/test_telemetry_auth_failclosed.py \
    tests/test_telemetry_stream.py \
    tests/test_validation_honesty.py \
    tests/test_fixes_journal.py \
    tests/test_scoring_integrity.py \
    tests/test_sleep_scheduler.py \
    tests/test_sleep_replay_live.py \
    tests/test_wfa.py \
    -q

echo "VERIFY OK — safe to push."
