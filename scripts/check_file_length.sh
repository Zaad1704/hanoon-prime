#!/usr/bin/env bash
# scripts/check_file_length.sh — Enforce R3: no source file > 200 lines.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)/src"
# Files with pre-existing line-count violations from the v2.0 restructure.
# FROZEN via R28 (tests/test_contract.py::FROZEN_FILE_SKIP) — growing this
# list to dodge the cap is a governance violation, not a bug fix.
SKIP="hands.py|validator.py|telemetry.py|halim_adapter.py|ib_cycle.py|orchestrator.py|ib_executor.py|ib_streamer.py|ironclad.py|consolidation.py|shadow_book.py|ib_adapter.py|realized_ev.py|config.py|immune.py|ev_gate.py|risk.py|exits.py|stdp.py|wfa.py|ablation.py|micro_live.py|phase7.py|phase8.py|phase9.py"
VIOLATIONS=$(find "$SRC" -name "*.py" -exec wc -l {} + | grep -v ' total$' | grep -Ev "$SKIP" | awk '$1 > 200 {print "FAIL: " $2 " has " $1 " lines (max 200)"}')
if [ -n "$VIOLATIONS" ]; then
  echo "R3b VIOLATION — files exceeding 200 lines:"
  echo "$VIOLATIONS"
  exit 1
fi
echo "R3b OK — all files within 200-line limit"
