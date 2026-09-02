#!/usr/bin/env bash
# scripts/check_file_length.sh — Enforce R3: no source file > 200 lines.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)/src"
VIOLATIONS=$(find "$SRC" -name "*.py" -exec wc -l {} + | grep -v ' total$' | awk '$1 > 200 {print "FAIL: " $2 " has " $1 " lines (max 200)"}')
if [ -n "$VIOLATIONS" ]; then
  echo "R3b VIOLATION — files exceeding 200 lines:"
  echo "$VIOLATIONS"
  exit 1
fi
echo "R3b OK — all files within 200-line limit"
