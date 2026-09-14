#!/usr/bin/env python3
"""Diagnostic: where the sizing bar sits in the cortex score distribution.

The entry sizing path (``orchestrator._size_entry``) refuses every candidate
whose ``abs(score) <= dynamics.threshold * patience`` (scalp patience=1.0,
so effectively ``threshold`` = 0.45). ``not_sized`` is the dominant HOLD
reason in the live journal — this script quantifies *how much* of the
cortex score distribution the bar cuts off, and the counterfactual pass
rate at a lower bar (0.15).

Journal verdicts are the only persisted source of cortex scores, and only
score-bearing verdicts survive (``direction_rejected``, ``low_penny_score``,
``admitted``, ``daily_loss_limit``); ``not_sized``/``no_signal`` drop their
score at the Verdict boundary, so this is a lower bound on the population
the bar actually gates.

Usage:
    python scripts/diag_threshold_bar.py [--journal PATH] [--last N]
        [--thresholds 0.45,0.30,0.20,0.15,0.10]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

DEFAULT_JOURNAL = Path("runtime/journal_live.jsonl")


def load_verdicts(journal: Path, last: int | None) -> list[dict]:
    """Read ``verdict`` events from the journal (optionally only the last N)."""
    verdicts: list[dict] = []
    with journal.open("r", errors="replace") as fh:
        for line in fh:
            if '"verdict"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "verdict":
                continue
            verdicts.append(rec)
    return verdicts[-last:] if last else verdicts


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    ap.add_argument("--last", type=int, default=None, help="only the last N verdicts")
    ap.add_argument(
        "--thresholds",
        default="0.45,0.30,0.20,0.15,0.10",
        help="comma-separated sizing bars to test",
    )
    args = ap.parse_args()

    if not args.journal.exists():
        raise SystemExit(f"journal not found: {args.journal}")

    verdicts = load_verdicts(args.journal, args.last)
    reasons = collections.Counter(v.get("reason") for v in verdicts)
    scored = [
        float(v["score"]) for v in verdicts if abs(float(v.get("score") or 0)) > 0
    ]
    abs_scores = sorted(abs(s) for s in scored)
    enters = sum(1 for v in verdicts if v.get("action") == "ENTER")

    print(f"journal          : {args.journal}")
    print(f"verdicts         : {len(verdicts)}")
    print(
        f"score-bearing    : {len(scored)} ({len(scored) / max(len(verdicts), 1):.1%})"
    )
    print(f"admitted ENTERs  : {enters}")
    print(f"top HOLD reasons : {reasons.most_common(6)}")

    if not abs_scores:
        raise SystemExit("no score-bearing verdicts to analyze")

    q = lambda p: _percentile(abs_scores, p)  # noqa: E731
    print(
        "\n|score| distribution (scored population):\n"
        f"  p25={q(.25):.3f}  p50={q(.50):.3f}  p75={q(.75):.3f}  "
        f"p90={q(.90):.3f}  p95={q(.95):.3f}  p99={q(.99):.3f}  max={abs_scores[-1]:.3f}"
    )

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    print(
        f"\n{'bar':>6s} {'pass':>8s} {'pass%':>7s} {'bar pctile':>11s} {'factor vs 0.45':>15s}"
    )
    base = sum(1 for s in abs_scores if s > thresholds[0])
    for thr in thresholds:
        passed = sum(1 for s in abs_scores if s > thr)
        pct = passed / len(abs_scores)
        bar_pctile = sum(1 for s in abs_scores if s <= thr) / len(abs_scores)
        factor = passed / base if base else float("inf")
        print(f"{thr:6.2f} {passed:8d} {pct:7.1%} {bar_pctile:11.1%} {factor:14.1f}x")

    print(
        "\nRead: the bar percentile is the share of real cortex convictions the\n"
        "sizing gate silently discards; the factor is how many more candidates\n"
        "clear a lower bar relative to the live 0.45."
    )


if __name__ == "__main__":
    main()
