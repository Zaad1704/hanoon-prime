# Phase 9 — Catalyst-Screen Benchmark (Path A, sandbox)

Earnings-day x pre-market volume screen over the same 180-day
Alpaca panel used by Phases 7-8. One fold protocol, same deflation,
only the per-session entry universe changes.

| variant | EV(R) | WR | R:R | trades | flagged sess | pooled SR | defl. edge | PBO | verdict |
|---|---|---|---|---|---|---|---|---|---|
| control | -0.116 | 23.7% | 2.73 | 3937 | — | -0.058 | -0.108 | 0.66 | FAIL |
| catalyst | -0.129 | 33.3% | 1.61 | 6 | 6 | +0.000 | +0.000 | 0.50 | INSUFFICIENT |
| earnings_only | -0.097 | 19.1% | 3.73 | 68 | 60 | +0.000 | +0.000 | 0.45 | INSUFFICIENT |
| rvol_only | -0.394 | 12.8% | 3.73 | 39 | 29 | +0.000 | +0.000 | 0.31 | INSUFFICIENT |

## Interpretation

**control**      — lean stack, every session (Phase-7 reference).
**catalyst**     — scheduled-earnings trading day AND pre-market RVOL>5.
**earnings_only**— any scheduled-earnings session (isolates pre-market 5x).
**rvol_only**    — pre-market RVOL>5 any day (isolates the calendar).

WIN criterion: ``catalyst`` must beat ``control`` on pooled EV AND
admissible EV AND carry a positive deflated edge. Any verdict short of
that (negatives included) keeps the sandbox answer NO-GO.

## Findings (computed from the run)

- Control (every session): pooled EV -0.116R, WR 23.7%, R:R 2.73, 3937 trades; admissible EV -0.116R.
- catalyst: EV -0.129R (vs control -0.013R); WR 33.3% (vs 23.7%); R:R 1.61; 6 trades across 6 flagged sessions; no admissible subset (0 trades clears MIN_TRADES); verdict INSUFFICIENT.
- earnings_only: EV -0.097R (vs control +0.019R); WR 19.1% (vs 23.7%); R:R 3.73; 68 trades across 60 flagged sessions; no admissible subset (0 trades clears MIN_TRADES); verdict INSUFFICIENT.
- rvol_only: EV -0.394R (vs control -0.278R); WR 12.8% (vs 23.7%); R:R 3.73; 39 trades across 29 flagged sessions; no admissible subset (0 trades clears MIN_TRADES); verdict INSUFFICIENT.

**Conclusion:** the catalyst screen did NOT overcome the negative OOS expectancy on this panel — sandbox answer remains NO-GO. With only ~2 earnings sessions per ticker per 180d, the catalyst universe is structurally too thin for the 30-trade floor; the number to fix is panel depth (years, not months), not the screen.

Admissible floor: 30 OOS trades/ticker. Weights: `{'vwap_deviation': 0.4, 'momentum': 0.35, 'relative_strength_spy': 0.25}`.
