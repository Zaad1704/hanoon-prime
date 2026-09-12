# Phase 8 — Staged-Exit Benchmark (sandbox)

Side-by-side OOS walk-forward on ONE panel (default: 180-day Alpaca
1-min, 23 tickers): the lean stack with STATIC exits (the Phase-7
reference) vs three staged-exit heads. Same fold scoring, same
gates, same deflation — only the exit policy differs.

| variant | EV(R) | WR | R:R | trades | pooled SR | defl. edge | PBO | verdict |
|---|---|---|---|---|---|---|---|---|
| control | -0.116 | 23.7% | 2.73 | 3937 | -0.058 | -0.108 | 0.66 | FAIL |
| stage_atr | -0.166 | 46.6% | 0.79 | 4612 | -0.128 | -0.175 | 0.52 | FAIL |
| stage_vwap | -0.176 | 49.2% | 0.67 | 5326 | -0.167 | -0.213 | 0.48 | FAIL |
| stage_nobe | -0.189 | 39.1% | 1.08 | 4400 | -0.127 | -0.176 | 0.42 | FAIL |

## Interpretation

**control**     — lean stack, static ATR stop/target exits.
**stage_atr**   — 50% scale @ 1.5xATR, stop->BE, runner trails 3xATR.
**stage_vwap**  — 50% scale @ 1.5xATR, stop->BE, runner trails VWAP.
**stage_nobe**  — stage_atr WITHOUT the breakeven floor.

Exit WIN criterion: a staged arm must beat ``control`` on deflated
edge AND pooled EV (all-ticker) AND on the admissible subset. If
not, the sandbox answer is NO-GO unchanged.

## Findings (computed from the run)

- Control (lean static) pooled EV: -0.116R
 (3937 trades); admissible EV -0.116R.
- Protocol verdict (control): FAIL
 (deflated edge -0.108).
- stage_atr: EV -0.166R (vs control -0.050R); WR 46.6% (vs control 23.7%); R:R 0.79 (vs control 2.73); admissible EV -0.166R (below control by -0.050R); verdict FAIL.
- stage_vwap: EV -0.176R (vs control -0.060R); WR 49.2% (vs control 23.7%); R:R 0.67 (vs control 2.73); admissible EV -0.176R (below control by -0.060R); verdict FAIL.
- stage_nobe: EV -0.189R (vs control -0.073R); WR 39.1% (vs control 23.7%); R:R 1.08 (vs control 2.73); admissible EV -0.189R (below control by -0.073R); verdict FAIL.

**Conclusion:** no staged arm beats the static control on pooled OOS R-expectancy with a positive edge — staged exits do NOT clear the gate on this set; sandbox answer remains NO-GO (data, not the exit policy, is the bound). Every staged arm roughly DOUBLED win rate (control 23.7% -> best 49.2%) but cut R:R ~3:1 -> ~0.7, so EV fell — the win-rate thesis is mechanically real and EV-negative. The bound is that the entrance edge is not positive, not the exit policy.

Admissible floor: 30 OOS trades/ticker. Weights:
`{'vwap_deviation': 0.4, 'momentum': 0.35, 'relative_strength_spy': 0.25}`.
