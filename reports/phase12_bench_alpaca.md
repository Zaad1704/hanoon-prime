# Phase 12 — Daily Cross-Sectional Momentum (sandbox)

Long top-7 / short bottom-7 by trailing 5-session return,
dollar-neutral, held session open→close. Same WFA verdict/
deflation/PBO surface as phases 7-9; folds indexed in sessions.

## Phase 12.4 fail-fast diagnostics (pre-registered)

**Monotonicity (§12.4.1):** NO-GO

```
d1=+0.0028 d2=+0.0028 d3=+0.0019 d4=+0.0014 d5=+0.0008 d6=+0.0010 d7=-0.0008 d8=+0.0024 d9=-0.0001 d10=+0.0016
```

**Gross vs Net (§12.4.2):** gross EV = -0.043R, net EV = -0.046R — no sign flip

| metric | value |
| --- | --- |
| pooled EV (R) | -0.043 |
| pooled WR | 48.4% |
| pooled R:R | 0.98 |
| pool trades | 1008 |
| admissible EV (R) | -0.043 |
| admissible trades | 1008 |
| pooled Sharpe | -0.029 |
| deflated edge | -0.127 |
| PBO | 0.45 |
| verdict | FAIL |

## Interpretation

**verdict = PASS** only if deflated edge > 0 AND pooled Sharpe > 0
with ≥ MIN_TRADES admissible tickers. PBO < 0.5 is an analyst
override criterion, not an engine gate. PASS here does NOT arm
live capital — MicroLiveGuard still requires a paper run (see §12.7).

Detail: Deflated OOS Sharpe -0.127 <= 0 after 132 trials.
