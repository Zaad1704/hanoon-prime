# Phase 14 — PDH/PDL Sweep-and-Reclaim (sandbox)

Fade the 1-min wick through the prior session's High/Low that
reclaims inside within 10 bars, entered at the reclaim close,
stop beyond the sweep extreme, target the session Mid.
Same WFA verdict/deflation/PBO surface as phases 7-13; folds
indexed in sessions.

## Pre-registered diagnostics (reported, not gating)

**RVOL decomposition:** RVOL>1.2 EV = -0.086R (409t) vs RVOL<=1.2 EV = -0.135R (729t) — evidence surviving only the >1.2 bucket would confirm the story.

**Long/Short:** long EV = -0.048R (538t) vs short EV = -0.183R (600t).

**Time split:** pre-10:30 EV = -0.131R (677t) vs post-10:30 EV = -0.096R (461t).

| metric | value |
| --- | --- |
| pooled EV (R) | -0.099 |
| net EV (R) | -0.120 |
| pooled WR | 44.1% |
| pooled R:R | 1.04 |
| pool trades | 1100 |
| admissible EV (R) | -0.099 |
| admissible trades | 1100 |
| pooled Sharpe | -0.087 |
| deflated edge | -0.170 |
| PBO | 0.50 |
| verdict | FAIL |

## Interpretation

**verdict = PASS** only if deflated edge > 0 AND pooled Sharpe > 0
with ≥ MIN_TRADES admissible tickers. PBO < 0.5 is an analyst
override criterion, not an engine gate. PASS here does NOT arm
live capital — MicroLiveGuard still requires a paper run (§14).

Detail: Deflated OOS Sharpe -0.170 <= 0 after 18 trials.
