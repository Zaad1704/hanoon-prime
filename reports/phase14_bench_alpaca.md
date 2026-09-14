# Phase 14 — PDH/PDL Sweep-and-Reclaim (sandbox)

Fade the 1-min wick through the prior session's High/Low that
reclaims inside within 10 bars, entered at the reclaim close,
stop beyond the sweep extreme, target the session Mid.
Same WFA verdict/deflation/PBO surface as phases 7-13; folds
indexed in sessions.

## Pre-registered diagnostics (reported, not gating)

**RVOL decomposition:** RVOL>1.2 EV = -0.037R (185t) vs RVOL<=1.2 EV = -0.198R (252t) — evidence surviving only the >1.2 bucket would confirm the story.

**Long/Short:** long EV = -0.179R (209t) vs short EV = -0.085R (228t).

**Time split:** pre-10:30 EV = -0.104R (235t) vs post-10:30 EV = -0.163R (202t).

| metric | value |
| --- | --- |
| pooled EV (R) | -0.104 |
| net EV (R) | -0.126 |
| pooled WR | 44.7% |
| pooled R:R | 1.00 |
| pool trades | 273 |
| admissible EV (R) | +0.000 |
| admissible trades | 0 |
| pooled Sharpe | +0.000 |
| deflated edge | +0.000 |
| PBO | 0.31 |
| verdict | INSUFFICIENT |

## Interpretation

**verdict = PASS** only if deflated edge > 0 AND pooled Sharpe > 0
with ≥ MIN_TRADES admissible tickers. PBO < 0.5 is an analyst
override criterion, not an engine gate. PASS here does NOT arm
live capital — MicroLiveGuard still requires a paper run (§14).

Detail: No ticker reached the OOS min-trade floor — needs more data.
