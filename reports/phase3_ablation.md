# Phase 3 — Per-Organ Ablation Report

Data: `data/fixtures` · pool = mean over tickers with trades
Baseline: EV/trade `-0.317R`, trades `397`, WR `19.2%`, R:R `2.21`

Marginal contribution = baseline_ev − variant_ev.
**Positive** = component carries edge (removing it hurt).
**Negative** = component harmful (removing it helped).
**~0** = inert.

## Factor contribution ranking (cerebellum alpha)

| factor | contribution (R) | off-EV | trades |
|---|---|---|---|
| orderbook_imbalance | +0.208 | -0.525 | 393 |
| momentum | +0.079 | -0.396 | 466 |
| institutional_flow | +0.077 | -0.395 | 579 |
| vwap_deviation | -0.074 | -0.243 | 272 |
| vpin | -0.115 | -0.203 | 367 |

## Organ toggles (all variants vs baseline)

| variant | EV/trade | ΔEV | trades | WR | R:R | sharp | DD | ret% |
|---|---|---|---|---|---|---|---|---|
| baseline | -0.317 | — | 397 | 19.2% | 2.21 | -0.65 | -313.39% | -1.8% |
| off_vpin | -0.203 | -0.115 | 367 | 20.7% | 2.75 | -0.24 | -213.62% | -1.2% |
| off_orderbook_imbalance | -0.525 | +0.208 | 393 | 15.4% | 1.85 | -0.85 | -292.56% | -3.1% |
| off_institutional_flow | -0.395 | +0.077 | 579 | 18.6% | 2.19 | -0.44 | -188.62% | -4.0% |
| off_momentum | -0.396 | +0.079 | 466 | 18.9% | 2.08 | -0.70 | -1628.90% | -3.2% |
| off_vwap_deviation | -0.243 | -0.074 | 272 | 23.6% | 1.87 | -0.04 | -429.19% | -0.6% |
| off_all_alpha | +0.000 | -0.317 | 0 | 0.0% | 0.00 | 0.00 | 0.00% | +0.0% |
| off_eyes | -0.288 | -0.029 | 376 | 21.1% | 2.28 | -0.34 | -3577.44% | -1.9% |
| threshold_flat | -0.374 | +0.056 | 807 | 18.5% | 2.18 | -0.39 | -909.93% | -5.0% |
| short_off | -0.441 | +0.124 | 215 | 15.2% | 2.04 | -1.29 | -376.76% | -1.4% |
| stop_tight | -0.469 | +0.152 | 418 | 10.6% | 3.04 | -2.37 | -1003.95% | -1.6% |
| stop_wide | -0.102 | -0.216 | 297 | 36.0% | 1.40 | -0.97 | -290.64% | -0.4% |
| target_off | -0.492 | +0.175 | 210 | 2.8% | 3.90 | -3.49 | -6.59% | -2.1% |
| timeout_30 | -0.284 | -0.033 | 556 | 27.3% | 1.49 | -0.53 | -497.11% | -1.8% |
| flat_sizing | -0.387 | +0.070 | 397 | 19.2% | 1.82 | -1.03 | -3025.27% | -3.1% |

## Caveat

Committed fixtures span only ~5 trading days (~1,900 bars / ticker).
Per-ticker trade counts are low; these deltas rank *direction of*
marginal contribution, they do not reach Phase-2 statistical
significance on their own. The WF gate remains the verdict on
whether any variant is *actually* better.

Sharpe is a mean across per-ticker sharpe (clipped to ±20; near-flat
equity curves blow up otherwise). Drawdown is mean per-ticker max
drawdown in per-share terms, so it can exceed 100% (sizing scaled up
through a negative equity drift) and is *not* a portfolio drawdown.
