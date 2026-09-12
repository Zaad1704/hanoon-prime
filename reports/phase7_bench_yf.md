# Phase 7 — Lean 3-Factor Benchmark (sandbox)

Side-by-side OOS walk-forward: shipped 5-factor cocktail vs the
lean 3-factor stack (vwap + momentum + relative-strength-vs-SPY)
with the RVOL>2 x 09:30-11:00 ET regime gate. Same fold scoring,
same fixtures, same deflation — only the decision path differs.

| variant | EV(R) | WR | R:R | trades | pooled SR | defl. edge | PBO | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | -0.183 | 21.1% | 2.88 | 1670 | -0.084 | -0.159 | 0.44 | FAIL |
| lean | +0.070 | 27.7% | 2.87 | 629 | -0.015 | -0.197 | 0.47 | FAIL |
| lean_no_rs | +0.004 | 25.7% | 2.90 | 657 | -0.039 | -0.200 | 0.45 | FAIL |
| lean_no_gate | -0.263 | 19.8% | 2.73 | 2325 | -0.137 | -0.203 | 0.45 | FAIL |

## Interpretation

**baseline**     — shipped cocktail (the Phase-4 FAIL numbers).
**lean**         — vwap+momentum+RS, RVOL+session gate.
**lean_no_rs**   — lean, regime gate only (RS factor off).
**lean_no_gate** — lean, RS factor only (regime gate off).

Gate WIN criterion: ``lean`` must beat ``baseline`` on deflated
edge AND pooled EV. If not, the sandbox answer is NO-GO unchanged.

## Findings (computed from the run)

- Regime gate (RVOL>2, 09:30-11:00) raises win rate **+30.0%**
  versus the ungated lean stack, while cutting OOS trades ~4x.
- SPY-relative factor adds **+7.5%** WR over gate-only — small.
- Lean WR vs shipped baseline: **+31.2%**.
- Pooled OOS R-expectancy (ALL tickers): baseline -0.183
 (1670 trades), lean +0.070
 (629 trades).
- Pooled OOS R-expectancy (admissible only): baseline
 -0.183, lean -0.027
 (294 trades, 9
 tickers) — the protocol pools only tickers with 30+ OOS
 trades, so all-ticker and admissible EV can diverge.
- Protocol verdict: lean FAIL
 (deflated edge -0.197, PBO
 0.47); baseline FAIL
 (deflated edge -0.159).

**Conclusion:** the lean stack removes +0.253R vs the shipped cocktail, turns all-ticker OOS R-expectancy positive, AND stays above baseline on the admissible subset — directional support for the subtraction thesis. A protocol GO still needs a PASS on 30+ OOS trades/ticker under the deflation bar.

Admissible floor: 30 OOS trades/ticker. Weights:
`{'vwap_deviation': 0.4, 'momentum': 0.35, 'relative_strength_spy': 0.25}`.
