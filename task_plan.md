# HANOON PRIME — Phases 2–5: From Honest Gate to Validated Live Edge

## Goal
Carry the Phase-1 state (real committed data, non-vacuous R2 gate, CI red by design)
through:
- **Phase 2** — anti-overfit statistics (walk-forward / purge-embargo / deflated
  Sharpe / PBO) so ANY EV claim carries statistical validity.
- **Phase 3** — per-organ ablation to find which signal actually carries edge.
- **Phase 4** — paper incubation with pre-locked stats (no goalposts moved after).
- **Phase 5** — micro-live deployment guarded by the latched kill switch.
Deliverable: honest verdict + recommendations doc, not a false "it works".

## Non-negotiable constraints
- R2 gate stays strict: CI red is a *signal*, not a bug. Goal is to make it green
  with *validated* edge, not to weaken the gate.
- No monkeypatching denylisted singletons (anti-monkeypatch guard is CI-enforced).
- `data/market_data` is gitignored (live runtime); `data/fixtures` is committed.
- Phase 1 = only ~5 trading days committed — statistically thin. Phase 2 must
  structure claims honestly around that, and Phase 4/5 need a data plan.

## Decisions so far
- Gate policy (user): Enforce strictly — CI red now. Blocking == intended signal.
- Data strategy (user): Commit what exists now (~5d), acquire deeper later.

---

## Phase 2 — Anti-Overfit Statistics
### 2.1 Walk-forward validation harness
- Build WFA: fixed roll window train → next-window test, per ticker & cross-sectional.
  Edge = only counted on held-out windows, never the training window.
- Purge/embargo: drop `purge` bars around train/test boundary + `embargo` tail bars
  (1-min data → purge+embargo in bars; document choice).
### 2.2 Deflated Sharpe / PBO
- Deflated Sharpe (Bailey/López de Prado): deflate the backtest Sharpe by #trials.
- PBO (probability of backtest overfitting): CSCV over the WFA folds.
### 2.3 Statistical verdict per ticker + universe
- Min-trade floor (e.g., ≥ 30 trades) before a ticker's EV is admissible; otherwise
  "INSUFFICIENT DATA" — which must FAIL the gate, not pass silently.
- Output: `reports/phase2_wfa.json` + human-readable `reports/phase2_wfa.md`.
### 2.4 Enforcement
- CI job `wf-gate`: runs WFA above on committed fixtures; fails on insufficient data
  or non-positive deflated edge. Gate stays honest.
Status: pending

## Phase 3 — Per-Organ Ablation
### 3.1 Confound ablation
- Each organ (cerebellum alpha, cortex, hippocampus recency weights, score→winprob
  mapping, eyes features, hands ATR stop/target/timeout params) toggled OFF one at a
  time. Only delta vs baseline counts. No simultaneous changes.
### 3.2 Which signal carries edge?
- Ablate each alpha factor individually; rank factors by marginal contribution.
- Determine whether current edge (if any) survives slippage/fees already in place.
### 3.3 Verdict
- Report which organs are additive, which are inert/harmful. This drives Phase 4
  (trade only what's proven additive).
Status: pending

## Phase 4 — Paper Incubation (pre-locked stats)
### 4.1 Lock the evaluation protocol BEFORE trading
- Pre-register: exact window, min trades, WFA folds, deflated-Sharpe threshold,
  what "pass" means. Written down; cannot be changed after.
### 4.2 Paper harness on live-feed data
- Route IB paper account data (or recorded live feed) through the same organs;
  evaluate against pre-locked protocol.
### 4.3 Release criteria
- Floor: e.g., ≥ N trades over ≥ M sessions AND deflated Sharpe above threshold
  AND no 3-day observation where kill-switch would have fired in paper.
Status: pending

## Phase 5 — Micro-Live (guarded)
### 5.1 Smallest possible live footprint
- 1 ticker, minimal notional (MAX_POSITION_NOTIONAL = $5k is a *cap*, not a start).
- Latched kill switch active (KILL_DAILY_LOSS_LIMIT=$500). Halt on $200.
### 5.2 Shadow the paper calls 1:1
- Live orders must match paper signals exactly; divergence = investigate.
### 5.3 Completion gate
- Only after Phase 4 release criteria met. Live results feed the realized-EV
  learning loop (which is already wired); no new unproven organs go live.
Status: pending

## Phase 6 — Recommendations Doc
- `reports/recommendations.md`: what's proven, what isn't, go/no-go for real money.

---

## Next Step
Start Phase 2.1: build the walk-forward harness on committed fixtures with
purge/embargo, min-trade floor, and honest per-ticker verdicts.