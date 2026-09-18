# LIVE CHECK — HANOON PRIME neuromorphic organ tracker

> **Purpose**: one living document that tracks, for every organ: what it actually does today
> (verified, with file:line), why it is off/inert, what the literature says, and the concrete
> work needed to make it live — plus the money-moving core it must plug into.
> **Last audit**: 2026-09-18. **How to use**: tick checkboxes as work lands; every item has a
> "verify by" line — do not mark done without the verification passing.

---

## 0. The money-moving core (do not break this)

Verified live path, in order:

1. **Signal**: `_evaluate_fast → _score_pipeline` — 27-indicator alpha
   (`src/hanoon_prime/brain/cerebellum.py:134-166`) → Cortex verdict
   (`cortex.evaluate`, `orchestrator.py:1112`) → blend
   `0.7·cortex + 0.3·static-SNN` (`orchestrator.py:1117`, `NEURO_BLEND=0.3` at `:85`) →
   Nash bounded penalty (`:629-653`) → stabilization (`:1123`).
2. **Gates**: `_apply_fast_gates` — session/direction/penny/halt
   (`orchestrator.py:375-411`); governor (2 entries/cycle); portfolio cap;
   hard limits `$5,000` notional / `$50` per trade / 3 concurrent / `$200` daily /
   `$500` latched kill (`src/hanoon_prime/immune.py:58-64`).
3. **Execution**: `IBExecutor.place_bracket` MKT entry + STP + LMT, DAY, outsideRth
   (`src/hanoon_prime/ib_executor.py:61`, `:109-114`).
4. **Exits**: `ExitLadder` — Tier1 hard-stop (`exit_ladder.py:139-143`) → Tier2 JULI
   verdict reading **frozen** learned thresholds (`:154-156`) → Tier3 mechanical
   `ExitPolicy.evaluate` (`src/hanoon_prime/brain/exits.py:343-380`).
5. **The only live learning**: `ExitPolicy.adapt_from_realized`
   (`exits.py:253-281`, called at `orchestrator.py:1418`) — retunes giveback
   keep-ratio + stale window only, bounded by `EXIT_ADAPT_*`.

**Invariants while working on organs** (enforced by `CONTRACT.md`, R1, and tests):
- Only Cortex produces verdicts. Neuromorphic/MoE/contrarian outputs stay numeric scores.
- Single-learning-system rule: synaptic modification paths are reviewed together
  (see note at `src/hanoon_prime/brain/neurons/stdp.py:7-9`).
- No live-path merge without backtest proof (WFA + deflated Sharpe)
  — `tests/test_contract.py`, `scripts/check_profit_gate.py`, CI.
- `SafetyProducer` starts disabled (`src/hanoon_prime/safety.py:36,84-85`) — halts stay latched.

---

## 1. Organ audit table (current truth)

| # | Organ | Status | Root cause | Effort to fix |
|---|-------|--------|-----------|---------------|
| A | STDP reward learning | REPAIRED + GATED | wrote `STDPLearner._synapses`, scorer read `LIFNetwork._weights` (two stores) + dead recency check — both fixed; write-back behind `NEURO_LEARN_ENABLED=False` | M (done) |
| B | AttractorMemory | PERSISTED (opt-in) | persisted to `STATE_DIR`, only live adapter opts in | S (done) |
| C | SleepReplayEngine | REPAIRED + GATED | was: wrong neuron ids, no plasticity, Poisson unused — all fixed | DONE |
| D | DynamicThresholdAdapter + `DECISION_THRESHOLD` | WIRED + GATED | adapter output scale was nonsense (0.02-range vs 0.6/0.7 neurons); repurposed to relative-vol multiplier, fed per tick | M (done) |
| E | Nine-rule exit learning | LIVE (wired into trade close) | missing call in trade-close path | S |
| F | Flag-gated organs (6 flags) | OFF (verified + enforced) | rollout switches awaiting validation; R29 contract test now blocks silent flips | per-flag M |
| G | MoE experts + cross-asset inputs | WIRED + GATED | experts had zero synapses; now wired/routed by a regime soft gate (`NEURO_MOE_GATE_ENABLED=False`); cross-asset ids still not built (documented) | L (routing done) |
| H | NetworkStepper | DELETED (provably divergent) | scalar vs vectorized `step_all` not score-identical; orphaned | S (done) |

Legend — S: <1 session. M: 1–3 sessions. L: multi-session design work.

---

## 2. Known corrections to earlier analysis (recorded so we don't re-litigate)

- [x] `adaptive_learning.py` is NOT dead code as a module — it is imported and correctly
  wired to `AdaptiveThresholds.update_from_outcome` (`adaptive_learning.py:88-108` →
  `adaptive_thresholds.py:115`). It is a **dead path because `update_from_outcome` has zero
  call sites** in `src/`. The module itself is healthy; the entry point is orphaned.
- [x] Sleep loser 3× weighting exists (`SLEEP_LOSS_WEIGHT`, `LOSS_BIAS=3.0`) — the *bias*
  mechanism is real; the *execution* is a no-op (see C). Do not assume "replay is driven 3×".
- [x] `DECISION_THRESHOLD` (0.65) and `bridge.threshold` are exported but consumed by nobody;
  the real decision gate is `dynamics.threshold` (`orchestrator.py:1111,1331-1336`).
- [x] Sizes are mislabeled in `moe_config.py:98-104` (claims 66 input / 89 neurons / 256
  synapses) — the built network in `bridge._build_network` is smaller and many of the
  advertised neurons have no incoming synapses.
- [x] **The live SNN scoring path was ALSO dead (not just sleep replay).**
  `InputEncoder._set_input_for_alpha` mapped plain alpha names to `bull_<name>`/`bear_<name>`,
  but `_build_network` creates `bull_<name>_bull`/`bear_<name>_bear` from ALPHA_KEYS
  (`<name>_bull`/`<name>_bear`). Empirically verified: 52 neurons, 0 inputs set, 0 spikes,
  score 0.0 — so NEURO_BLEND 0.3 scored exactly 0 contribution. Only 5 of 11 EPISODIC_KEYS
  indicators have neurons (vpin, orderbook_imbalance, institutional_flow, momentum,
  vwap_deviation); `{"vpin_bull": 0.5}`-style keys always worked.
**Fixed** (encoder plain-key → `bull_<name>_bull`/`bear_<name>_bear`, two-sided),
   **restored blend gated behind `NEURO_BLEND_ENABLED=False`** (immune.py) so the live
   0.7·cortex+0.3·neuro path is byte-identical until a backtest gate clears it.
- [x] **STDP reward loop's recency check was dead code.** `Synapse.last_pre_spike`/`last_post_spike`
  were assigned nowhere — `apply_reward`'s `now - last_pre_spike < window_sec` could never pass, so
  reward never landed on any synapse (findings at `stdp.py:169-170` after A1 debugging). Fixed by
  stamping the receiving synapses in `_record_pre_spike`/`_record_post_spike`. This made C1's
  reward-modulated consolidation real too, and A1's fast-path write-back observable.
- [x] **Attractor `store()` semantics:** a fresh pattern creates the attractor with `trade_count==0`
  (stats increment only on *merged* updates, diff < 0.1) — sleep replay selection needs
  `trade_count >= 2`, so the first store of a ticker is not replay-eligible until a second close merges.
- [x] **`moe_config.NETWORK_ARCHITECTURE` lied** (claimed 66/20/89/256). Audited real build:
  32 inputs (16 bull + 16 bear), 17 hidden (8 generalist + 9 experts — `cross_regime`/`cross_*`
  are NOT in `HIDDEN_NEURONS`), 3 decision, 52 total, 37 wired pairs. Correlated: expert neurons
  existed but had `{}` target dicts (zero synapses) — the dormant state G1 routes.

---

## 3. The mental map (annotated snapshot)

```
STARTUP  start.command -> scripts/bot_supervisor.py -> python -m hanoon_prime.cli
         -> IBStreamingBot (ib_adapter.py:38) -> JuliBrain (juli.py:24)
         -> NeuromorphicBrain (orchestrator.py:97)
         -> ConsolidationEngine (S2 thread, consolidation.py:81-89) — sleep etc.
         -> SafetyProducer DISABLED; TelemetryAPI :8080

DATA     IB mkt data -> DataBudget seats (data/budget.py:48, ROTATE 20/cycle)
         -> build_snapshot (ib_cycle.py:391-420) -> 27-alpha (cerebellum.py:134-166)
         -> _last_alpha cache (orchestrator.py:105)

LIVE DECISION (per cycle)
   _evaluate_fast (:697-736)
     ├ Cortex base.score/direction            [LIVE, verdict authority — R1]
     ├ Nash predict + penalty (:629-653)      [LIVE]
     ├ _compute_neuro_score (:616-621)        [REPAIRED; context-fed; GATED OFF -> 0.0]
     │      feed_context per tick: price history -> adaptive threshold adapter
     │      + regime label/mul -> MoE route (both inert while gated OFF)
     ├ _calibration_nudge                     [FLAG OFF -> 0]
     └ blend :1117 = 0.7·cortex + 0.3·neuro
   -> _score_pipeline (:1099-1137) -> gates (:375-411) -> shadow book (bandit only)

SIZING/EXEC  jury (verdict.py) -> IBExecutor.place_bracket (ib_executor.py:61,:109-114)
   hard limits immune.py:58-64

EXIT     ExitLadder (exit_ladder.py:57)
   Tier1 hard stop (:139-143)                 [LIVE]
   Tier2 JULI (:150-163) <- AdaptiveThresholds [READ frozen :154-156 / TRAIN DEAD]
   Tier3 ExitPolicy.evaluate (exits.py:343-380) [LIVE]
   adapt_from_realized (exits.py:253-281)     [LIVE — only live learning, 2 params]

LEARNING on_trade_close (orchestrator.py:1264-1301)
   ├ adapt_from_realized (:1418)              [LIVE]
   ├ STDP + attractor (:1303-1321)            [LIVE store / C+A DONE, gated OFF]
   ├ RPE phasic/tonic (:1338-1348)            [LIVE telemetry]
   ├ _adapt_threshold (:1323-1336)            [LIVE entry tolerance]
   └ nine-rule update_from_outcome            [0 CALLERS -> E]

CONSOLIDATION (S2) _cycle (consolidation.py:177-202)
   └ _maybe_sleep_replay (:505-518) -> SleepReplayEngine
        [gated idle 30min/close + >=3 patterns] -> REPAIRED: real replay (see C)

OPS / VALIDATE   telemetry :8080 -> api.hanoonweb.xyz; backtest = eyes->hands->metrics;
   WFA + deflated-Sharpe merge gate (CONTRACT.md, CI)
```

---

## 4. Work plan (systemic, in dependency order)

### P0 — Cheap, high-certainty, do first
- [x] **E1. Wire the nine-rule exit learner into trade close.**
  Invoke `update_from_outcome(...)` from the trade-close path next to
  `adapt_from_realized`, passing `won`, `hold_minutes`, `pnl_pct`, `peak_pct`,
  `exit_reason`, `exit_likelihood`. Confirm `adaptive_thresholds.json` persists and
  `ExitLadder._tier2` consumes learned values (not DEFAULTS).
  **DONE 2026-09-18**: `orchestrator._learn_exit_thresholds` called from `on_trade_close`
  (before `exits.deregister`, so hold/peak are live); reason = `","".join(exit_triggers)`
  else `"mechanical"`; likelihood = entry confidence (`_last_conf`) as the bounded advisory
  signal. `ExitPolicy.observe()` + direction-aware `_update_peaks` added so peaks track on
  every ladder evaluation (decision-neutral: tier3 recomputes the same max; `_peak_price`
  was write-only). Peak learner consumes `peak_return_fraction` (direction-corrected,
  clamped ≥0). `adaptive_thresholds.json` persists via existing `_save`/`_load`.
  *Verify by*: `tests/test_exit_threshold_wiring.py` (4 tests) — giveback loss tightens
  trail_keep_ratio, profit-lock tier reinforced from REAL peak, IRONYCLADE-gated
  (backtest source can't learn, `update_count==0`), short peak clamp. 74 targeted tests pass,
  contract suite green, mypy strict clean, ruff clean.
- [x] **C1. SleepReplayEngine replay is real — repaired + gated.**
  **DONE 2026-09-18**: attractor centers map to actual neurons
  (`replay.encode_pattern`, EPISODIC_KEYS order → `bull_<name>_bull`/`bear_<name>_bear`,
  same two-sided convention as the live encoder); `_replay_pattern` now runs the full
  input→hidden→decision wiring with STDP pre/post spike handlers + trace decay
  (`replay.drive_steps`), Poisson background activity (`replay.inject_poisson` on quiet
  input neurons, POISSON_RATE), reward-modulated bonus/malus, and honest telemetry
  (weights_updated / mean_weight_change from synapse-strength snapshots). All wire-level
  synapses registered into the STDP store (`bridge._register_all_synapses_for_plasticity`
  — previously only hidden→hold), which is the prerequisite for A1. Live scoring remains
  untouched: the repaired SNN input path (plain-name encoder bug, section 2) and the sleep
  consolidation it could affect are gated behind `NEURO_BLEND_ENABLED=False` →
  `_compute_neuro_score` returns 0.0 (byte-identical live 0.7·cortex+0.3·neuro).
  Legacy `alpha_<i>` pattern keys still work via `_apply_indexed_input` back-compat.
  *Verify by*: `tests/test_sleep_replay_live.py` (8 tests) — winner replay → `weights_updated>0`,
  `spikes_generated>0`, `mean_weight_change!=0`; decision edge strengthened; stdp covers
  input→hidden→decision; encoder plain keys fire the SNN; blend gate off-by-default (on
  after monkeypatch). 126 targeted tests + full non-backtest suite (1205) pass; mypy
  strict clean; ruff clean (one pre-existing PLR0913 in `consolidation.py:47` at HEAD).
- [x] **B1. Persist AttractorMemory.**
  Serialize `_attractors` (ticker/center/won/wins/losses/pnl) to `STATE_DIR`
  (same pattern as `_save`/`_load` in `adaptive_thresholds.py:148-181`); load at
  `NeuromorphicBridge.__init__`. Remove the "not persisted" admission at `orchestrator.py:1310`.
  **DONE 2026-09-18**: `attractor.py` grows `filepath`/`load()`/`save()` (best-effort, atomic
  tmp+`os.replace`), `store()` auto-saves in both create and merge branches; bridge takes
  `memory_path=None` (in-memory default), static `_memory_path()` = `STATE_DIR/attractor_memory.json`.
  Opt-in at the live boundary only: `ib_adapter.py:47` → `JuliBrain(ib_client, persist_memory=True)`;
  `NeuromorphicBrain`/`JuliBrain` default `persist_memory=False` so the test suite stays hermetic
  (an earlier default-ON design leaked TEST tickers into the runtime file and broke
  `test_closes_populate_replayable_attractors`). Fresh-store semantics confirmed: `trade_count==0`
  until a merge (see section 2).
  *Verify by*: `tests/test_attractor_persistence.py` (5 tests) — roundtrip, merge update, corrupt-file
  recovery, bridge default in-memory, declarative JSON load. Hermetic default confirmed: no runtime
  state file persists during the suite; full non-backtest suite 1205→1214 pass.
- [x] **H1. NetworkStepper: hook in or delete.**
  Swap `bridge.py:76` (`self._network.step_all`) to `NetworkStepper.step_all` and assert
  score-identical outputs; otherwise remove `network_step.py`.
  **DONE 2026-09-18 (delete)**: empirical property run (10 random alpha sets, drive 5 steps)
  showed scalar vs vectorized `step_all` NOT output-identical (scalar produced nonzero
  `decision_long/short` currents/scores; vectorized produced 0.0 in several trials) — the pair
  would silently change live scoring. Deleted `network_step.py`, removed its mypy override from
  `pyproject.toml:72-74`, corrected the `network.py` docstring. Live path untouched (scalar
  `step_all` remains the only driver).

### P1 — Make SNN learning real
- [x] **A1. STDP feedback loop to the scorer.**
  `learn_from_outcome` (`bridge.py:89-93`) must write back into `LIFNetwork._weights`
  (the dict read at `network.py:90`) for the synapses driving `decision_long/short`,
  not just `STDPLearner._synapses`. Ensure all fast-path synapses are registered in STDP
  (`create_synapse` for the input-hidden-decision wiring, currently only hidden→hold at
  `bridge.py:51`). Add an eligibility trace for the distal-reward problem (see RSTDP lit).
  Expect NEURO_BLEND impact to change behavior — gate behind a test/backtest window,
  do not flip live blind.
  **DONE 2026-09-18**: `learn_from_outcome` now syncs STDP strengths back into
  `LIFNetwork._weights` via `_sync_plasticity_to_network()` (setdefault-merge per synapse) —
  the store the scorer actually reads — gated by **`NEURO_LEARN_ENABLED=False`** (immune.py);
  OFF = byte-identical live. The eligibility trace is each `Synapse.trace` (per-spike Δw,
  decayed by `update_traces`, gated by recency in `apply_reward`). Debugging exposed the dead
  recency check (section 2) — fixed in `_record_pre_spike/_record_post_spike`, which also made
  C1 reward consolidation real. Distal credit: hidden/decision spikes record pre timestamps
  (`replay.drive_steps` calls `on_pre_spike` for every spiking neuron), so decision-path
  synapses are eligible.
  *Verify by*: `tests/test_neuromorphic_learning_loop.py` (4 tests) — write-back gated off by
  default; winning close raises `hidden_trend→decision_long` above static default, losing close
  stays below; sync covers every STDP synapse (strength match in `_weights`); losing outcome pulls
  the fast path down toward `STEEP_MIN`. Full non-backtest suite 1214 pass; mypy strict clean;
  ruff clean.
- [ ] **A2. (Optional) Train the SNN head offline.**
  Unsupervised pre-train on tick/alpha history via `scripts/` backtest path, with a
  spike-rate-aligned objective (cf. PSA in literature, section 5). Output a learned
  `weights_config` default for deployment.

### P2 — Wire what exists
- [x] **D1. DynamicThresholdAdapter → neuron thresholds.**
  Per cycle, feed realized vol into `_threshold_adapter` and apply
  `compute_dynamic_threshold`/`adapt_for_market` to neuron firing thresholds
  (`lif.py:35`, currently fixed 0.7 at `bridge.py:45`). Needs a live VIX/realized-vol
  source (none today — `vix=15.0` constant at `threshold_adapter.py:23`).
  Remove or repurpose `DECISION_THRESHOLD` (decide: delete export vs. wire into evidence).
  **DONE 2026-09-18**: real vol source found — `BrainState.get_latest_prices()`
  (50-slot per-ticker price history). `compute_dynamic_threshold` repurposed to return
  `base * multiplier` where multiplier = (realized per-bar vol / `NOMINAL_REL_VOL`) ×
  VIX scale, clamped [0.5x, 5.0x] — the old `base * vol` form could never exceed its own
  base with realistic vol, so the scale would have pinned at 0.5x and the wiring been a
  no-op (zero consumers, safe to repurpose). `orchestrator._compute_neuro_score` feeds the
  bridge per evaluated tick REGARDLESS of the blend gate (so D1 validates independently);
  `bridge.update_market_env` → `adaptive_thresholds_wiring.feed_market_env` (helper module
  keeps bridge.py under the R3 200-line cap), scaling EVERY neuron layer from its build-time
  base (input/hidden 0.7, decision 0.6). Gated by **`NEURO_ADAPTIVE_THRESHOLD_ENABLED=False`**
  (immune.py): OFF only records inert adapter history + re-asserts build-time thresholds —
  byte-identical live. `DECISION_THRESHOLD` export left as-is (still consumer-less; documented
  dead export, not wired).
  *Verify by*: `tests/test_dynamic_threshold_wiring.py` (6 tests) — thresholds unchanged by
  default; high vol + VIX raises input/decision thresholds above build-time; calm market lowers
  them; scale clamped [0.5x, 5.0x]; OFF-after-ON restores 0.6/0.7; short history falls back to
  base. Full non-backtest suite 1214→1220 pass; mypy strict clean (195 files); ruff clean.
- [x] **G1. MoE soft-gate using live regime.**
  Route `regime_mul` (available at `orchestrator.py:1119`) through a learned soft gate to
  residual expert outputs (momentum/meanrev/liquidity, `moe_config.py:15-55`), following
  the literature principle: regime modulates expert *routing*, not the base forecast.
  Wire expert neurons into `weights_config` (`:23-79` currently has zero expert synapses).
  Keep verdict authority with Cortex (R1).
  **DONE 2026-09-18**: new `moe_gate.py`. `soft_route(regime, multiplier)` returns normalized
  routing weights ∈ (0,1) summing to 1 — softmax over tuned regime affinities
  (trending → momentum-heavy, mean_reverting → mean-reversion-heavy, ranging → tilts
  liquidity+meanrev, unknown → balanced); regime_mul scales the momentum affinity before
  softmax (strong-trend read tilts routing further, never hard-switches). `route_experts`
  lazily wires the dormant expert neurons (input→expert at 0.2 per family source map,
  expert→decision at `MOE_EXPERT_DECISION_WEIGHTS × route[family]`) and applies the gate
  every tick — gated by **`NEURO_MOE_GATE_ENABLED=False`** (immune.py), so experts stay in
  the exact audited dormant state (empty target dicts) live. `bridge.feed_context(...)`
  folds thresholds + routing into one per-tick feed; `orchestrator._compute_neuro_score`
  passes regime label/multiplier from state. Offline expert-head training is still A2
  (the plumbing accepts learned weights without rewiring). Cortex verdict authority
  untouched (score-level only).
  *Verify by*: `tests/test_moe_soft_gate.py` (8 tests) — routing normalized/ordered/balanced,
  multiplier tilts momentum, dormant-by-default, gated-on wiring matches
  `MOE_EXPERT_DECISION_WEIGHTS × route`, inputs reach experts, `process_alpha` shape intact.
  Full non-backtest suite 1220→1229 pass; mypy strict clean (196 files); ruff clean.

### P3 — Flag rollouts (each needs a validation gate before flipping)
Each is a literal switch in `immune.py:92-124`; "flip" = local test → paper/shadow →
backtest proof → flip → monitor telemetry for one session.
**Status 2026-09-18**: all six verified default-OFF and now *enforced* by
`test_R29_gated_organs_default_off` (contract/CI + pre-commit `r29-gates-default-off`).
No flip is permitted until the baseline money gates clear (see P4 backtests).

> **PROMOTION POLICY (2026-09-18)** — Baseline money gates FAIL with all organs OFF:
> R2 backtest (`python -m hanoon_prime.backtest --data-dir data/fixtures --tickers ALL
> --output metrics` + `check_profit_gate`) = 6/23 tickers profitable, 432 trades,
> gate FAILED; WFA (`scripts/wfa_report.py ...`) = deflated OOS Sharpe **−0.307**
> (≤ 0 → FAIL), PBO 0.48 after 126 trials. Therefore: **NO neuro/P3 gate may be
> flipped** — doing so before the baseline clears would invalidate any A/B. The
> per-feature A/B recipe (flip one gate locally → rerun both gates on fixtures →
> only promote with **deflated Sharpe > 0 AND profitability gate pass**) stays armed;
> it activates the moment the baseline cross-clears. C1/A1/D1/G1/B1 all stay OFF.

### P4 — Enforcement, facade, verification (2026-09-18)
- [x] **F-flags verification.** All 6 P3 flags + 4 neuro organ gates confirmed literal
  `False` defaults (immune.py) and byte-identical live with gates off (existing suite).
- [x] **Enforcement.** New `test_R29_gated_organs_default_off` pins every rollout gate to
  default-OFF (a silent flip fails contract/CI); pre-commit local hook
  `r29-gates-default-off` blocks it at commit time. CI's existing contract job runs
  `tests/test_contract.py` so the new rule is live there too.
- [x] **Facade refactor (behavior-identical).** `orchestrator.snapshot()` no longer owns its
  telemetry rendering — `brain/telemetry_facade.build_brain_snapshot` (stable `SNAPSHOT_KEYS`
  contract, same None-guards, same attribute reads). `tests/test_telemetry_facade.py` (4)
  pins delegation parity, key contract, no-neuromorphic and no-sleep-engine branches. Object
  goes 160+ lines slimmer; live path untouched.
- [x] **Backtests (CSV fixtures) + promotion verdicts.** Ran the R2 + WFA money gates on the
  committed `data/fixtures/*.csv` with all organs OFF. Results (above) = baseline FAIL →
  verdict for every gated feature is **HOLD/HOLD-OFF** pending a baseline cross-clear.
  `metrics/` output is now gitignored (CI regenerates).

### P5 — Brain-path organ A/B replay (2026-09-18) — HOLD, with evidence
Instrument: `scripts/ab_brain_replay.py` (+ `tests/test_ab_replay_smoke.py`).
Replays each fixture bar through the REAL live loop (`decide_entry` →
`check_exit` → `on_trade_close`) — the path `hanoon_prime.backtest` can never
reach (it runs `hands.py`, bypassing NeuromorphicBrain). Gates flipped at
runtime (module alias / lazy readers), fresh brain per variant,
`persist_memory=False` (hermetic — no runtime/ writes).

**Funnel autopsy.** At live defaults the brain books ~13–20 trades across
22 tickers × ~1,900 bars (OFF ~13–14, BLEND 20 — the funnel is seed-sensitive;
TSLA alone 0/1,703, and still 0 in
both-direction mode). Binding gates measured: `direction_mode=long_only` vetoes
the SHORT-dominated score deck; `Dynamics` self-calibrates the entry threshold
to the ~90th percentile of \|stabilized\|; the risk EV gate + Kelly
sub-rounding guard refuse below-rounding conviction. **The brain trades this
sparsely BY DESIGN.** A P/L organ A/B on these fixtures is therefore empty
(15–20 trades/variant; any deflated value is noise). Forcing trades via
research thresholds/`--both` was trialled and rejected as an anti-overfit risk
— that is precisely what the WFA gate exists to catch.

**Signal-channel A/B** (pre-sizing score/side, 37,400 bars/variant):

| variant | mean\|Δscore\| | side_flip | corr vs OFF | mean\|neuro\| |
|---|---|---|---|---|
| BLEND | 0.3014 | 0.386 | 0.999 | 0.998 |
| BLEND+LEARN | 0.3014 | 0.386 | 0.999 | 0.998 |
| BLEND+THRESH | 0.3014 | 0.386 | 0.999 | 0.998 |
| BLEND+MOE | 0.3014 | 0.386 | 0.999 | 0.998 |
| BLEND+SLEEP | 0.3014 | 0.386 | 0.999 | 0.998 |

**Saturation finding.** `process_alpha`'s neuro score is near-binary: on TSLA,
151/155 reads = 1.0 exactly, 3/155 = 0.0. The repaired SNN input path emits an
always-full-conviction read. At the 0.7·base + 0.3·neuro blend this is a fixed
±0.3 tilt that flips the OFF side on ~39% of bars whenever \|base\| < ~0.43 —
a real behavior change carrying no graded information. The D1/G1 inner
modulations (neuron firing thresholds, expert routing weights) are invisible
behind the saturated read (identical deltas across those variants). +LEARN and
+SLEEP show zero delta by construction: both organs are CLOSE-driven (STDP
write-back; attractor→replay consolidation) and signal mode has no closes.

**Promotion verdicts (the evidence):**
- **NEURO_BLEND_ENABLED — HOLD, do-not-flip:** SNN read saturated/near-binary;
  adds a fixed tilt, not conviction. Needs a graded read (process_alpha) AND a
  funnel that trades before it can be re-tested.
- **NEURO_LEARN/A1, NEURO_ADAPTIVE_THRESHOLD/D1, NEURO_MOE_GATE/G1, sleep/C1 —
  HOLD:** no discriminative funnel exists on these fixtures; their effect is
  unmeasurable by construction. Re-test path = (a) core alpha must first cross
  the R2/WFA gates (it currently fails: 6/23 profitable, deflated −0.307) so
  base scores carry conviction mass; (b) then validate organ P/L in the LIVE
  shadow/journal telemetry on real sessions — fabricated fixtures cannot.
- R29 contract + pre-commit keep all 10 gates OFF; nothing changes live. The
  harness is retained as the instrument for that day.

- [ ] **F1. `CALIBRATION_NUDGE_ENABLED`** — score nudge toward realized win-rate bands.
  Requires realized-data confidence; start with `CALIB_BOUND=0.10` capped.
- [ ] **F2. `HYSTERESIS_EXIT_ENABLED`** — soft (Tier2) exits must persist 3 bars.
  Raises exit latency; measure giveback/win-rate impact before keeping.
- [ ] **F3. `PROBE_RECOVERY_ENABLED`** — 1 quality-gated re-entry after 3-loss streak.
  Proceed only after halt behavior is proven edge-positive (docs at `immune.py:86-96`).
- [ ] **F4. `CONTRARIAN_MODE_ENABLED`** — contrarian override on extreme vwap z.
  Conflict with R1 (verdicts): must stay advisory/score-level, never flip verdict.
- [ ] **F5. `DELIBERATION_TRACE_ENABLED`** — diagnostic CoT trace.
  Harmless (candidate score NOT blended, `orchestrator.py:1178`); used for explainability.
- [ ] **F6. `HALIM_EVIDENCE_LEARNING`** — evidence-grounded HALIM CoT → recs.
  Depends on validated evidence extraction; keep off until F5 tooling exists.

---

## 5. Literature digest (what the research actually says)

- **Reward-modulated STDP / SNN trading** — RSTDP actor-critic works on control benchmarks;
  STDP-trained SNNs beat supervised baselines on HFT price-spike prediction *when* trained
  unsupervised + tuned with a spike-rate-aligned objective (Penalized Spike Accuracy).
  Keyword lesson: pre-training + proper decode + threshold tuning, not hand-wired weights.
  Refs: arXiv 2512.05868; arXiv 2402.01533 (iSpikformer, SNN≈ANN accuracy at ~70% less energy).
- **Sleep replay / SRC** — unsupervised Hebbian sleep replay demonstrably reduces
  catastrophic forgetting in ANNs (Nature Communications 2022) and in *spiking* networks via
  joint-synaptic-weight representation (PLOS Comp Biology). The mechanism requires
  (a) spontaneous/Poisson activity and (b) local plasticity during replay — both currently
  absent in our `sleep.py`.
- **MoE for markets** — validated (TradingMoE vs 22 baselines; volatility-gated MoE).
  Strong 2026 finding: **soft, regime-gated residual routing beats hard routing**; value comes
  from *where* regime info enters, not from extra capacity. Our regime signal exists and is unused.
- **Adaptive exit thresholds** — per-reason threshold updates from realized outcomes is
  standard systematic practice; our nine-rule engine is well built but orphaned (item E1).
- **Continual RL replay** — naive experience replay can still forget; replay needs
  policy-distillation/normalization (RECALL). Relevant when we scale attractors into real
  learning — but C1/B1 first.

---

## 6. Session log

- **2026-09-18** — Full audit: read every file in `brain/neurons/` + `adaptive_learning.py`,
  `adaptive_thresholds.py`, `threshold_limits.py`, `sleep_scheduler.py`, traced all call sites.
  Produced corrections (section 2) and this plan. No code changes yet.
- **2026-09-18** — Complexity hotspot review vs R3 (200-line cap) incl. `FROZEN_FILE_SKIP`
  allowlist and dual weight-store/manifest issues; delivered spaghetti assessment.
- **2026-09-18** — E1 DONE: nine-rule exit learner no longer orphaned. Wired
  `update_from_outcome` into `on_trade_close` via `_learn_exit_thresholds` (hold/peak captured
  pre-deregister; reason from exit_triggers; likelihood = entry confidence). Added
  `ExitPolicy.observe()` + direction-aware peaks (decision-neutral — `_peak_price` was
  write-only; ladder now observes every evaluation, peak learning uses `peak_return_fraction`).
  `tests/test_exit_threshold_wiring.py` (4 hermetic tests, monkeypatched tmp filepath);
  74 targeted tests + contract suite green; mypy strict clean; ruff clean.
- **2026-09-18** — C1 DONE + SNN input repair: `replay.py` module holds network/STDP driving
  (encode_pattern, drive_steps w/ pre/post spike + trace decay, inject_poisson, synapse
  snapshot/delta); `sleep.py` slims to policy/bookkeeping (173 lines, under R3 cap). Full
  input→hidden→decision synapse registration into STDPLearner (`bridge._register_all_synapses_for_plasticity`)
  = prerequisite for A1. InputEncoder plain-key bug fixed (was `bull_<name>`, real id
  `bull_<name>_bull`); restored blend gated via `NEURO_BLEND_ENABLED=False` so live path is
  byte-identical. `consolidation._sleep_patterns` + runner all encode real ids. New
  `tests/test_sleep_replay_live.py` (8 tests); 126 targeted + full suite 1205 pass; mypy
  strict clean; ruff clean (pre-existing PLR0913 at `consolidation.py:47` left as-is).
- **2026-09-18** — Wrote two Excalidraw diagrams under `diagrams/` (validated JSON, 33 elements each):
  `current-spaghetti.excalidraw` (as-built: live green path, inert/amber cluster, dead/red cluster,
  GOD objects) and `target-clean.excalidraw` (target: thin facade, decoupled layers, one manifest,
  one weight store, wired learning loop). Research basis: god-object refactoring (Nakov), NexusFi
  7-layer trading architecture (Risk boundary authorizes OrderIntent, fail-closed), sleep replay
  + soft regime-gated MoE literature. No code changes yet.
- **2026-09-18** — B1 + H1 DONE. B1: AttractorMemory persisted (opt-in at live boundary only —
  `ib_adapter` passes `persist_memory=True`; brain/bridge default in-memory so tests stay hermetic;
  an earlier default-ON attempt leaked TEST tickers and broke `test_closes_populate_replayable_attractors`).
  `tests/test_attractor_persistence.py` (5). H1: NetworkStepper DELETED — empirical runs proved it
  not score-identical to scalar `step_all` (vectorized path produced 0.0 vs nonzero scalar in
  several trials); removed mypy override + docstring correction. Also merged sleep-engine
  mechanics into `replay.py` (sleep.py back under R3 cap) and trimmed `juli.py`/`orchestrator.__init__`
  for R3 (200-line, 40-line caps). Full non-backtest suite 1205 → 1214 pass; mypy strict clean.
- **2026-09-18** — A1 DONE. STDP → scorer feedback loop: `learn_from_outcome` syncs STDP strengths
  into `LIFNetwork._weights` behind **`NEURO_LEARN_ENABLED=False`** (immune.py; OFFs keep the live
  path byte-identical). Debugging found the recency check dead (`Synapse.last_pre_spike/post_spike`
  never assigned — reward never landed; see section 2), fixed in the spike recorders, which also
  made C1's reward path real. `tests/test_neuromorphic_learning_loop.py` (4) prove write-back,
  gating, full-synapse coverage, and loss pulling the path down. Full non-backtest suite 1214 pass;
  mypy strict clean; ruff clean.
- **2026-09-18** — D1 DONE. Adaptive firing thresholds wired + gated
  (`NEURO_ADAPTIVE_THRESHOLD_ENABLED=False`). Realized vol now from `BrainState.get_latest_prices`;
  `compute_dynamic_threshold` repurposed as a relative-vol × VIX multiplier in the caller's unit
  (old form was un-closable for realistic vol). `orchestrator._compute_neuro_score` feeds every
  evaluated tick regardless of blend gate; new `adaptive_thresholds_wiring.py` keeps bridge.py
  under R3's 200-line cap. `tests/test_dynamic_threshold_wiring.py` (6). Full suite 1220 pass;
  mypy strict clean (195 files); ruff clean.
- **2026-09-18** — G1 DONE (+ manifest audit). `moe_gate.py` adds soft regime routing
  (`soft_route`: normalized weights ∈ (0,1), sum 1; regime_mul tilts momentum pre-softmax) and
  lazily wires the dormant experts (input→expert 0.2, expert→decision `MOE_EXPERT_DECISION_WEIGHTS`
  × route) — gated by `NEURO_MOE_GATE_ENABLED=False`. `bridge.feed_context` = threshold feed +
  routing in one per-tick call; orchestrator passes regime label/multiplier. `moe_config` manifest
  corrected to the real 32/17/3/52/37 (was fabricated 66/20/89/256). `tests/test_moe_soft_gate.py`
  (8). Full suite 1229 pass; mypy strict clean (196 files); ruff clean.
- **2026-09-18** — Enforcement + facade + backtests. New R29 contract test pins all 10 rollout
  gates to default-OFF (silent flips now fail CI and pre-commit via `r29-gates-default-off`).
  Facade: `telemetry_facade.build_brain_snapshot` extracted (orchestrator.snapshot delegates;
  `SNAPSHOT_KEYS` contract, `tests/test_telemetry_facade.py` 4 tests). Ran the real money gates
  with everything OFF on the committed fixtures: **R2 FAIL** (6/23 tickers, 432 trades) and
  **WFA FAIL** (deflated Sharpe −0.307, PBO 0.48) → **promotion policy = no gate flips until the
baseline cross-clears**; A/B recipe documented in P4. `metrics/` now gitignored. Full suite
   1229+ pass; mypy strict clean (197 files); ruff clean.
- **2026-09-18** — Organ A/B replay (P5). Built `scripts/ab_brain_replay.py` — replays fixtures
  through the REAL live loop (`decide_entry`→`check_exit`→`on_trade_close`), bypassing the
  brain-free hands.py backtest path; runtime gate flips (module alias + lazy readers) with
  try/finally restore now pinned by a smoke test; hermetic (no runtime/ writes). Funnel autopsy:
  live-sparse by design (~13–20 trades/22 tickers; TSLA 0/1703) → P/L A/B statistically empty.
  Signal-channel A/B (37,400 bars/variant): **neuro score is saturated/near-binary** (151/155
  reads = 1.0 on TSLA) → blend adds a fixed ±0.3 ± pure tilt, flipping OFF's side on ~39% of
  bars; D1/G1 modulations invisible behind the saturated read; +LEARN/+SLEEP show zero delta
  (close-driven by construction). Verdicts: NEURO_BLEND = HOLD/do-not-flip (needs graded SNN
  read + trading funnel); A1/C1/D1/G1 = HOLD (unmeasurable on fixtures; validate live once R2/WFA
  cross-clear). Full non-backtest suite 1237 pass; mypy strict clean (197 files); ruff clean.