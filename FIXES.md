# Fixing Journal — hanoon_prime

Append-only log of every production bug found and fixed, with the test
that guards against regression. **Enforced by test:**
`tests/test_fixes_journal.py` verifies that every journal entry has a
guard, every guarded test exists, every fix is classified under a bug
class, and every `Regression:` marker in the test suite points at a
real entry.

## Rules

1. **No fix lands without a guard.** A guard is either a unit test
   whose docstring carries `Regression: FIX-YYYY-MM-DD-NN`, or a
   named smoke check (`smoke: scripts/smoke_live.py::<check>`).
2. **Analyze before fixing.** Before fixing any new bug, read the
   Bug Classes section below; classify the new bug into an existing
   class (hardening the class defense) or open a new class. The
   enforcer test fails if any FIX ID is missing from the class map.
3. **Newest entries last.** IDs are `FIX-YYYY-MM-DD-NN`, assigned in
   order within a day.

## Bug Classes (patterns from past fixes — read before fixing anything)

### Class A — path arithmetic / environment assumptions
Wrong `parents[N]` depth, CWD-relative paths, ambient-import
assumptions. Bugs #1, (earlier) per_ticker_calib CWD-relative path.
**Defense:** one canonical constant + depth regression tests; hermetic
test fixtures.

### Class B — single-writer violations
Two modules writing the same store (double-counted outcomes, episodic
double-entries). Bug #2.
**Defense:** ownership documented at the write site; smoke delta
checks assert exact expected deltas (e.g. +5, not +10).

### Class C — typed-container truthiness (`arr or fallback`)
`or []` / `or {}` on numpy arrays raises ValueError or silently
starves data. Bug #5 — killed every live entry evaluation; the `_SRC`
mapping variant silently starved volume/depth indicators.
**Defense:** snapshot-shaped regression test with numpy arrays;
static grep watchdog in the enforcer test.

### Class D — silent exception swallowing hides systemic failure
Per-ticker `except: log.warning` turned a total outage (zero
decisions) into quiet warnings. Bug #5's invisibility.
**Defense:** eval-failure counter + PipelineMonitor probe that alerts
when failures accumulate while the market is open; failures are
counted, never just logged away.

### Class E — unwired capability assumed to be working
StrategyGenome had zero callers; watchdog/monitor modules were dormant.
Bugs #4, monitor gap.
**Defense:** smoke asserts every organ is reachable and advancing;
monitor probes verify the monitors themselves run.

### Class F — safety-net integrity (kill switch / halt / bypass)
A safety mechanism that is documented but unwired, silently disabled,
non-persistent, or bypassed without an explicit, loud, fail-closed
override. A kill switch that only cancels orders, a halt that vanishes
on reboot, a documented auto-resume that was never implemented, or a
training bypass that could start on a live account are all Class F.
**Defense:** named override constant + fail-closed startup guard;
halt/latch persisted atomically and restored on boot; kill flattens
positions, not just orders; auto-resume bounded by PAUSE_DURATION_MIN
(kill latches never auto-resume); every safety fix carries a
regression test.

### Class G — unasserted arithmetic (silent scaling/sign distortion)
A formula that runs without error but produces a systematically wrong
value: a dead gated term that still damps the live path (06), a
calibrated threshold whose meaning drifted when the score scale changed
(07), a reward sign inferred from a magnitude that correlates the wrong
way (08). Nothing raises; the numbers are just quietly wrong — and
every downstream gate inherits the distortion.
**Defense:** numeric regression tests pinning a formula's output at
representative inputs (not just "runs without error"); explicit
sign/polarity threading instead of inferring sign from magnitudes;
change-detector tests on calibrated constants so any retune is a
deliberate, documented decision.

### Class H — validation-scope overclaim
A harness or report that measures less than it implies: the arithmetic
is right, but the claim around the numbers is wrong — a verdict
presented as validating the production system while a live component
(the MetaDNN gatekeeper) is never exercised, or purge/embargo theater
wired into a run that fits nothing. Nothing raises; the reader just
concludes more than the evidence supports.
**Defense:** scope disclosure travels with the numbers (serialized into
every report); regression pins that the run never calls the
purge/embargo helpers and never loads the trained DNN artifact.

---

## Entries (append-only; newest last)

### FIX-2026-09-07-01 — Brain learning state stranded in `src/runtime/`
- **Symptom:** smoke persistence checks failed; `runtime/` empty after
  live cycles.
- **Root cause:** `brain/config.py`, `brain/slow_cortex.py`, and
  `brain/consolidation.py` computed the runtime dir with
  `parents[2]` from inside `brain/` — resolving to `src/`, not repo
  root. All learned state was written inside the source tree and lost
  on any clean deploy.
- **Fix:** `parents[3]` in all three; migrated existing state to
  `runtime/`.
- **Class:** A
- **Guard:** `test: tests/test_strategy_organs.py::test_state_dir_resolves_to_repo_root_runtime`

### FIX-2026-09-07-02 — Every real trade double-counted in learning
- **Symptom:** 5 smoke closes → `total_trades +10`; episodic grew 2x.
- **Root cause:** `memory.record_outcome`, `update_pred_error`, and
  `episodic.add` each fired twice per close (orchestrator +
  Reflector both writing the same stores).
- **Fix:** Reflector is the single writer for memory writes;
  orchestrator owns episodic (one entry per close).
- **Class:** B
- **Guard:** `smoke: scripts/smoke_live.py::realized-EV ate 5 trades`

### FIX-2026-09-07-03 — Regime fallback could starve itself
- **Symptom:** fallback never published a label after a dataless
  attempt.
- **Root cause:** the 30s throttle was stamped *before* the
  prices-available check, so an attempt without data blocked the next
  valid one.
- **Fix:** stamp only on successful publish.
- **Class:** E
- **Guard:** `smoke: scripts/smoke_live.py::regime fallback publishes`

### FIX-2026-09-07-04 — StrategyGenome had zero callers
- **Symptom:** the live read-model was never instantiated; telemetry
  showed no genome.
- **Fix:** wired into the orchestrator (`self.genome`) and exposed in
  `brain.snapshot()` / telemetry.
- **Class:** E
- **Guard:** `test: tests/test_strategy_organs.py::test_genome_reflects_live_state`

### FIX-2026-09-07-05 — numpy truthiness killed every live entry evaluation (critical)
- **Symptom:** replay smoke: `thinks=+0`, `decisions=0` on every
  ticker; log full of `Entry eval failed ... truth value of an array
  is ambiguous`. At the open, **zero decisions would ever fire**.
  Invisible to unit tests, which passed list-shaped snapshots.
- **Root cause:** three sites using `arr or fallback` on numpy arrays:
  (a) `juli_feed.entry_bars` (`snap.get("close_arr") or prices`),
  (b) `juli_feed._SRC` mapping — `vol_arr`/`bid_sizes`/`ask_sizes`
  did not match the snapshot's actual keys, silently starving
  volume/depth indicators of all data,
  (c) `orchestrator._vol_pct` (`list(bars.get("close") or [])`).
- **Fix:** explicit `is None`/`len()` checks; `_SRC` corrected to
  `volume_arr`/`bid_sizes_arr`/`ask_sizes_arr`.
- **Class:** C, D
- **Guard:**
  `test: tests/test_bugfixes.py::test_snapshot_shaped_feed_no_array_truthiness`
  (builds a snapshot-shaped dict with numpy arrays) + smoke replay
  phase (real bars through the real tick path, asserts thinks +9,
  bandit selects > 0).

### FIX-2026-09-07-06 — Minor
- **Class:** A (documentation drift)
- `juli.tick` docstring claimed `(entries, exits)`; code returns
  `(exits, entries)`. Docstring fixed to match callers.
- **Guard:** docs-only (no behavioral change)

### FIX-2026-09-07-07 — Portfolio risk manager was a stub with dead inputs
- **Symptom:** `_sync_portfolio_risk` called
  `update(net_liq, {})` — the positions dict was hardcoded empty, so
  exposure, concentration, and position-count logic could never fire;
  the entry gate took no arguments (no per-trade check); rebuild's
  portfolio profit protection (peak unrealized-P&L giveback) was
  entirely absent. Class E (unwired capability assumed working).
- **Fix:** full port of rebuild `risk/portfolio.py`: IB-fed equity +
  `read_portfolio(ib)` holdings, parameterized gate (stress size cap,
  exposure cap, concentration cap, position count, stress block),
  size adjustment (scalar x concentration dampening), and
  `check_portfolio_giveback()` — exit weakest winners when total
  unrealized P&L fades 25% from peak (min $20 peak, batch of 3,
  60s cooldown, losers never exited). Wired into `_sync_portfolio_risk`
  (giveback exits via `_closing`/`_exit_reasons` so reflection sees the
  reason) and `_exec_decision` (per-entry gate + size adjustment).
  Telemetry: `/risk` endpoint.
- **Class:** E, B (the empty-positions feed was a silent single-writer
  style data loss — IB positions never reached the risk layer)
- **Guard:**
  `test: tests/test_coverage_monitors.py::TestPortfolioRisk`
  (12 tests: unsynced block, safe pass, max positions, drawdown scalar
  + stress, stress size cap, exposure cap, concentration cap, size
  adjustment, equity never fabricated, giveback weakest-first,
  giveback min-peak, snapshot shape) + smoke `giveback_unit`/
  `giveback_no_losers` checks exercise the decision on live IB data.

## 2026-09-07 — earlier commits (same session, pre-smoke)

### FIX-2026-09-07-00 — test-environment hardening
- **Class:** A
- Hermetic fixtures (tmp_path redirect of leaf-module state paths),
  deterministic stub test, ib_insync mypy override.
- **Guard:** `test: tests/conftest.py` (autouse hermetic fixture — the
  whole suite fails without it)

- **Hermetic tests:** leaf learning modules bound state paths by
  value; shared `runtime/` files caused cross-test pollution.
  Guard: `tests/conftest.py` autouse fixture (tmp_path redirect).
- **Deterministic stub test:** `test_place_oca_uses_stub_when_ib_missing`
  depended on ambient `ib_insync` presence; now monkeypatches.
- **mypy:** `ib_insync` optional-import override in pyproject.

### FIX-2026-09-08-01 — Blocking history seeding delayed position protection
- **Symptom:** 12 orphan positions adopted on restart; only the LAST one
  (DVLT) got OCA protection. Others stayed unprotected for minutes.
- **Root cause:** `_adopt_orphan_positions` called `streamer.seed_history()`
  synchronously per ticker (≈2s each, 24s total). `protect_position` ran
  after all seeds; the earliest tickers still lacked live price/ATR and
  were silently skipped. Adoption also blocked the main loop (bursty
  startup + late protection).
- **Fix:** adoption now subscribes only; history seeding stays in
  `_sync_subs` (one ticker/cycle, off the hot path). Protection sees
  every position within ~1s → all 12 ADOPTed immediately.
- **Class:** D (silent skip behind blocking hot path)
- **Guard:** `test: tests/test_flow_throttle.py::test_closing_positions_skipped`

### FIX-2026-09-08-02 — Entry evaluation scored all candidates in one burst
- **Symptom:** THINK lines for every scanner candidate fired at once per
  cycle, then silence — bursting, not a stable flow.
- **Root cause:** `juli._evaluate_entries` looped over the whole tracked
  universe every cycle.
- **Fix:** rotating `EVAL_WINDOW` (4 tickers/cycle) with a persistent
  cursor, so scoring is continuous and the loop never stalls on a batch.
- **Class:** D (burst was the symptom; no systemic failure was hidden)
- **Guard:** `test: tests/test_flow_throttle.py::test_eval_window_rotates_across_cycles`

### FIX-2026-09-08-03 — `_watched` lazily initialized, crashed reflect
- **Symptom:** `AttributeError: 'IBStreamingBot' object has no attribute
  '_watched'` in `_reflect_closed` during flatten fill confirmation.
- **Root cause:** `_watched` was created only in `_attach_position_watchers`,
  but adopted (non-watched) positions reach `_reflect_closed`.
- **Fix:** initialize `self._watched = set()` in the bot's `__init__`.
- **Class:** E (unwired assumption — watched set assumed to exist)
- **Guard:** `test: tests/test_flow_throttle.py::test_market_orders_and_no_cancel`

### FIX-2026-09-08-04 — Positions mid-flatten were re-adopted/re-protected
- **Symptom:** after a flatten request, the next cycle re-ADOPted and
  re-protected positions that were being closed.
- **Root cause:** `_adopt_orphan_positions` had no notion of a `closing`
  set, so a still-open (fill pending) position was adopted again.
- **Fix:** pass the bot's `_closing` set into sync/adoption and skip
  those tickers; manual flatten also marks all positions as closing.
- **Class:** B (single-writer race — two subsystems acting on one set)
- **Guard:** `test: tests/test_flow_throttle.py::test_closing_positions_skipped`

### FIX-2026-09-08-05 — Flatten retracted its own orders (limit + cancelAll)
- **Symptom:** flatten appeared to "work" but most positions stayed open;
  only a couple filled.
- **Root cause:** `close_all_positions` placed limit orders then called
  `cancelAllOrders` (retracting the very orders it just placed), and
  low-liquidity limit orders didn't fill post-market.
- **Fix:** flatten uses MKT orders (`_ib.Order(orderType="MKT", tif="DAY",
  outsideRth=True)`) and never cancels; EOD/horizon-aware `only=` path
  unchanged.
- **Class:** B (cancelAllOrders clobbered another subsystem's in-flight orders)
- **Guard:** `test: tests/test_flow_throttle.py::test_market_orders_and_no_cancel`

### FIX-2026-09-08-06 — Sub-dollar bar: raise the bar, no hard block
- **Symptom:** MOST_ACTIVE scanner fed sub-$1 micro-caps (DVLT $0.20,
  WHLR $0.39) into scalp sizing; their 2×ATR brackets were meaningless
  (stop 1 cent from entry).
- **Root cause:** no confidence gate differentiated sub-$1 candidates.
- **Fix:** `PENNY_PRICE`/`PENNY_SCORE_BAR` — below $1.00, |score| must
  clear 0.85 (vs 0.65 base). Not a price block; a genuinely strong
  setup can still qualify. Logged as `sub_dollar_bar`.
- **Class:** E (unwired risk preference)
- **Guard:** `test: tests/test_flow_throttle.py::test_bar_rejects_low_score_penny`

### FIX-2026-09-08-07 — Long-only default; shorts opt-in via telemetry
- **Symptom:** bot opened shorts by default while the operator wanted
  long-only now and shorts later.
- **Root cause:** `direction_mode` defaulted to `"both"`.
- **Fix:** default `direction_mode = "long_only"`; the telemetry API
  still toggles `both`/`short_only` on demand.
- **Class:** E (policy default)
- **Guard:** `test: tests/test_flow_throttle.py::test_direction_defaults_long_only`

## 2026-09-23 — Phase 1: safety-net hardening (training bypass made explicit)

### FIX-2026-09-23-01 — $500 kill bypass is now a named, loud, fail-closed training override
- **Symptom:** the kill switch was disabled for paper training with no
  named flag, no startup warning, and nothing in the dashboard showing it.
- **Root cause:** the disablement was implicit (SafetyProducer constructed
  with `enabled=False`); the $500 kill had no explicit training override,
  so "disabled for training" was invisible to code, logs, and telemetry.
- **Fix:** `TRAINING_KILL_BYPASS = True` in `immune.py` (user-directed,
  2026-09-23 — NOT re-enabled against the user's wishes). `SafetyProducer`
  logs a CRITICAL banner at startup, exposes `kill_bypass`, and skips
  ONLY the $500 kill while bypassed (every other halt still applies when
  enabled). The bypass state is published into `policy_state` and
  `GET /safety-net` (`training_kill_bypass`, `kill_limit`).
- **Class:** F
- **Guard:** `test: tests/test_safety_training.py::test_kill_skipped_while_bypassed_other_halts_apply`

### FIX-2026-09-23-02 — fail-closed guard: a bypassed kill switch cannot start non-paper
- **Symptom:** nothing prevented the bot from connecting to the live IB
  port (4001) or a live account while the $500 kill was bypassed.
- **Root cause:** no startup check tied the bypass to the paper configuration.
- **Fix:** `assert_paper_only_for_training(account, port)` in `immune.py`
  raises RuntimeError unless `account="PAPER"` and port 4002 while the
  bypass is on; wired into `IBStreamingBot.__init__` (account) and
  `connect()` (port), so `run_live()` refuses to start while bypassed.
- **Class:** F
- **Guard:** `test: tests/test_safety_training.py::test_fail_closed_refuses_live_port`

### FIX-2026-09-23-03 — safety halt/latch now persist across restarts
- **Symptom:** a restart silently cleared safety halts and the kill latch
  — the bot could come back trading with no memory of why it was stopped.
- **Root cause:** SafetyProducer kept halt/latch only in memory; nothing
  was written to disk.
- **Fix:** halt/latch/reason/halt-timestamp are persisted atomically
  (tmp-file + replace) to `runtime/safety_state.json` on every `_halt`,
  `_kill`, `resume`, and `release_kill`; restored in `__init__` with a
  warning log. A corrupt state file starts clean (logged) instead of
  crashing startup.
- **Class:** F
- **Guard:** `test: tests/test_safety_training.py::test_halt_persists_across_restart`

### FIX-2026-09-23-04 — kill switch flattens positions, not just cancels orders
- **Symptom:** the kill hook only called `executor.cancel_all()` — open
  positions stayed exposed while new entries were blocked.
- **Root cause:** `_wire_kill_hook`'s `_kill_cancel` never flattened.
- **Fix:** the kill hook is now `IBStreamingBot._on_kill`, which cancels
  working orders AND calls `executor.close_all_positions(self.streamer)`
  (market orders, so the flatten fills — same rationale as the 2026-09-08
  flatten fix). The `SafetyProducer.on_kill` docstring now documents
  cancel + flatten.
- **Class:** F
- **Guard:** `test: tests/test_safety_training.py::test_kill_hook_cancels_and_flattens`

### FIX-2026-09-23-05 — documented 60-minute auto-resume actually implemented
- **Symptom:** the safety docstring promised "auto-resume after 60 min
  pause" for the 3-consecutive-losses halt, but halts persisted until a
  manual resume.
- **Root cause:** `_halt` never stamped a time and `authorized()` never
  checked expiry.
- **Fix:** `_halt` stamps `_halt_started_at`; `authorized()` calls
  `_maybe_auto_resume()`, which resumes ordinary halts after
  `PAUSE_DURATION_MIN` (60) minutes. Kill latches are NEVER auto-resumed
  — only `release_kill` re-arms them. Restored (persisted) halts expire on
  the first `authorized()` check after their time is up.
- **Class:** F
- **Guard:** `test: tests/test_safety_training.py::test_halt_auto_resumes_after_pause_minutes`

## 2026-09-23 — Phase 2: scoring-math integrity

### FIX-2026-09-23-06 — dead neuro blend damped every score by 0.7x
- **Symptom:** with NEURO_BLEND_ENABLED=False (default), every cortex
  score was multiplied by 0.7 before reaching the penny gate, the
dynamics threshold band, and ENTRY_THRESHOLD — all calibrated in
un-damped cortex units. A 0.85-conviction setup scored 0.595 and could
never clear the 0.85 penny bar (it needed a cortex-only 1.21,
impossible on the tanh scale).
- **Root cause:** `_score_pipeline` unconditionally computed
`(1 - 0.3) * (base.score + cal_adj) + 0.3 * neuro_score` while
`_compute_neuro_score` returned 0.0 with the gate off — a fixed 0.7x
damping of every score for zero neuro contribution.
- **Fix:** branch on NEURO_BLEND_ENABLED: gate off ->
`blended = base.score + cal_adj` (un-damped, restoring the designed
meaning of the threshold band and PENNY_SCORE_BAR); gate on -> the
0.7/0.3 blend exactly. Recorded (not retuned): Dynamics' rolling
90th-percentile calibration now observes un-damped scores.
- **Class:** G
- **Guard:** `test: tests/test_scoring_integrity.py::test_gate_off_passes_cortex_through_undamped`

### FIX-2026-09-23-07 — PENNY_SCORE_BAR re-derived against the un-damped scale (kept at 0.85)
- **Symptom:** the 0.85 sub-dollar bar's meaning had drifted with the
damped scale — under damping it demanded an impossible cortex-only
1.21, and no recorded derivation matched what the gate compared.
- **Root cause:** no derivation was ever recorded for 0.85 against the
score scale the gate actually sees.
- **Fix:** value kept at 0.85 and re-derived in `immune.py`: the gate
compares |score| in un-damped cortex (tanh) units; atanh(0.85) ~= 1.26
— the weighted indicator z-sum must reach ~1.26 sigma of conviction
AND survive the Nash penalty (up to 0.15) — and 0.85 sits deliberately
above the normal admission band, so sub-dollar names always face a bar
regular names never see. Not lowered to make trading easier.
- **Class:** G
- **Guard:** `test: tests/test_scoring_integrity.py::test_penny_bar_above_normal_entry_threshold`

### FIX-2026-09-23-08 — sleep replay rewarded losers (sign inferred from drive magnitude)
- **Symptom:** every losing pattern replayed during sleep received
`apply_reward(+1.0)` — sleep consolidated the exact losing behavior it
was built to punish, at 3x replay drive.
- **Root cause:** `_replay_pattern` inferred the reward sign from the
drive magnitude (`1.0 if weight >= 1.0 else -1.0`); losers replay at
3x drive (SLEEP_LOSS_WEIGHT=3.0 >= 1.0), so the inference was exactly
backwards for the patterns the 3x weighting exists to prioritize.
- **Fix:** explicit polarity threaded end to end: `replay_weights`
emits `(pattern, drive, won)`; `select_patterns` / `run_cycle` /
`run_sleep_replay` carry the triple; `_replay_pattern(pattern, weight,
won)` applies -1.0 for losers, +1.0 for winners — never inferred from
drive. Losers keep 3x replay drive, so punishment stays dominant per
CONTRACT. All producers, consumers, and tests updated to triples.
- **Class:** G
- **Guard:** `test: tests/test_scoring_integrity.py::test_loser_replay_applies_negative_reward`

## Standing defenses (why mid-session breakage should stay rare)

1. **Smoke harness** — `scripts/smoke_live.py`: real Gateway, false
   closes, false quotes, replay through the real tick path, zero
   orders. 23/23 checks. Run before any session.
2. **PipelineMonitor** — `src/hanoon_prime/monitor/pipeline.py`:
   daemon probing IB link, bar freshness, brain advancement, entry-eval
   failure bursts (Class D), and subs presence every 15s; journals
   incidents, sets a heal flag the cycle consumes to force
   re-subscribe. Telemetry: `GET /pipeline`.
   Guard: `test: tests/test_pipeline_monitor.py`.
3. **Journal enforcer** — `tests/test_fixes_journal.py` makes this
   document a live contract: entries must classify + guard, guards
   must exist and mark back (`Regression: FIX-...`), smoke guards
   must name real checks, and a static watchdog bans array
   truthiness patterns (Class C) unless annotated `# array-safe`.
4. **Dormant-by-design:** `monitor/watchdog.py` (panic auto-flatten)
   and `decision_health`/`enforcement`/`position_monitor`/
   `reconciliation` remain unwired — deliberate, pending review;
   recorded here so dormancy is a decision, not an accident.

## 2026-09-23 — Phase 3: validation honesty

### FIX-2026-09-23-09 — WFA scope disclosure: no purge/embargo theater, DNN not exercised
- **Symptom:** the walk-forward report presented a verdict with no
  statement of what the harness does NOT validate: the shipped WFA never
  fits anything (wiring purge/embargo into the run would be theater, not
  leakage control), and the sim scores via the standalone cortex path
  (`hands.simulate_ticker` → `Cortex.evaluate`) — the production MetaDNN
  gatekeeper (`META_DNN_ENABLED=True`, `brain/orchestrator.py::_dnn_gate_veto`)
  is never exercised, so the verdict never covered DNN veto/sizing
  behavior, and the DNN's own fitting trials were never counted in
  deflation.
- **Root cause:** `wfa.serialize` emitted only numbers; nothing in code or
  report disclosed the validation scope, and nothing pinned the
  no-fit/no-DNN boundary.
- **Fix:** `VALIDATION_CAVEATS` in `src/hanoon_prime/wfa.py` (three
  bullets: standalone cortex scorer, DNN excluded + its fitting trials
  uncounted, purge/embargo intentionally not applied), serialized by
  `serialize()` and rendered into `scripts/wfa_report.py` under "Validation
  scope (what this report does NOT cover)"; honesty note in
  `run_walk_forward`'s docstring. The run itself is unchanged — fitting
  nothing is the correct behavior; claiming otherwise would be the bug.
- **Class:** H
- **Guard:** `test: tests/test_validation_honesty.py::test_serialize_carries_validation_caveats`, `test: tests/test_validation_honesty.py::test_run_walk_forward_never_calls_purge_helpers`, `test: tests/test_validation_honesty.py::test_sim_does_not_consult_trained_dnn_artifact`, `test: tests/test_validation_honesty.py::test_sim_scoring_path_excludes_dnn_machinery`

### FIX-2026-09-23-10 — deflation trial count included tickers the pooled Sharpe excluded
- **Symptom:** `_count_trials` counted ticker×fold cells for every ticker
  with ≥2 trades per cell, while `_pooled_returns` only admits tickers with
  ≥ MIN_TRADES (30) OOS trades — the deflation null was calibrated to a
  larger trial set than the Sharpe it deflated, over-penalizing thin
  universes.
- **Root cause:** trial counting and admission used different ticker sets.
- **Fix:** `_count_trials` skips tickers below the MIN_TRADES floor, so the
  deflation penalty is calibrated to the same admissible set that feeds the
  pooled Sharpe. Direction of effect (pinned by test): fewer trials →
  lower 95th-percentile best-by-chance Sharpe → deflated edge moves UP
  (less conservative).
- **Class:** G
- **Guard:** `test: tests/test_wfa.py::test_count_trials_counts_admissible_ticker_folds_with_trades`, `test: tests/test_wfa.py::test_count_trials_skips_inadmissible_tickers`, `test: tests/test_validation_honesty.py::test_count_trials_admissible_only`, `test: tests/test_validation_honesty.py::test_deflation_less_conservative_with_fewer_trials`

### FIX-2026-09-23-11 — Phase-4 P5 gate weaker than protocol; P1–P6 never evaluated
- **Symptom:** `run_paper` gated P5 on `wfa.verdicts()` PASS (deflated > 0,
  pooled > 0, no PBO check) instead of the pre-locked protocol thresholds
  (deflated OOS Sharpe > 0.05 AND PBO < 0.05) — and `PaperVerdict.evaluate()`
  was never called, so the report's criteria dict held only P5_wfa_pass and
  P1–P4/P6 never gated the verdict despite the docstring claiming P1–P6.
- **Root cause:** no separate protocol-threshold gate; the evaluate() call
  was missing.
- **Fix:** pre-locked `P5_DEFLATED_FLOOR = 0.05` / `P5_PBO_CEILING = 0.05`
  in `scripts/paper_run.py`, applied as a separate strictly-additive
  `P5_protocol_thresholds` gate (strict inequalities; flips PASS→FAIL,
  never FAIL→PASS); `v.evaluate()` called before the P5 override so P1–P6
  all gate the verdict. Thresholds recorded in `lock_values`.
- **Class:** G
- **Guard:** `test: tests/test_validation_honesty.py::test_p5_protocol_gate_blocks_weak_engine_pass`, `test: tests/test_validation_honesty.py::test_p5_protocol_gate_passes_above_strict_thresholds`, `test: tests/test_validation_honesty.py::test_p5_protocol_gate_strict_boundaries`, `test: tests/test_validation_honesty.py::test_run_paper_evaluates_full_p1_to_p6`

## 2026-09-23 — Phase 4: telemetry auth hardening

### FIX-2026-09-23-12 — telemetry mutation gate failed open; bearer token served over HTTP; /flatten invisible to audits
- **Symptom:** three compounding defects on the POST-mutation surface
  (`do_POST` → `_authorized()`, `src/hanoon_prime/telemetry.py`), with the
  API on a public cloudflared tunnel: (1) `_authorized()` returned True
  when auth was disabled or the token was unset — one misconfiguration (or
  an unreadable token file) silently made `/safety-net`, `/config` and
  `/flatten` unauthenticated, i.e. remote control of the kill switch,
  direction mode, EOD flatten, and full position flatten; (2) `GET /auth`
  handed the bearer token to any CORS-allowed browser origin AND to any
  client sending no Origin header at all (`curl .../auth`), so the token
  "protecting" the mutations was fetchable by anyone — the bearer gate was
  theater; (3) `POST_ROUTES` was a stale hand-maintained subset
  `{"/safety-net", "/config"}` that omitted `/flatten` while
  `POST_HANDLERS` served it — invisible to allow-list audits.
- **Root cause:** fail-open default ("Open when auth disabled or token
  unset"); token auto-provisioning designed around "the tunnel is the
  perimeter", which collapsed once the tunnel became public; two sources
  of truth for the POST route inventory.
- **Fix:** `_authorized()` is now FAIL-CLOSED — denies on auth off, missing
  token, or bad/missing `Authorization` header (401); `GET /auth` returns
  metadata only (`auth`/`scheme`/`provisioning: manual`/`hint`) and NEVER
  the token, which lives only in `runtime/telemetry.token` (0600) — the
  user copies it into the dashboard's bearer-token field; `start()` log
  line corrected (auth OFF now means mutations DISABLED, not "ungated");
  `POST_ROUTES = frozenset(POST_HANDLERS)` — single source of truth, so
  `/flatten` is auditable. `tests/test_telemetry_auth.py` updated to the
  new contract (its old tests encoded the vulnerability).
  Frontend follow-up required: the dashboard's auto-provisioning via
  `GET /auth` no longer yields a token — it needs a manual token field;
  until then, dashboard mutation buttons will 401. The existing token file
  is unchanged, so any token already in the user's browser keeps working.
- **Class:** F
- **Guard:** `test: tests/test_telemetry_auth_failclosed.py::test_denies_when_auth_disabled_even_with_token`, `test: tests/test_telemetry_auth_failclosed.py::test_post_denied_when_auth_disabled`, `test: tests/test_telemetry_auth_failclosed.py::test_every_mutation_route_denied_without_token`, `test: tests/test_telemetry_auth_failclosed.py::test_flatten_gated_but_reachable_with_token`, `test: tests/test_telemetry_auth_failclosed.py::test_post_routes_covers_every_handler`, `test: tests/test_telemetry_auth.py::test_token_never_served_without_origin`, `test: tests/test_telemetry.py::test_post_without_token_rejected_401`, `test: tests/test_telemetry.py::test_post_with_wrong_token_rejected_401`

### FIX-2026-09-23-13 — DNN gatekeeper failed open on missing/defective/never-trained models; no bounded self-correction existed
- **Symptom:** the MetaDNN gatekeeper reported "last trained: NEVER" yet
  kept passing entries; missing, corrupt, defective, or guard-rejected
  artifacts stepped aside instead of blocking; a drift z-score of 3.8107
  produced no automated response — the gate was documented as a safety
  control but was silently disabled in exactly the states where it
  mattered most.
- **Root cause:** inference treated an unusable model as "no opinion"
  (admit-by-default) instead of a blocking condition; no prediction
  ledger, calibration monitor, derating policy, or correction journal
  existed.
- **Fix:** new `src/hanoon_prime/brain/self_correction.py` (200 lines):
  append-only prediction ledger under `runtime/self_correction/` with
  ticker-specific resolution; rolling calibration monitor (classification
  accuracy / Brier / slope / overconfidence gap over 200 resolved
  predictions); monotonic derating policy `{1.0, 0.5, 0.0}` — immediate
  derate down, recovery needs 100 new healthy samples and moves one step
  per evaluation; drift z>3 caps weight at 0.5, z>5 abstains; critical
  derating writes a `PENDING_HUMAN_APPROVAL` retrain request — the module
  contains zero training machinery by construction (source-level test
  asserts this); high-confidence mistakes go to an append-only correction
  journal with a persistent `reviewed.jsonl` audit, drained (review-only)
  by consolidation's sleep cycle, never replayed into parameters.
  `meta_label_dnn.py` / `meta_label.py`: an unusable DNN now BLOCKS with
  `(False, 0.5, 0.0)` plus a CRITICAL `DNN BLOCK` log instead of stepping
  aside; blocks are excluded from drift history. Changes are inert until
  a user-authorized restart.
- **Class:** F
- **Guard:** `test: tests/test_self_correction.py::test_log_and_resolve`, `test: tests/test_self_correction.py::test_resolve_is_ticker_specific`, `test: tests/test_self_correction.py::test_block_keeps_drift_history_clean`, `test: tests/test_self_correction.py::test_queues_high_confidence_mistakes`, `test: tests/test_self_correction.py::test_drain_preserves_reviewed_audit`, `test: tests/test_meta_label_dnn.py::test_infer_blocks_when_never_trained`, `test: tests/test_meta_label_dnn.py::test_infer_admits_with_trained_artifact`, `test: tests/test_meta_label_dnn.py::test_infer_blocks_without_artifact_despite_fake_predict`
- **Resolution 2026-09-24:** the two legacy tests that encoded the old
  admit-by-default assumption were updated to the fail-closed semantics
  at the user's direction: `test_infer_admit_above_threshold` became
  `test_infer_blocks_when_never_trained` (injected layers with no
  artifact/report must return `(False, 0.5, 0.0)`), and
  `test_size_scale_monotonic` became
  `test_infer_blocks_without_artifact_despite_fake_predict`; a new
  `test_infer_admits_with_trained_artifact` preserves admit-path coverage
  for models that pass the readiness gate. P(Win)->scale monotonicity
  remains pinned at the unit level by
  `TestCalculateMetaSizeScale::test_monotonic`.
- **Resolution 2026-09-24 (R3):** the module was split into
  `self_correction.py` (ledger + calibration, 193 lines) and
  `self_correction_policy.py` (derating + journal, 193 lines) so the
  200-line contract survives Black formatting; both modules are fully
  typed, documented, and covered by the no-training-machinery guard.
