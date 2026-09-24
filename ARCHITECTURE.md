# Hanoon Prime — Architecture Study

**Date:** 2026-09-23 · **Repo:** `/Users/mdsabersajib/Downloads/hanoon_prime` (user's Mac, read-only study — nothing modified)
**Method:** Four parallel code-study workstreams, each reading the Mac repo via `files.read`/`files.list` only. Every technical claim below is cited as `repo-relative-path:line` and was verified by direct read on 2026-09-23. Exhaustive detail lives in the part files: `part1_live_path.md`, `part2_organs.md`, `part3_safety.md`, `part4_validation.md` (same directory).

**One-line summary:** a ~29k-line, 185-module JULI-brain → IB-Gateway paper-trading system on IBKR paper (port 4002) whose docs are unusually honest — its own walk-forward gate currently FAILS — and whose live path contains several verified safety and scoring defects detailed in §8.

---

## 1. Live money path: tick → order fill

### 1.1 Boot (once)
1. `python3 -m hanoon_prime.cli` → `main()` — `src/hanoon_prime/cli.py:85-114`. Tickers = CLI args, else seed universe `LIQUID_US_SEED` (22 names) — `cli.py:94`, `immune.py:227-249`.
2. `IBStreamingBot(account="PAPER")` — `cli.py:98` → `ib_adapter.py:48-101`. Wires `JuliBrain`, `IBStreamer`, `IBExecutor`, `PipelineMonitor`; journal → `runtime/journal_live.jsonl`; `executor.on_fill_confirmed = self._confirm_fill`. Note the bot's own `Hippocampus(safety_enabled=False)` — `ib_adapter.py:64`.
3. `bot.run_paper(tickers)` → `connect(port=IB_PAPER_PORT=4002)` → `run()` — `ib_adapter.py:197-200`; main loop `while self._running: self._cycle(poll, pnl)` — `ib_adapter.py:160-180`.

### 1.2 The cycle (~1 s)
`BotCycleMixin._cycle` — `ib_cycle.py:487-518`, in order:
1. Sleep-manager check; gateway supervision (reconnect + resubscribe).
2. `executor.sync_from_ib(...)` — IB is source of truth: sweep zombies, adopt orphans, re-attach protection (`_protect.py`), rebuild `_brackets` (`ib_bracket.py:29-47`), `read_ib_positions` (`_ib_sync.py:106-134`).
3. `_sweep_stale_orders()` — cancels stale (>60 s, PendingSubmit/PreSubmitted) DAY entry parents — `ib_cycle.py:262-298`.
4. `_sync_subs()` — market-data subscriptions from the rotated budget pool.
5. Manual/EOD flatten (`ib_cycle.py:845-906`) — **skips the brain**, sends MKT flattens directly.
6. `_run_brain_cycle(poll, pnl, started)` — `ib_cycle.py:561-598` → `juli.tick(watch, ...)` — `juli.py:44-74`.

`JuliBrain.tick`: re-entrancy lock → `governor.begin_cycle()` (resets 2-entry budget — `orchestrator.py:258-260`, `policy/governor.py:26-28`) → `_data_preamble` → `_eval_window` (`juli.py:76-106`; tickers with ≥20 prices, `READY_FLOOR=20`, `juli.py:21`; no-data rest rotated 4/cycle, `EVAL_WINDOW=4`, `juli.py:20`) → each ticker `brain.decide_entry(...)` → `_evaluate_exits` (`juli.py:145-175`).

### 1.3 Entry decision — `decide_entry` (`brain/orchestrator.py:263-306`), exact order
1. `_check_snapshot_valid` — veto `no_data` if snap missing / <20 prices / NaN last/bid/ask/mid — `orchestrator.py:318-330`.
2. `_score_candidate` (`orchestrator.py:336-366`): `compute_alpha_from_snap` — `juli_feed.py:59-77` → `compute_all_alpha` (**5 core + 22 higher-order = 27 keys** — `brain/indicators.py:30-55`), fallback to 5-core `cerebellum.compute_alpha`, plus live tape absorption keys.
3. `self.tick` → `_evaluate_fast` (`orchestrator.py:837-873`): regime fallback → cross-asset modifier (clamped ±`CROSS_ASSET_MOD_BOUND=0.04`) → horizon classify → horizon bandit → absorption-scalp override → regime weight hot-swap → strategy select → episodic bias (±`EPISODIC_MOD_BOUND=0.10`) → **`_score_pipeline`**.
4. **`_score_pipeline`** (`orchestrator.py:1286-1319`):
   - `cortex._threshold = dynamics.threshold`; `base = cortex.evaluate(alpha)` — z-score → `tanh(Σw·z/Σ|w|)` over 27 keys (`cortex.py:80-115`).
   - `nash_pred` modifier (`orchestrator.py:770-772`); `neuro_score` = **always 0.0** (`NEURO_BLEND_ENABLED=False` — `immune.py:132`, `orchestrator.py:747-768`).
   - `blended = 0.7 × (base.score + cal_adj) + 0.3 × 0` — **a 30% damping of cortex, not a live blend** (`orchestrator.py:1296`).
   - `mods` = halim (±0.03) + episodic + nash_op + news (±0.03) + cross − advisor_delta + thinker + strategy (`orchestrator.py:1233-1253`); `raw = blended·regime_mul + somatic + precision·mods`.
   - `_stabilize` (`orchestrator.py:1386-1397`): Nash bounded penalty (`NASH_PENALTY_MAX=0.15`, veto bands 0.45/0.55 — `brain/config.py:196-199`) → EOD zero → `dynamics.process` (hysteresis ±0.03).
5. `_maybe_size` (`orchestrator.py:1399-1422`): if `|stabilized| ≤ dynamics.threshold × patience` (scalp patience 1.00 — `horizons.py:58-65`) → HOLD `not_sized`. Else `risk.evaluate(...)` (`brain/risk.py:190-252`): preflight, Kelly × `KELLY_FRACTION=0.25`, realized-EV (**advisory only** — `_ev_scale` ∈ {1.0, 0.75, 0.5}, "NEVER refuses" — `risk.py:36-41,190-205`); `open_positions ≥ 3` → shares=0; quality penalties (VWAP chase ≤0.10, momentum ≤0.08, spread ≤0.06 — `brain/config.py:91-103`); `_size` (`risk.py:107-145`): `shares = max(1, min(5000/price, 50/risk_per_share, ...))`; stop 2.0×ATR, target 6.0×ATR (3:1).
6. `_extract_thought`: direction≠0 and `|score| ≥ DIRECTION_MIN_SCORE=0.02` else HOLD `no_signal` (`immune.py:60`, `orchestrator.py:368-388`).
7. `_apply_fast_gates` (`orchestrator.py:419-448`): session → direction → penny → halt (+ safety `authorized` snapshot).
8. `_apply_strategy_gates` (`orchestrator.py:449-483`): `ENTRY_COST_AVERSE_GATE`/`ENTRY_REGIME_GATE` both False → pass-through.
9. `governor.may_enter`: ≤2 entries/cycle (`immune.py:55`), 60 s per-ticker cooldown (`immune.py:56`) — `policy/governor.py:31-42`.
10. `_admit_verdict` (`orchestrator.py:535-552`): `_dnn_gate_veto` (`META_DNN_ENABLED=True`, `META_WIN_THRESHOLD=0.52` — `brain/learning_config.py:34-36`) → `_portfolio_admit` (`orchestrator.py:587-625`; exposure<1.0, concentration≤0.25, count<3 — `policy/portfolio_gate.py:55-96`) → ENTER verdict.

### 1.4 Verdict → order
`_finish_cycle` (`ib_cycle.py:600-633`) → `_execute_entries` (`ib_cycle.py:645-662`): ENTER only if `market_open` **and** `monitor.bar_feed_fresh()`, else "ENTRY SUPPRESSED: bar feed stale" → `_execute_verdict` (`ib_cycle.py:690-716`): sizing>0, live bid/ask, not already open → `price = mid` → `executor.place_bracket(...)` → `brain.note_entry` (cooldown stamped **at placement** — `ib_cycle.py:716`).
`place_bracket` (`ib_executor.py:62-151`): `ib.bracketOrder(action, shares, round(mid,2), target, stop)`, `tif="DAY"`, `outsideRth=True` — **parent is a LIMIT at the rounded mid** (ib_insync builds a LimitOrder parent); children = LMT take-profit + STP stop-loss (OCA). SELL blocked under `long_only`.
Fill accounting: `sync_from_ib` → `_notify_open_fills` (`ib_executor.py:403-418`) → `_confirm_fill` (`ib_cycle.py:928-941`): `brain.register_position` → `ExitPolicy.register` (`brain/exits.py:329-346`); `_attach_position_watchers` (`ib_cycle.py:943-982`) enqueues `hard_stop_breach` on tick; `_drain_event_exits` (`ib_cycle.py:984-997`) closes sub-cycle.

### 1.5 Exit path (how positions close)
1. **Brain exits** — `juli._evaluate_exits` → `brain.check_exit` (`orchestrator.py:1711-1739`) → `ExitLadder.evaluate` (`brain/exit_ladder.py:62-104`), first decisive tier wins:
   - TIER1 hard stop: price breaches `stop_price` — absolute (`exit_ladder.py:139-153`).
   - TIER2 JULI verdict: `ExitPolicy.exit_likelihood` (8 pillars — `brain/exits.py:283-327`) vs learned thresholds (`adaptive_thresholds.json`, 2,091 updates); hysteresis disabled (`HYSTERESIS_EXIT_ENABLED=False` — `immune.py:128`) → soft exits immediate.
   - TIER3 mechanical `ExitPolicy.evaluate` (`brain/exits.py:348-390`): absorption-break → profit_lock tiers (`brain/config.py:77-82`) → giveback (`GIVEBACK_KEEP_RATIO=0.55`) → stale (`STALE_EXIT_MINUTES=120.0`) → consolidation (<0.1% for 6 pulses).
   - → `close_position` — **MKT** DAY outsideRth (`ib_executor.py:531-552`). IB-native bracket children also fire independently.
2. **Tick-watcher hard stop** — sub-cycle MKT close on breach.
3. **EOD flatten** — last 5 min (`policy/trading_policy.py:41-42`); intraday horizons only (`holds_through_close` — `horizons.py:110-113`); brain skipped.
4. Close → `_record_exit` (`ib_executor.py:285-330`): P&L via `get_ib_pnl` (IB fills → return fraction — `_ib_sync.py:19-27`); journaled; appended to `_closed_trades`.

### 1.6 Learning loop (per close)
`_reflect_closed` (`ib_cycle.py:1005-1033`) → `brain.on_trade_close` (`orchestrator.py:1450-1500`):
1. `_ironclade_gate` — only sources `{"real_trade","ib_fill","ib_paper","reconciled_exit"}` learn (`brain/config.py:209-214`).
2. `_adapt_threshold` → `dynamics.adapt_threshold` (+0.01 if err≥0.50, −0.005 if <0.45; clamped [0.45, 0.70] — `brain/dynamics.py:112-143`, `brain/config.py:29-31`); `memory.threshold` persists.
3. `_update_rpe`; `_learn_exit_thresholds` (`AdaptiveThresholds.update_from_outcome`); `exits.deregister`; `nash.record_outcome`; `_store_neuromorphic_outcome` (STDP traces mutate, write-back gated); `episodic.add`; `extinction.record`; `_learn_strategy_organs` (meta-label, horizon bandit, strategy bandit/registry; `REGIME_MIN_TRADES=15`).
4. `_learn_from_real` (`orchestrator.py:1570-1616`): `Reflector.on_trade_close` — loss-aversion 2.4:1 (`PENALTY_SCALE=1.2` vs `REWARD_SCALE=0.5`, LR 0.02, decay 0.999 — `brain/config.py:52-58`) → regime-weight hot-swap → `realized.add_outcome` → **`exits.adapt_from_realized`** (retunes giveback keep-ratio ±0.20, stale window ±60 min — `brain/exits.py:254-281`, `brain/config.py:84-88`).

### 1.7 Gate/veto inventory (condensed)
Session toggles · `direction_mode="long_only"` (overrides `SHORT_ALLOWED=True` — `policy/trading_policy.py:26,48-55`, `immune.py:59`) · `DIRECTION_MIN_SCORE=0.02` · penny bar: sub-$1 needs |score|≥0.85 (`PENNY_PRICE=$1.00`, `PENNY_SCORE_BAR=0.85` — `policy/trading_policy.py:57-66`, `immune.py:46-48`) · safety `authorized` snapshot (see §3) · score admission `|score| > dynamics.threshold × patience` · governor 2/cycle + 60 s cooldown · DNN veto P(win)<0.52 · realized-EV advisory-only · mechanicals $5k/​$50/3 positions · portfolio gate (exposure<1.0, concentration≤0.25) · feed-freshness suppression · 60 s stale-parent sweep · EOD flatten. `MAX_SPREAD_BPS=5` is declared (`immune.py:38`) but the part-1 trace found **no spread veto on the live entry path** (part 3 lists it as an ib_cycle execution-path veto without a line ref — treat as unverified).

### 1.8 Corrections to stale doc claims (verified 2026-09-23)
- `brain/cerebellum.py` **does not exist**; the 27-indicator alpha is `brain/indicators.py:30-55`. Top-level `cerebellum.py` is only the 5-core fallback.
- "0.7·cortex + 0.3·SNN" is wrong: with `NEURO_BLEND_ENABLED=False` the live formula is `0.7 × cortex + 0.3 × 0` — pure damping.
- Entry parent is **LIMIT at mid**, not market (only closes are MKT).
- `_check_safety` / `_exec_decision` do not exist in current `ib_cycle.py`; execution is `_execute_verdict`/`_execute_entries`.
- The realized-EV "gate" never refuses (docstring: "NEVER refuses… only scales").
- `decide_entry` never reads `cortex`'s BUY/SELL verdict — only score/direction feed sizing; the R1 "cortex is sole verdict emitter" claim doesn't match the live admission path.

## 2. Brain organs — live vs dead

Audit of every learning/adaptive organ against `src/hanoon_prime/` (full table in `part2_organs.md`).

| Organ | LIVE_CHECK (2026-09-18) claimed | Verified 2026-09-23 |
|---|---|---|
| STDP reward learning | REPAIRED + GATED | **GATED OFF** — `NEURO_LEARN_ENABLED=False` (`immune.py:153`); traces mutate (`brain/neurons/bridge.py:148-149`) but write-back to scorer weights only when flag true (`:150-151`) |
| Neuromorphic score/blend | (implied live) | **GATED OFF** — `NEURO_BLEND_ENABLED=False` (`immune.py:148`); `_compute_neuro_score` returns 0.0 (`orchestrator.py:760-761`); feed path still runs every tick (cost without effect) |
| AttractorMemory | PERSISTED (opt-in) | **LIVE** — writes on every real close (`orchestrator.py:1459-1475` → `bridge.store_outcome`); production persists (`ib_adapter.py:47`, `runtime/attractor_memory.json` current). Recall-to-scoring consumers not fully mapped |
| SleepReplayEngine | REPAIRED + GATED | **LIVE (condition-gated, not flag-gated)** — constructed `orchestrator.py:176-185`; S2 `_maybe_sleep_replay` (`consolidation.py:177-190,505-518`); fires on 30-min idle or session close, ≥3 buffered patterns (`SLEEP_MIN_PATTERNS=3` — `learning_config.py:94`). Effect bounded only because neuro gates are off |
| DynamicThresholdAdapter (neuro) | WIRED + GATED | **GATED OFF** — `NEURO_ADAPTIVE_THRESHOLD_ENABLED=False` (`immune.py:157`); feed runs, then fixed thresholds restored (`adaptive_thresholds_wiring.py:30-53`) |
| MoE experts | WIRED + GATED | **GATED OFF** — `NEURO_MOE_GATE_ENABLED=False` (`immune.py:162`); `route_experts` immediate no-op (`moe_gate.py:112-134`). Classic cross-asset lead-lag scoring is separately LIVE (`orchestrator.py:851-854`) |
| Nine-rule learned exits | LIVE | **LIVE** — `update_from_outcome` on every close (`orchestrator.py:1533-1556`); consumed in exit-ladder Tier 2 (`exit_ladder.py:154-165`); `runtime/adaptive_thresholds.json` update_count 2091 |
| ExitPolicy.adapt_from_realized | LIVE ("only live learning") | **LIVE** (`orchestrator.py:1624`, `brain/exits.py:253-281`) — but it is **not** the only live learning (headline is false) |
| Episodic memory | (assumed live) | **LIVE** — `episodic.add` on close (`orchestrator.py:1467-1477`); modifier read every tick, ±0.10 bound (`orchestrator.py:952-967`, `episodic.py:100-108`) |
| Strategy bandit/registry | (Phase H) | **LIVE** — per-tick Beta-posterior select (`strategy_bandit.py:72-107`); updates on real + shadow closes; `runtime/juli_strategy_bandit.json`: 5,787 trials, 23.9M selects, 893k overrides |
| StrategyResearch / HALIM `/v1/research` | (Phase H) | **LIVE background loop** — S2 `_maybe_research` (`consolidation.py:177-190,203-210`), 5-min throttle, no feature flag (`strategy_research.py:62-107`); POSTs `http://127.0.0.1:8765/v1/research`; graceful on failure |
| HALIM modifier | warming up | **LIVE** — S2-polled, 60 s cache (`halim_adapter.py:125-133,213-233`); folded into score (`orchestrator.py:735-743,858-863`) |
| Meta DNN gatekeeper | ACTIVE, never trained | **LIVE** — `META_DNN_ENABLED=True` (`learning_config.py:29-31`); `_dnn_gate_veto` (`orchestrator.py:542-544`); 9→32→16→1 MLP, admit if P(win)≥0.52 (`meta_label_dnn.py:211-237`); train report accuracy 0.5896, p_win_median 0.3325 (`juli_meta_dnn.report.json`); **fails open** on defective model (see §8) |
| Nash learning/veto | (implied live) | **LIVE** — predict every tick (`cognitive/nash.py:44-68`), penalty ≤0.15 in `_stabilize`, outcome recorded per close |
| Reflector weight adaptation | (learning loop) | **LIVE** — 2.4:1 loss/reward ratio (`reflection.py:44-55,77-104`, `brain/config.py:53-54`); persists to `juli_regime_weights.json` |
| LearnedExitPolicy | (not in audit) | **PARTIAL** — `record()` every close (`orchestrator.py:1478`) but `get_policy()` has **no callers**: write-only learning |
| NetworkStepper | DELETED | **DEAD** — confirmed gone |

### Feature-flag inventory (verified values)
From `immune.py`: `DYNAMIC_PRIOR_TOP_ENABLED=True`(:81) · `PROBE_RECOVERY_ENABLED=False`(:94) · `CONTRARIAN_MODE_ENABLED=False`(:105) · `CALIBRATION_NUDGE_ENABLED=False`(:114) · `HYSTERESIS_EXIT_ENABLED=False`(:122) · `DELIBERATION_TRACE_ENABLED=False`(:126) · `HALIM_EVIDENCE_LEARNING=False`(:127) · `NEURO_BLEND_ENABLED=False`(:148) · `NEURO_LEARN_ENABLED=False`(:153) · `NEURO_ADAPTIVE_THRESHOLD_ENABLED=False`(:157) · `NEURO_MOE_GATE_ENABLED=False`(:162) · `TELEMETRY_AUTH_ENABLED=True`(:178) · `ENTRY_REGIME_GATE=False`(:196) · `ENTRY_COST_AVERSE_GATE=False`(:197) · `ALLOW_EXTENDED_HOURS=True`(:272) · `FRACDIFF_ENABLED=True`(:281) · `LABEL_ENABLED=True`(:293) · `WFA_PURGE_ENABLED=True`(:302). From `brain/learning_config.py`: `META_DNN_ENABLED=True`(:29-31).
⚠️ The three AFML flags carry "off-by-default" comments but are set **True** — intent unconfirmed (comment/code drift).

### Key corrections to LIVE_CHECK.md
1. "Only live learning is ExitPolicy" is materially false — at least 8 other learners run on the close path.
2. Sleep replay is condition-gated and live, not flag-off.
3. Nine-rule "zero callers" is stale — live via exit-ladder Tier 2.
4. Attractor persistence is production-enabled, not merely opt-in.

---

## 3. Safety nets

### 3.1 Hard limits (literals locked by `tests/test_constants_contract.py`; R6 forbids env configurability)

| Limit | Value | Defined | Enforced |
|---|---|---|---|
| Max position notional | $5,000/trade | `immune.py:57` | sizing ceiling `brain/risk.py:136-166` |
| Max loss per trade | $50 | `immune.py:58` | sizing ceiling, never raises |
| Max concurrent positions | 3 | `immune.py:59` | `risk.py:213-217` (≥3 → zero-size); `portfolio_gate.py:69-72` |
| Daily loss → ordinary halt | −$200 | `immune.py:60` | `brain/policy/safety.py:76-94` |
| Daily loss → latched kill | −$500 | `immune.py:61` | `safety.py:76-94,131-164` |
| 3 consecutive losses → pause | 3 | `immune.py:62` | `safety.py:76-94` — **indefinite halt; `PAUSE_DURATION_MIN=60` (`immune.py:63`) is never read** |
| Max entries/cycle; reuse cooldown | 2; 60 s | `immune.py:55-56` | `policy/governor.py:31-42` |
| EOD flatten | enabled, 5-min window | `policy/trading_policy.py:29-35` | `ib_cycle.py:904-934` (MKT, brain skipped) |

Strict `<` comparisons: exactly −$200.00 / −$500.00 does **not** trip.

### 3.2 SafetyProducer semantics (`brain/policy/safety.py`)
- Defaults `enabled=False, halted=False, latched=False` (`safety.py:29-43`); every new `ConsolidationEngine` constructs a fresh disabled producer (`brain/consolidation.py:73`); the live bot explicitly keeps safety off (`ib_adapter.py:47-54`). **Nothing persists state — a restart clears even a latched kill.**
- `authorized()` check order (`safety.py:76-94`): (1) existing latch vetoes even when disabled; (2) **if disabled → allow immediately — the −$500 kill check is skipped unless already latched**; (3) `daily_pnl < −500` → latched kill; (4) existing halt → veto; (5) `daily_pnl < −200` → halt; (6) 3 consec losses → halt; (7) `position_count > 3` → halt.
- Ordinary halt: sets fields, journals, notifies, returns `(False, reason)` — **never stops the process, never closes anything.**
- Kill: sets halted+latched, journals/notifies, invokes the one hook → `executor.cancel_all()` (`ib_adapter.py:95-104`) — **cancels open orders only; does NOT flatten positions.**
- `resume()` clears ordinary halt but not the latch; `release_kill()` re-arms; `set_enabled(False)` clears ordinary halt as a side effect.
- Wiring: slow cortex publishes `policy_state` every ~30 s (`consolidation.py:34-35,111-151`); fast path reads the snapshot in `_apply_fast_gates` (`orchestrator.py:416-458`) — **up to ~30 s of stale authorization**; no synchronous `authorized()` on the hot path.
- `SafetyNetStopped` exception exists (`ib_cycle.py:116-117`) with no live raise found. Sim-path `Hippocampus.check_safety_nets()` raises on the same trips but is constructed with `safety_enabled=False` in live (`ib_adapter.py:47-54`, `orchestrator.py:~150`) — inert.

### 3.3 Bypass inventory (ways limits get weakened/disabled)
1. `POST /safety-net {"action":"disable"}` — disables producer (kill check skipped unless latched); also clears any ordinary halt (`safety.py:54-58`).
2. `POST /safety-net {"action":"resume"}` — clears halt immediately (also forces `authorized=True` in the published snapshot — `orchestrator.py:674-683`).
3. `POST /safety-net {"action":"kill_release"}` — re-arms latched kill (`safety.py:95-101`).
4. **Process restart** — resets enabled/halted/latched; kill does not survive.
5. `POST /config` — cannot weaken the immune-literal limits, but **can disable EOD flatten** and enable long-hold horizons (`telemetry.py:559-593`).
6. Disabled-by-default + check ordering — a −$500 day with safety off never latches (contradicts CONTRACT.md principle 9 "non-overridable").
7. Kill = cancel-only (no flatten). 8. 30 s stale authorization window. 9. `PROBE_RECOVERY_ENABLED=False` — no bypass today; enabling it would let one gated entry override a halt.
- Auth posture on the POSTs: **disputed between workstreams** — parts 2 and 3 read `TELEMETRY_AUTH_ENABLED=True` in `immune.py:178` (bearer-gated in current code; token at `runtime/telemetry.token`; bind `127.0.0.1`); part 4 read `_authorized()` (`telemetry.py:695-699`) as fail-open when the flag is off and described off as the default. The literal is True per two independent reads; the fail-open *design* (returns True when off/unset) remains a latent risk, and the live tunnel is the real perimeter. Verify at runtime via `GET /auth` behavior.

---

## 4. Config and thresholds

- **The live entry threshold is `Dynamics.threshold`** — base `SIGNAL_THRESHOLD=0.58`, adaptive in `[THRESHOLD_MIN=0.45, THRESHOLD_MAX=0.70]` (`brain/config.py:24-27`). It gates entries at `orchestrator.py:1386` (`_maybe_size`), `~636` (`_size_entry`), and seeds `cortex._threshold` each cycle (`~1245`). Horizon patience scales it: scalp 1.00 → longterm 0.90 (`brain/horizons.py:42-49`).
- **Dashboard "LIVE Θ 0.4500"** = `brain.dynamics.threshold` (`brain/telemetry_facade.py:61`). It drifted to the 0.45 floor via **quintile self-calibration** (`brain/dynamics.py:62-77`: every tick, once ≥20 scores accumulate, the bar anchors at the 90th percentile of recent |scores|) — this runs with **zero trades**, so the live bar can sit at the floor while `memory.threshold` (persisted target, `runtime/juli_state.json`, refreshed only on real closes — `orchestrator.py:1518-1530`) still reads 0.58. They are different quantities; both are "correct."
- **Per-close adaptation** (`brain/dynamics.py:84-119`): prediction error ≥0.50 → +0.01; <0.45 → −0.005; +0.025 per losing confidence bin (≥10 losses, 0 wins), capped +0.15/close.
- **Dead/confusing constants:** `immune.ENTRY_THRESHOLD=0.65` (`immune.py:30`) — unused on the live path, unasserted by the contract test (legacy). `DECISION_THRESHOLD` (neuromorphic neuron, 0.6) — exported but consumed by nobody (`tests/test_dynamic_threshold_wiring.py:1-7`).
- **No hard confidence gate:** `CONFIDENCE_FLOOR=0.50` only floors reported confidence (`orchestrator.py:~1240`); `_maybe_size` gates on score only.
- **EV gate never refuses:** `ev = (p·r − (1−p)) · direction_mod` (`brain/ev_gate.py:75`) — **no fee drag subtracted**; `round_trip_cost_fraction()` exists (`immune.py:~192-205`) but is wired only via `ENTRY_COST_AVERSE_GATE`, default off (`immune.py:179`). `should_enter=False` paths exist (`ev_gate.py:104-110`, floors `ENTRY_EV_THRESHOLD=0.05` / `CONSERVATIVE_EV_MIN=0.07` — `brain/config.py:175-176`) but `RiskEngine.evaluate()` discards `should_enter` and only scales size ("NEVER refuses… only scales" — `risk.py:190-199`).
- EOD: within 5 min of close `_stabilize` zeroes scores (`orchestrator.py:800-810`); `_check_eod_flatten` (`ib_cycle.py:904-934`) force-closes intraday horizons only.

---

## 5. Validation harness

### 5.1 Walk-forward (`src/hanoon_prime/wfa.py`, 521 lines)
- Per ticker: load `<data_dir>/<TICKER>_1min.csv`; split into `DEFAULT_FOLDS=6` contiguous non-overlapping OOS windows after warmup (`wfa.py:92-112`); simulate with **static production weights** (no fitting → no in-fold leakage); a trade counts only if it exits strictly inside `[start, end)` (`wfa.py:262-266`).
- **Pooled universe stats:** `_pooled_returns` concatenates PnL of all trades from **admissible** tickers (≥`MIN_TRADES=30` OOS trades — `wfa.py:43,278-285`).
- **Deflated Sharpe** (`wfa.py:296-330`): observed pooled Sharpe − 95th percentile of best-by-chance null over `n_trials` bootstraps (`BOOTSTRAP_ITERS=2000`, seeded RNG — `wfa.py:45-46`). ≤0 ⇒ no edge after deflation.
- **PBO** (`wfa.py:345-390`): CSCV over trial half-splits (capped 64); 0.5 if <4 tickers. **Reported, never gated.**
- **Purge/embargo** helpers exist (`wfa.py:117-157,185-211`) but the shipped WFA never calls them (no training set); they're used by research harnesses (`train_meta_dnn.py`, phase scripts).

### 5.2 Reproduce command
`scripts/wfa_report.py` (101 lines; exit 0 only on universe PASS — `:94-101`):
```
python scripts/wfa_report.py --data-dir data/fixtures --tickers ALL --output reports/wfa_abs_new.json
```
`reports/wfa_abs_new.json` is byte-identical in content to `reports/phase2_wfa.json` (same fixtures, folds=6). The `abs` batch naming (written 2026-09-23 14:41–14:44 UTC) is undocumented in-repo. CI `wf-gate` (`.github/workflows/ci.yml:94-102`) runs the identical command with no `continue-on-error`.

### 5.3 Pass criteria (`wfa.py:425-462`)
Ticker PASS: ≥30 OOS trades **and** `ev_per_trade > 0` **and** per-trade Sharpe > 0. Universe PASS: ≥1 admissible ticker **and** `deflated_edge > 0` **and** `pooled_sharpe > 0`. Related R2 gate: `python -m hanoon_prime.backtest` + `scripts/check_profit_gate.py` (exit 1 if any ticker EV ≤ 0 — `backtest.py:115-136`).
**Latest result (2026-09-23): FAIL** — pooled Sharpe −0.0456, deflated edge −0.3068 (126 trials), PBO 0.4844; 4/23 tickers admissible, MARA the only PASS, 19 INSUFFICIENT.
**Protocol drift:** `task_plan.md` Phase 4 pre-locks `deflated > 0.05 AND PBO < 0.05`, but `scripts/paper_run.py:176-184` reuses `wfa.verdicts` verbatim — the stricter criteria are **not enforced in code**. Also `_count_trials` (`wfa.py:332-338`) counts ticker×fold cells over **all** tickers (126) while pooling only **admissible** ones (4) — deflation penalty calibrated to far more trials than feed the Sharpe (conservative bias on thin universes).

---

## 6. Telemetry / dashboard (`src/hanoon_prime/telemetry.py`, 1774 lines)

- `ThreadingHTTPServer` on **`127.0.0.1:TELEMETRY_PORT`** (`telemetry.py:~1699`); refresher rebuilds snapshot every 1 s (`SNAPSHOT_INTERVAL=1.0` — `:51`); dashboard (hanoonweb.xyz via cloudflared tunnel) polls `GET /snapshot` or holds `GET /stream` SSE (`SseRegistry.broadcast`).
- **29 GET endpoints**, none authenticated: `/snapshot`, `/stream`, `/health`, `/journal`, `/positions`, `/safety-net`, `/brain`, `/trades`, `/system2`, `/pipeline`, `/risk`, `/config`, `/halim`, `/verdicts`, `/session`, `/account`, `/resources`, `/inspection`, `/logs`, `/ib` (raw broker surface: quotes, orders, fills, executions, IB errors), `/decisions`, `/trade-quality`, `/trust`, `/vitals`, `/exec-quality`, `/monitors`, `/metrics`, `/auth`.
- **State-changing POSTs** — gated by `_authorized()` (`telemetry.py:695-699`), which **returns True when `TELEMETRY_AUTH_ENABLED` is off/unset** (fail-open design; current literal reads True — see §3.3 dispute):
  - `POST /safety-net` — `enable`/`disable`/`resume`/`kill_release` (re-arms the latched kill — `:493-510`).
  - `POST /config` — sessions, `direction_mode` (both/long_only/short_only), EOD flatten, horizons (`:559-593`).
  - `POST /flatten` — sets `_FLATTEN_REQUESTED` → main cycle flattens **all positions** (`:1426-1460`). **Missing from `POST_ROUTES`** (`:78`) though present in `POST_HANDLERS` (`:83-87`) — works but invisible to allow-list audits.
- `GET /auth` **returns the bearer token** to CORS-allowed browser origins (`:700-725`); the allow-list auto-includes the live tunnel URL from `runtime/tunnel_url.txt` (`:657-668`).
- Design note in code: "tunnel is the perimeter" (`:704`) — i.e., public exposure of the tunnel = public exposure of all of the above.

---

## 7. Known bug history (`FIXES.md`, 264 lines, append-only, guard-enforced)

| Class | Pattern | Entries |
|---|---|---|
| **E** — unwired capability assumed working | dead/dormant features treated as live | **6** (most recurrent; latest 2026-09-08-07): regime fallback throttle, StrategyGenome zero callers, portfolio-risk stub, `_watched` lazy init, penny-score bar, direction default |
| B — single-writer violations | two modules writing one store | 4: trade double-counting, re-adopt mid-flatten, `cancelAllOrders` clobbering flatten orders |
| A — path arithmetic / env assumptions | wrong `parents[N]`, CWD-relative paths | 3 |
| D — silent exception swallowing | systemic failure logged away | 3: incl. blocking seeding → skipped protection |
| C — typed-container truthiness | `arr or fallback` on numpy → ValueError / silent starvation | 1 — but has a permanent static watchdog in the enforcer test |

Most instructive:
1. **FIX-2026-09-07-05 (C+D)** — `arr or fallback` on numpy arrays in `juli_feed` killed **every** live entry evaluation (`thinks=+0`, `decisions=0`); invisible to list-shaped unit tests. Fix: explicit `is None`/`len()` checks + snapshot-shaped fixtures. Lesson: fixtures must be numpy-shaped, not convenient lists.
2. **FIX-2026-09-07-07 (E+B)** — portfolio risk manager was a stub (`update(net_liq, {})` — positions hardcoded empty); exposure/concentration logic could never fire. Full port + wiring + `/risk` endpoint.
3. **FIX-2026-09-07-02 (B)** — every real trade double-counted in learning (orchestrator + Reflector both writing). Fix: documented single-writer ownership + exact-delta smoke asserts.
Standing defenses: `scripts/smoke_live.py` (23 checks), `PipelineMonitor`, journal-enforcer test, deliberately-dormant monitor list.

---

## 8. Top 10 concrete problems (ranked by safety/correctness impact, then edge impact)

**1. Safety net off by default; the −$500 kill is skipped while disabled; the latch doesn't survive restart.**
`brain/policy/safety.py:29-43` (defaults), `:78-84` (disabled early-return precedes the kill check), `brain/consolidation.py:73` (fresh disabled producer per engine). A −$500 day with safety off never latches; a restart wipes even a latched kill. *Fix:* evaluate the kill condition before the disabled early-return; persist `{enabled,halted,latched}` to `runtime/safety_state.json` and restore on boot; default `enabled=True`. — **Safety: critical.**

**2. Kill cancels orders but never flattens positions.**
`safety.py:131-164` → `executor.cancel_all()` (`ib_adapter.py:95-104`). After a kill day the bot holds its losers into the next session with entries vetoed but nothing forcing a flat book. *Fix:* second kill hook → `executor.close_all_positions(...)` (market) before/after cancel-all; journal it. — **Safety: high.**

**3. Fixed 0.7 score damping with the neuromorphic blend disabled.**
`brain/orchestrator.py:1285-1296`; `NEURO_BLEND_ENABLED=False` (`immune.py:148`). Every cortex score is multiplied by 0.7 for zero neuro contribution: the penny bar (0.85) becomes nearly unreachable, and the declared threshold band [0.45, 0.70] no longer means what it says in cortex units (the quintile self-calibration absorbs the distortion). *Fix:* when the gate is off, `blended = base.score + cal_adj`. — **Correctness: high (warps every entry decision).**

**4. Sleep replay rewards losers.**
`brain/neurons/sleep.py:167-171` sets `reward = 1.0 if weight >= 1.0 else 0.5`, but the scheduler tags losers weight 3.0 and winners 1.0 (`brain/sleep_scheduler.py:51-73`) — both satisfy `>= 1.0`, so losers get the positive reward the docstring says they shouldn't. *Fix:* thread explicit `(pattern, won)` polarity into `_replay_pattern`. — **Edge: high (actively teaches the wrong lesson).**

**5. Meta DNN gatekeeper fails open.**
`MetaDNN.infer` returns `(True, META_WIN_THRESHOLD, scale)` on defective/unbuilt model (`brain/meta_label_dnn.py:175-210`); `MetaLabelModel._fallback_gate` always admits (`brain/meta_label.py:186-199`); `gate()` falls back on any DNN exception (`:142-156`). A missing/corrupt `juli_meta_dnn.json` silently disables the entry veto. *Fix:* fail closed (or startup warning) when `live_snapshot()["bypassed"]`. — **Safety/correctness: high.**

**6. The EV and confidence "gates" never refuse; no fee drag in EV math.**
`RiskEngine.evaluate()` discards `should_enter`, only scales size 0.5–1.0 ("NEVER refuses… only scales" — `brain/risk.py:190-205`); no hard confidence gate exists; `ev = (p·r − (1−p)) · direction_mod` subtracts no costs (`brain/ev_gate.py:75`), and `round_trip_cost_fraction()` is wired only via the default-off `ENTRY_COST_AVERSE_GATE` (`immune.py:179`). The contract's central promise — no trade without positive expectancy — is advisory-only in code. *Fix:* make `should_enter=False` refuse (or document the downgrade); subtract fee+slippage in the refuse math. — **Edge: high.**

**7. Entry parent is a limit at mid, not a market order; governor cooldown stamped at placement.**
`ib_executor.py:134`; `ib_cycle.py:716`; stale-parent sweep at `:262-298`. Entries can rest unfilled in a fast move, then get swept after 60 s; re-entry needs a fresh ENTER **plus** the 60 s cooldown that was stamped even though nothing filled — the brain believes it's "in" while flat. *Fix:* marketable limit offset (or MKT parent + STP/LMT children); stamp cooldown on confirmed fill (`_confirm_fill`). — **Correctness: medium-high.**

**8. ATR trailing-stop logic is dead code.**
`IBExecutor.monitor_orders` / `_handle_parent` (`ib_executor.py:419-494`) implement trailing, but no call site exists in `ib_cycle.py` or `ib_adapter.py`; `exits.py` docstrings still claim it runs alongside mechanical exits. *Fix:* call `monitor_orders` per cycle (with skip-if-closing) or delete the code and its claims. — **Correctness: medium.**

**9. Validation-harness drift: Phase-4 criteria unenforced; WFA trial-count mismatch.**
`scripts/paper_run.py:176-184` reuses `wfa.verdicts` verbatim, ignoring the pre-locked `deflated > 0.05 AND PBO < 0.05` (`task_plan.md` Phase 4); `_count_trials` (`wfa.py:332-338`) counts cells over all tickers (126) while pooling only admissible ones (4); shipped WFA never calls purge/embargo (`wfa.py:117-157`). The harness cannot currently certify edge the way the protocol requires. *Fix:* implement the P5 check in `run_paper`; count trials within the admissible set; document the conservatism. — **Edge-measurement integrity: medium.**

**10. Telemetry mutation surface is one misconfiguration from unauthenticated remote control.**
Fail-open `_authorized()` (`telemetry.py:695-699`); `GET /auth` serves the bearer token to CORS origins including the auto-added tunnel URL (`:657-672,700-725`); `POST /flatten` (real orders) missing from `POST_ROUTES` (`:78` vs `:83-87`); loopback bind + cloudflared tunnel = public perimeter. Current `TELEMETRY_AUTH_ENABLED=True` literal (`immune.py:178`, two reads) mitigates today, but the design fails open. *Fix:* default auth on and verify at runtime; never serve the token to browser origins; single source of truth for POST routes. — **Security/safety: medium (latent critical).**

*Honorable mentions:* 30 s stale authorization window on the entry hot path (`consolidation.py:34-35` → `orchestrator.py:416-458`); documented 60-min consecutive-loss pause unimplemented — halt is indefinite (`immune.py:63` never read); HALIM threshold recommendations crash on the read-only `Dynamics.threshold` property (`halim_recommendations.py:113-118` vs `brain/dynamics.py:79-82`); held tickers consume the 2-per-cycle entry budget (`juli.py:76-106` → `ib_cycle.py:702-706` "SKIP open"); `LearnedExitPolicy.get_policy()` write-only; AFML "off-by-default" comments contradict `True` values (`immune.py:281,293,302`); dead `ENTRY_THRESHOLD`/`DECISION_THRESHOLD` constants invite misreading.

---

## Appendix — part files and key entry points
- `part1_live_path.md` — full tick→fill trace, 23-gate table, 5 candidate problems.
- `part2_organs.md` — organ table, flag inventory, LIVE_CHECK corrections, 5 candidate problems.
- `part3_safety.md` — limit table, SafetyProducer semantics, threshold analysis, 12-item bypass inventory, 5 candidate problems.
- `part4_validation.md` — WFA mechanics, reproduce command, 33-endpoint inventory, FIXES.md analysis, 51-script inventory, 5 candidate problems.
- Live entry: `src/hanoon_prime/cli.py:85` → `ib_adapter.py:160` (`_cycle`) → `ib_cycle.py:561` (`_run_brain_cycle`) → `juli.py:44` (`tick`) → `brain/orchestrator.py:263` (`decide_entry`) → `ib_cycle.py:645` (`_execute_entries`) → `ib_executor.py:62` (`place_bracket`).
- Validation: `python scripts/wfa_report.py --data-dir data/fixtures --tickers ALL --output reports/wfa_abs_new.json` (exit 0 ⇔ universe PASS).
