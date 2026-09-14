# AWAKENED_BRAIN.md — Living Design & Implementation Contract

> **This document is the single source of truth for Juli's biological learning system.
> Every implementation session MUST update this document before committing.**

**Created:** 2026-09-14
**Status:** Phase A done, Phase B done, Phase C done, Phase D done, Phase E done — commit hash pending (recorded on commit)
**Last updated:** 2026-09-14 — Phase E implementation complete

---

## 1. Current State

### 1.1 Architecture Summary

Juli's brain is a `NeuromorphicBrain` (1272-line orchestrator) running two speeds:

- **Fast path (S1):** per-tick cortex scoring → Nash veto → SNN neuro-score → regime/episodic/halim/cross-asset modifiers → threshold gate → entry decision.
- **Slow path (S2):** 30s `ConsolidationEngine` cycle → regime detection, HALIM modifier, Thinker 5-pillar fusion, evidence learning, policy/safety pulse, pillar update, state persistence.

**~29,000 lines across 185 modules** under `src/hanoon_prime/`. The live core is ~50 wired modules; ~20 legacy "bio-named" modules are dead scaffolding.

### 1.2 Learning Loop (Closed)

```
Position closes
  → _ib_sync.get_ib_pnl (FRACTION, dollars normalized)
  → ib_cycle._reflect_closed → brain.on_trade_close
      ├─ Ironclade gate (source ∈ {real_trade, ib_paper, ib_fill, reconciled_exit})
      ├─ dynamics.adapt_threshold (pred_error_ema + losing conf bins)
      ├─ episodic.add (k-NN pattern memory)
      ├─ nash.record_outcome (pattern veto)
      ├─ neuromorphic.learn_from_outcome (SNN STDP)
      ├─ strategy organs: meta_label, horizon_bandit, regime_weights, learned_exit
      └─ _learn_from_real:
           Reflector._adapt_weights (loss-aversion 2.4:1, LR 0.02, decay 0.999)
           → regime weights hot-swap into cortex
           → realized.add_outcome + add_confidence_outcome
           → exits.adapt_from_realized
           → advisor.record_outcome
  → Next tick reads all of it back
```

### 1.3 Outcome Stores

| File | Writer | Content |
|---|---|---|
| `runtime/journal_live.jsonl` | Journal | Append-only trade log |
| `runtime/state.json` | ConsolidationEngine | brain_state (regime/halim/thinker/pillar/policy), realized snapshot |
| `runtime/juli_state.json` | JuliMemory | weights, episodes, pred_error_ema, threshold, lessons |
| `runtime/juli_realized.json` | RealizedStats | band_wins/losses, conf_wins/losses, rr_samples |
| `runtime/juli_meta_label.json` | MetaLabelModel | SGD weights, Brier score |
| `runtime/juli_horizon_bandit.json` | HorizonBandit | Beta posteriors per regime×horizon |
| `runtime/juli_regime_weights.json` | RegimeWeights | Per-regime weight vectors |

### 1.4 What Works Well

- Closed-loop single-writer discipline (no data races between S1/S2)
- Loss-aversion asymmetry (PENALTY_SCALE=1.2 vs REWARD_SCALE=0.5) — matches prospect theory
- Multi-resolution outcome tracking (binary, continuous, confidence-bin, score-band, regime, horizon, exit-trigger)
- All modulators mathematically bounded (Thinker ±0.06, Nash ±0.03, Episodic ±0.10, Halim ±0.03, Emotion ±0.05)
- Ironclade gate blocks synthetic/paper data from the realized learning loop
- Pillar awareness (edge-vs-break-even, upright/tipping/fallen/warming)
- Complementary timescales: online (per-trade), batch (30s), sleep-ready (SNN replay exists but unwired), slow (daily/weekly supervisor)

---

## 2. Biological Gaps

### 2.1 No Multi-Timescale Dopamine-Like RPE

**What biology does:** Dopamine neurons encode prediction error (expected vs received). Recent work (Masset et al. 2025, Nature 642:682) shows individual DA neurons carry different discount factors — some encode immediate surprise, others track long-horizon value drift. Garud & Morris (2026) unify phasic bursts, tonic baseline, and ramping under continuous TD learning. Nature Neuroscience (2024) shows RPE is a special case of policy-information-gain.

**What we have:** `pred_error_ema` — a single scalar EMA (α=0.10) used only for threshold adaptation.

**Gap:** No phasic component (per-trade surprise → transient LR boost), no tonic component (running expectation drift → "mood"), no multi-timescale representation. The brain can't distinguish "this trade surprised me" from "my last 50 trades have been worse than expected."

### 2.2 No Interoception / Allostatic Setpoint

**What biology does:** Allostasis (Sterling 2012) — the brain doesn't have fixed thresholds; it dynamically adjusts expectations based on predicted future needs. Keramati & Gutkin (2014) proved reward-seeking ≡ homeostatic regulation when primary rewards are need-fulfillment. Pezzulo et al. (2015) formalized this in active inference.

**What we have:** Fixed `PILLAR_EDGE_FALL = -0.10`. Fixed confidence thresholds.

**Gap:** No dynamic reference point for "what should my edge be right now?" No regime-conditioned setpoint. No dyshomeostasis detector (sustained deviation requiring structural change vs normal fluctuation).

### 2.3 No Somatic Markers

**What biology does:** Damasio (1994, 1996) — somatic markers are bodily-state signals that rapidly prune the decision search space before conscious reasoning. VMPFC patients can reason logically but make terrible decisions because they lack these fast bias signals.

**What we have:** Emotion pillar (fear/greed from streak count), Nash veto (pattern-matcher). Neither produces a holistic, situation-specific "gut feel" from the entire recent outcome trajectory.

**Gap:** No rapid pre-scoring bias generator that combines win/loss history, confidence calibration, regime, and pattern quality into a single bounded signal.

### 2.4 No Extinction / Context-Tagged Memory

**What biology does:** Extinction learning (Bouton 2004) — the brain learns that a previously rewarded action is no longer rewarding in this context, without erasing the original memory. Context-dependent retrieval allows reactivation when returning to the original context.

**What we have:** Weight decay (×0.999) — slow forgetting without context tags. No distinction between "never learned" and "learned but now obsolete."

**Gap:** No inhibition weights, no context tagging (regime+confidence+horizon), no reactivation on regime return.

### 2.5 Sleep Replay Unwired

**What biology does:** Complementary Learning Systems (CLS, McClelland 1995, Kumaran et al. 2016) — hippocampal replay during sleep interleaves new experiences with old, gradually integrating into neocortical representations. Singh et al. (2022) showed bidirectional hippocampus-neocortex dialogue during SWS.

**What we have:** `SleepReplayEngine` and `AttractorMemory` exist and are wired to `orchestrator.sleep_replay()`. But nothing in the live loop auto-invokes them. Only called from tests.

**Gap:** No automatic scheduler, no goal-dependent replay weighting (losing trades replayed more), no interleaving.

### 2.6 No Metacognitive Confidence-of-Confidence

**What biology does:** Anterior prefrontal cortex (BA10) represents second-order confidence — "how reliable is my current confidence estimate?" Fleming & Dolan (2012) showed this enables attention allocation and threshold adjustment.

**What we have:** `cognitive/metacognition.py` bounded ±0.04, reads only its own confidence. No second-order monitoring.

**Gap:** No calibration reliability tracking, no surprise detection (situation matches no known pattern), no curiosity drive (explore when safe, retreat when unsafe).

---

## 3. Implementation Roadmap

### Phase A: Multi-Timescale RPE

**File:** `src/hanoon_prime/brain/rpe.py` (new, ~190 lines)
**Status:** `done` — implemented, wired, tested (17 tests, all green)
**Priority:** 1st — touches every learning organ
**Effort:** ~300 lines total (module + wiring + tests)
**Timeline:** Week 1-2

**Acceptance Criteria:**
- [x] `MultiTimescaleRPE` class with three channels: phasic (τ≈3), tonic (τ≈50), meta (regime-specific)
- [x] Updated on every real close via `brain.on_trade_close`
- [x] `Reflector._adapt_weights` uses phasic RPE to modulate per-trade LR
- [x] `consolidation._update_pillar` reads tonic RPE as mood signal
- [x] `cognitive/emotion.py` uses tonic RPE for fear/greed instead of raw streak count
- [x] Published to shared state as `rpe_phasic`, `rpe_tonic`, `rpe_meta`
- [x] Tests: `tests/test_rpe.py` — phasic/tonic separation, regime conditioning, persistence
- [x] Pre-commit hooks pass

**Biological Basis:** Masset et al. 2025 (Nature 642:682), Garud & Morris 2026, Nature Neuroscience 2024 policy-IG

**Design:**

```python
class MultiTimescaleRPE:
    """Three dopamine-inspired prediction error channels.

    phasic_rpe:  per-trade surprise (fast α≈0.33, ~3 trade memory)
    tonic_rpe:   running expectation drift (slow α≈0.02, ~50 trade memory)
    meta_rpe:    regime-conditioned expectation per regime key
    """

    def update(self, pred_win_prob: float, won: bool, regime: str) -> dict:
        # pred_win_prob seeds the phasic/tonic channels on the FIRST trade
        # only (the brain's own calibrated prior, from score_to_win_prob);
        # afterwards V moves toward outcomes: V += α * (r - V).
        ...

    def phasic(self) -> float: ...
    def tonic(self) -> float: ...

    def save(self) -> dict: ...
    def load(self, data: dict) -> None: ...
```

**Implementation Steps:**
1. ✅ Write `brain/rpe.py` with `MultiTimescaleRPE`
2. ✅ Wire update in `orchestrator.on_trade_close` after `dynamics.adapt_threshold` (extracted to `_update_rpe`; `_adapt_threshold` helper keeps on_trade_close ≤40 lines for the R3 contract test)
3. ✅ Wire phasic into `reflection.py:Reflector._adapt_weights` as LR modulator
4. ✅ Wire tonic into `pillar_awareness.py` and `consolidation._update_pillar`
5. ✅ Wire tonic into `cognitive/emotion.py` replacing streak-based fear/greed
6. ✅ Publish all three to `shared_state.py` BrainState (+ `rpe_surprise` used by reflector LR)
7. ✅ Write `tests/test_rpe.py` (17 tests)
8. ⏳ Update this doc with commit hash (done on commit)

**Design Decisions:**
- **Value channels are self-referential** (`V += α·(r−V)`, `r = 1 win / 0 loss`), NOT driven by `score_to_win_prob` per trade — the brain's own prediction is only used to **seed** the fast/slow channels on the first trade (calibrated prior). Chosen over using prediction every trade because (a) surprise must fade after ~3 consistent trades (a curiosity signal), (b) signed score-vs-outcome calibration already lives in `JuliMemory.pred_error_ema` — duplicating it would double-count the same signal.
- **RPE errors are computed BEFORE the V update** (error/response precedes assimilation), then V moves toward `r`.
- **`lr_modulator(surprise)` maps |phasic| to a bounded LR multiplier** `clamp(1 + 1.5·s, 0.5, 2.0)`; surprise 0 → 1.0 (identical to today's LR, zero-behavior-change safety), full surprise → 2.0 (learn harder), floor 0.5 is a hard safety only.
- **Tonic RPE blended 50/50 with the win-rate streak** in `EmotionState.confidence_mod` (streak is retained so affect isn't purely dopamine); fear remains loss-magnitude-driven.
- **Pillar geometry NOT tilted by mood** — RPE channels are surfaced *alongside* the pillar (`rpe_phasic/tonic/meta` keys) so the webapp/Inside Man see feel next to geometry, but `state/upright/tilt/edge` stay purely edge-vs-break-even (decision semantics unchanged; this preserves the just-fixed pillar contract).
- **Persistence** to `runtime/juli_rpe.json` (atomic tmp+replace, `HANOO_RPE_FILE` override hermetic in tests). Only value estimates + count persist; the last-error fields are in-memory-only by design.
- **R3 contract**: `R3 no-function>40-lines` pytest test does NOT skip orchestrator.py, so `on_trade_close`/`_learn_from_real` were refactored into helpers (`_adapt_threshold`, `_update_rpe`, `_reflect_close`) to stay ≤40 lines.

---

### Phase B: Homeostatic Setpoint + Interoception

**File:** `src/hanoon_prime/brain/allostasis.py` (new, 133 lines)
**Status:** `done` — implemented, wired, tested (21 tests, all green)
**Priority:** 2nd — pillar becomes dynamic
**Effort:** ~400 lines total (module + wiring + tests)
**Timeline:** Week 2-3

**Acceptance Criteria:**
- [x] `AllostaticController` class with dynamic setpoints per regime
- [x] Setpoints updated from recent outcome statistics (slow EMA α=0.05, gated until `ALLOS_MIN_TRADES=10`)
- [x] `dyshomeostasis` detection: sustained deviation from setpoint (`ALLOS_VIOLATION_MIN=5`, resets on good update)
- [x] `pillar_awareness.py` compares edge to dynamic setpoint (fallen line = `max(PILLAR_EDGE_FALL, setpoint)` when setpoint < 0; never looser than structural -0.10)
- [x] `dynamics.adapt_threshold` receives `dyshomeostatic` → aggressive tightening at `ALLOS_TIGHTEN_STEP=0.02`
- [x] HALIM evidence includes setpoint deviation (via `pillar_setpoint` / `pillar_deviation` keys in `pillar_fields`)
- [x] Webapp pillar panel shows setpoint and deviation *(hanoon-dash `a837644` — PillarBalancePanel setpoint line + dyshomeostasis chip; inspection `pil__setpoint` keys on the `pillar_balance` evidence via commit `689f9b6`)*
- [x] Tests: `tests/test_allostasis.py` — setpoint tracking, per-regime independence, dyshomeostasis detection, persistence
- [x] Pre-commit hooks pass

**Biological Basis:** Sterling 2012 (allostasis), Keramati & Gutkin 2014 (homeostatic RL), Pezzulo et al. 2015 (active inference + interoception)

**Design:**

```python
class AllostaticController:
    """Per-regime expected-edge setpoints with dyshomeostasis detection.

    Each regime keeps its own learned "expected edge" (slow EMA of the
    realized edge, gated until ALLOS_MIN_TRADES closes). The setpoint
    must never loosen the pillar: the structural PILLAR_EDGE_FALL floor
    always applies. Sustained deviation beyond ALLOS_MARGIN for
    ALLOS_VIOLATION_MIN learned updates trips dyshomeostasis.
    """

    def update(self, record: dict | None, regime: str) -> dict: ...
    def setpoint(self, regime: str) -> dict: ...
    def is_dyshomeostatic(self, regime: str) -> bool: ...
    def snapshot(self) -> dict: ...
```

**Implementation Steps:**
1. ✅ Write `brain/allostasis.py` with `AllostaticController` (133 lines, all funcs ≤40; persistence to `runtime/juli_allostasis.json`, atomic tmp+replace, `HANOO_ALLOSTASIS_FILE` hermetic override)
2. ✅ Wire into `consolidation._update_pillar` — controller updates each cycle from the realized win/loss record, publishes `allostatic` to shared state
3. ✅ Modify `pillar_awareness.py` — `compute_pillar_awareness(record, rpe=None, setpoint_edge=None)`; `_apply_state` tightens the fallen line via the learned norm; `_decorate` surfaces `setpoint`/`setpoint_deviation`/`below_setpoint` (replaces Phase A `_with_rpe`, handling rpe + setpoint keys)
4. ✅ Wire dyshomeostasis into `dynamics.adapt_threshold` as `dyshomeostatic` aggression trigger
5. ✅ `orchestrator._adapt_threshold` reads `allostatic.dyshomeostatic` from shared state (isinstance-guarded — avoids the Class C `or {}` truthiness hazard the watchdog test bans)
6. ⏳ Extend webapp PillarBalancePanel with setpoint line *(hanoon-dash, after backend commit)*
7. ✅ Write `tests/test_allostasis.py` (20 tests)
8. ⏳ Update this doc with commit hash (done on commit)

**Design Decisions:**
- **Setpoint EMA (α=0.05) gated until `ALLOS_MIN_TRADES=10`** — the first updates on a fresh controller never move the norm; the norm starts at break-even 0.0 and only learns after the regime is old enough. The record's cumulative `trades` count drives the gate (same count the pillar uses).
- **Fallen line tightens but never loosens** — when a regime learns a negative norm, the pillar's fallen threshold becomes `max(PILLAR_EDGE_FALL, setpoint)`; a positive norm keeps the structural -0.10 floor. The pillar can only get *stricter* from allostasis, never laxer.
- **`allostatic` published as a dict, walled off from `pillar`** — `dyshomeostatic`, `setpoint`, `deviation`, `violations`, `trades`, `regime` live in shared state under `allostatic`; the pillar carries its own `setpoint`/`setpoint_deviation` keys for HALIM + webapp. Decision semantics of `upright/tipping/fallen` unchanged for positive norms.
- **Emotion blends, geometry doesn't** — the allostatic setpoint only recalibrates the *fallen threshold*; `tilt`, `state`, and `edge` remain realized-data-driven. Mood stayed out of geometry (consistent with Phase A).
- **Dyshomeostasis is transient by design** — deviation from an *unadapted* norm accumulates violations (alarm); as the EMA converges onto a persistent negative edge the deviation shrinks below `ALLOS_MARGIN` and the alarm clears — the negative edge has become the *new normal* (Sterling: the body anticipates a recurring threat). Durable discipline then comes from the tightened fallen line, not the alarm.
- **Class C hazard honored** — `orchestrator._adapt_threshold` uses an `isinstance(obj, dict)` guard instead of `get(...) or {}` because `test_class_c_watchdog_no_array_truthiness` bans the pattern (an empty-list FQ would silently mask a shape bug).

---

### Phase C: Somatic Markers

**File:** `src/hanoon_prime/brain/somatic.py` (new, ~130 lines)
**Status:** `done` — implemented, wired, tested (18 tests, all green)
**Priority:** 3rd — rapid pre-scoring bias
**Effort:** ~300 lines total (module + wiring + tests)
**Timeline:** Week 3-4

**Acceptance Criteria:**
- [x] `SomaticMarkerGenerator` class producing per-decision bias from full outcome trajectory
- [x] Inputs: pillar tilt, RPE channels (phasic/tonic), allostatic alarm (dyshomeostasis)
- [x] Output: single bounded bias (`±SOMATIC_MAX = ±0.10`) applied to raw score
- [x] Applied in `_score_pipeline` as `somatic` bias; precision-dampens all non-regime modifiers when negative (scaled by `precision_weight(marker)`)
- [x] Published to shared state as `somatic_marker` / `somatic_precision`
- [x] Tests: `tests/test_somatic.py` — lean, RPE mood, allostatic alarm, precision dampening, bounds, snapshot
- [x] Pre-commit hooks pass

**Biological Basis:** Damasio 1994, 1996 (somatic marker hypothesis), Bechara et al. 1996 (Iowa Gambling Task)

**Design:**

```python
class SomaticMarkerGenerator:
    """Gut-feel bias from the entire recent outcome trajectory.

    Combines pillar body-state, RPE mood, and allostatic alarm into a
    single bounded bias signal added to the raw score BEFORE stabilization
    — Damasio's vmPFC markers that prune the decision space before the
    deliberative system kicks in. A negative marker also precision-dampens
    all the learned modifiers (HALIM, episodic, Nash, etc.), scaling them
    toward zero like a gut-level "pay extra attention, reduce noise".
    """

    def generate(self, pillar: dict, rpe: dict, allostatic: dict) -> float: ...
    def precision_weight(self, marker: float) -> float: ...
    def snapshot(self) -> dict[str, float]: ...
```

**Implementation Steps:**
1. ✅ Write `brain/somatic.py` with `SomaticMarkerGenerator` (~130 lines, all funcs ≤40)
2. ✅ Wire into `orchestrator._score_pipeline` via `_somatic_context()` helper; somatic bias added to `raw`, precision scales all non-regime modifiers in `_compute_mods()`
3. ✅ Publish `somatic_marker` and `somatic_precision` to `shared_state`
4. ✅ Add `somatic_marker`, `somatic_precision`, `allostatic` defaults to `brain/shared_state.py`
5. ✅ Write `tests/test_somatic.py` (18 tests)
6. ⏳ Update this doc with commit hash (done on commit)

**Design Decisions:**
- **Pillar tilt drives a one-way negative pressure** — a leaning pillar always hurts the marker (never helps). Tilt ∈ [0,1] maps linearly to `[-SOMATIC_TILT_GAIN, 0]`. The brain's body language (pillar) is a loss signal only; wins don't produce a euphoric somatic marker (the reflex is asymmetric, matching prospect theory).
- **RPE mood is signed** — tonic and phasic channels map along their sign (negative = worse than expected → lower marker; positive = better → lift). This gives the somatic marker a *directional* mood signal, not just a "danger" flag.
- **Dyshomeostasis is a one-shot penalty** — when the alarm fires, the somatic marker takes an explicit `SOMATIC_DYS_PENALTY` hit, independent of the other signals. This prevents a high tonic RPE (which could temporarily mask the allostatic alarm) from fully canceling the stress signal.
- **Precision dampening is linear and floored** — `precision = 1 + 2*marker` for negative markers, floored at 0.6. The modifiers can shrink to 60% of their voice, never fully silenced (the deliberative cortex must still weigh them, even when the gut says be cautious).
- **No persistence** — the somatic marker is assembled fresh each tick from the current body state; it has no history. The marker's *temporal integration* happens naturally via the pillar tilt history and RPE tonic channel (both of which are persistent).
- **`_compute_mods` extraction** — the non-regime modifier sum was pulled into a private helper to keep `_score_pipeline` at exactly 40 lines (R3). This also makes the modifier inventory explicit: `halim + episodic + nash_op + news_bias + cross − advisor_delta + thinker_mod`.

---

### Phase D: Extinction + Context-Tagged Memory

**File:** `src/hanoon_prime/brain/extinction.py` (new, ~189 lines) + context tags in `brain/episodic.py`
**Status:** `done` — implemented, wired, tested (27 tests, all green)
**Priority:** 4th — regime transition handling
**Effort:** ~300 lines total
**Timeline:** Week 4-5

**Acceptance Criteria:**
- [x] Context tags on episodic memories: (regime, confidence_bin, horizon) — `episodic.add(alpha, outcome, context)` / `modifier(alpha, context)`; neighbors filtered to a context when it has ≥`EPISODIC_MIN_SAMPLES` tags
- [x] Inhibition weights separate from excitatory weights — `ExtinctionTracker._cells` are independent of the k-NN buffer
- [x] At retrieval: `modifier = excitation − inhibition` (both context-gated) — `_episodic_bias` in orchestrator
- [x] On regime transition: reactivate suppressed patterns if regime matches — `_episodic_bias` calls `reactivate(canon)` when `canon != _extinction_last_regime`
- [x] Extinction signal when pattern performance degrades within context — perf EWMA < `EXTINCT_LOSS_BELOW` grows the inhibition (step `EXTINCT_STEP`, cap `EXTINCT_MAX`, healthy shares decay `EXTINCT_DECAY`)
- [x] Tests: `tests/test_extinction.py` — growth, decay, caps, context gating, neighbor overlap, renewal, persistence, corrupt-file recovery, episodic tags
- [x] Pre-commit hooks pass

**Biological Basis:** Bouton 2004 (context-dependent extinction), Myers & Davis 2007 (extinction retrieval / renewal)

**Design:**

```python
class ExtinctionTracker:
    """Per-context inhibition weights over episodic pattern signatures.

    Every patterned alpha vector gets signature-tagged with the context it
    was learned in (regime | conf-bucket | horizon). A pattern whose recent
    outcomes inside a context degrade grows an inhibition weight; at
    retrieval the net modifier is excitation − inhibition (both
    context-gated). Re-entering a regime renews its suppressed traces
    (Bouton's renewal — the original memory returns with its context).
    """

    def record(self, alpha, outcome, regime, conf, horizon) -> float: ...
    def inhibition(self, alpha, regime, conf, horizon) -> float: ...
    def reactivate(self, regime) -> int: ...
    def save(self) -> dict: ...
    def load(self, data: dict) -> None: ...
```

**Implementation Steps:**
1. ✅ Write `brain/extinction.py` (~189 lines; `ExtinctionTracker` + `conf_bin_label`/`context_key`/`_signature` helpers — all functions ≤40 lines)
2. ✅ Modify `brain/episodic.py`: `add`/`predict`/`modifier` accept an optional context tag; retrieval restricted to a context when it has enough samples, else falls back to full buffer (backward compatible — all callers unchanged)
3. ✅ Wire extinction + context into `orchestrator.on_trade_close` — context-tagged `episodic.add` + `_extinction.record` (after the horizon/last-conf context is resolved; only real IB fills via the ironclade gate)
4. ✅ Wire reactivation into regime transition detection — `_episodic_bias` renews on canon change, then returns `clip(excitation − inhibition)` and mirrors it into shared state as `episodic_bias`
5. ✅ Write `tests/test_extinction.py` (27 tests)
6. ⏳ Update this doc with commit hash (done on commit)

**Design Decisions:**
- **Signature = rounded alpha bins, neighbors by shared-dim overlap** — each vector is rounded to 1 decimal across `EPISODIC_KEYS` (11 dims); a query's inhibition is the max over same-context cells sharing ≥`EXTINCT_OVERLAP=2` dimension bins, scaled by `shared/len(tags)`. Coarse (not nearest-in-R²), robust to intra-situation jitter, and cheap — no extra numpy.
- **Inhibition is per-context, excitatory weights are global** — the k-NN buffer stays context-free; only the extinction overlay (and query-time context filtering) is context-scoped. Excitation can't be destroyed, only gated — matching Bouton: extinction suppresses, never erases.
- **Performance is a slow EWMA, not a loss streak** — `perf = 0.7·perf + 0.3·outcome`. One or two losses don't flip a reliable pattern; sustained degradation does. Decay is continuous (`EXTINCT_DECAY` per healthy share), so a pattern that recovers re-earns its voice without a manual reset.
- **`EXTINCT_MAX = EPISODIC_MOD_BOUND`** — the inhibition can fully cancel a pattern's positive excitation but never drive the modifier negative; the episodic channel stays a dampener (R1's "bounded, advisory" contract).
- **Renewal is regime-scoped** — only cells tagged in the entering regime get inhibition cleared, leaving other contexts' learning intact. The `ctx.startswith(f"{regime}|")` match keeps scoping cheap.
- **`_compute_mods` reuse** — Phase C's modifier-sum helper is unchanged; the context-gated excitation flows in as `episodic` and the net of inhibition is already folded into the `eb` the pipeline sees (excitation − inhibition).
6. Update this doc with commit hash

**Design Decisions:** (none yet)

---

### Phase E: Sleep Replay Scheduler

**File:** `src/hanoon_prime/brain/sleep_scheduler.py` (new, 81 lines) + modifications to `brain/consolidation.py` and `brain/neurons/sleep.py`
**Status:** `done` — commit hash recorded on commit
**Priority:** 5th — engine exists, needs scheduling
**Effort:** ~200 lines total
**Timeline:** Week 5-6

**Acceptance Criteria:**
- [x] Auto-trigger `sleep_replay` after 30min inactivity or session close
- [x] Replay losing trades 3× more often than winners (prevent overconfidence)
- [x] Interleave with random historical trades (CLS interleaved learning)
- [x] After replay: update regime_weights and episodic from consolidated patterns
- [x] Tests: `tests/test_sleep_scheduler.py` — 22 tests
- [x] Pre-commit hooks pass

**Biological Basis:** McClelland 1995 (CLS theory), Kumaran et al. 2016 (CLS updated), Singh et al. 2022 (bidirectional replay)

**Design:**

```python
class SleepScheduler:
    """Triggers offline replay during inactivity periods.

    Criteria: no trades for SLEEP_THRESHOLD_SEC (default 1800) or
    session close. Replay weights losing trades 3× higher.
    """

    def check(self, last_trade_time: float, session_close: bool, now=None) -> bool: ...
    def replay_weights(self, recent_trades: list, historical=None) -> list: ...
```

**Implementation Steps:**
1. ✅ Write `brain/sleep_scheduler.py`
2. ✅ Wire into `consolidation._cycle` — check trigger each cycle
3. ✅ Modify `sleep_replay` call to use weighted replay list
4. ✅ Write `tests/test_sleep_scheduler.py` (22 tests)
5. ⏳ Update this doc with commit hash

**Design Decisions:**
1. **`SleepScheduler.check` takes an explicit `now` param** — tests inject synthetic timestamps instead of monkeypatching `time.time`; `_last_trigger` sentinel is `-inf` so the first trigger never hits the cooldown guard.
2. **Cooldown only after first trigger** — one replay per `SLEEP_COOLDOWN_SEC` (3600s) of downtime; session-close replays share the same guard so a close immediately after an idle replay doesn't double-run.
3. **`replay_weights(recent, historical)` returns `[(pattern, drive)]`** — losers `SLEEP_LOSS_WEIGHT=3.0`, winners `SLEEP_WIN_WEIGHT=1.0` (the old engine inverted this at 2.0/0.5 — winners were over-replayed, feeding overconfidence); up to `SLEEP_INTERLEAVE_MAX=6` random historical traces interleave per replay.
4. **Engine override path** — `SleepReplayEngine.select_patterns(replay_list=None)` and `run_cycle(..., replay_list=None)` accept a pre-weighted list; default attractor weighting uses `LOSS_BIAS=3.0` when `losses >= wins` else `WIN_BIAS=1.0`.
5. **Auto-replay runs a short `duration_sec=5.0` cycle** inside the 30s S2 loop so consolidation is not blocked; the orchestrator's explicit offline `sleep_replay()` still uses the full 60s default.
6. **`_sleep_patterns()` builds `{"alpha_i": v}` centers** from the shared attractor memory and tags each `(center, att.wins > att.losses)` so the scheduler can weight them; gated on `SLEEP_MIN_PATTERNS=3` so a near-empty memory never churns.

---

### Phase F: Metacognitive Confidence-of-Confidence

**File:** `src/hanoon_prime/brain/metacog.py` (new, ~100 lines) + modifications to `brain/cognitive/metacognition.py`
**Status:** `pending`
**Priority:** 6th — prevents overconfidence
**Effort:** ~200 lines total
**Timeline:** Week 6-7

**Acceptance Criteria:**
- [ ] `MetaMonitor` tracking correlation between confidence bins and actual outcomes
- [ ] `confidence_reliability` score (rolling calibration correlation)
- [ ] When reliability drops: all confidence-based sizing shrinks
- [ ] Surprise detection: situation matches no known pattern → flag
- [ ] Curiosity drive: high surprise + stable pillar → exploration; high surprise + falling pillar → retreat
- [ ] Tests: `tests/test_metacog.py`
- [ ] Pre-commit hooks pass

**Biological Basis:** Fleming & Dolan 2012 (metacognition BA10), Schwartenbeck et al. 2015 (active inference + uncertainty)

**Design:**

```python
class MetaMonitor:
    """Second-order confidence: how reliable is my current confidence?

    Tracks correlation between confidence bins and actual outcomes
    over a rolling window. Low reliability → shrink sizing, increase
    exploration.
    """

    def update(self, conf_bin: int, outcome: bool) -> None: ...
    def reliability(self) -> float: ...
    def surprise(self, alpha: dict, regime: str, episodic) -> float: ...
    def sizing_scalar(self) -> float: ...

    def save(self) -> dict: ...
    def load(self, data: dict) -> None: ...
```

**Implementation Steps:**
1. Write `brain/metacog.py`
2. Wire into `orchestrator._score_pipeline` — surprise detection
3. Wire reliability into `risk.py` sizing
4. Wire curiosity into exploration/exploitation balance
5. Write `tests/test_metacog.py`
6. Update this doc with commit hash

**Design Decisions:** (none yet)

---

## 4. Module Contracts Summary

| Module | Input | Output | Wired Into | Persisted |
|---|---|---|---|---|
| `rpe.py` | predicted_win_prob, won, regime | {phasic, tonic, meta, surprise, v_fast, v_slow, v_meta} | `_adapt_weights` (LR mod), `_update_pillar` (mood), `emotion.py` (tonic affect), `shared_state` | `runtime/juli_rpe.json` |
| `allostasis.py` | win_loss_record, regime | {setpoint, deviation, dyshomeostatic, violations, trades} | `_update_pillar` (dynamic fallen line), `dynamics.adapt_threshold` (tighten on dyshomeostasis), `shared_state` (`allostatic`), HALIM via `pillar_setpoint` | `runtime/juli_allostasis.json` |
| `somatic.py` | pillar tilt, rpe.phasic/tonic, allostatic.dyshomeostatic | {marker ±0.10, precision ∈ [0.6, 1.0]} | `_score_pipeline` (raw bias + modifier dampen), `shared_state` (somatic_marker, somatic_precision) | state-only (in-memory snapshot) |
| `extinction.py` | alpha, outcome, regime, conf, horizon | inhibition weight (context-gated, ≤EXTINCT_MAX) | `_episodic_bias` = excitation − inhibition; regime renewal via `reactivate(canon)` | `runtime/juli_extinction.json` |
| `sleep_scheduler.py` | last_trade_time, session_close | trigger bool + replay weights | consolidation `_cycle` + `stop()`; `neurons/sleep.py` `replay_list` override | — |
| `metacog.py` | conf_bin, outcome, alpha, regime | reliability, surprise, sizing_scalar | _score_pipeline, risk | `state.json` |

---

## 5. Design Decisions Log

| Date | Decision | Rationale | Alternatives Rejected |
|---|---|---|---|
| 2026-09-14 | Multi-timescale RPE as 3-channel scalar decomposition | Matches biological DA heterogeneity (Masset 2025); simpler than full TD(n) | Full TD(λ) — too complex for first pass; separate per-organ RPE — redundant |
| 2026-09-14 | Allostatic setpoints per-regime, not global | Market regimes have different normal distributions; global setpoint would trigger false alarms | Single global setpoint — loses regime specificity; fully adaptive (no setpoint) — loses reference |
| 2026-09-14 | Setpoint EMA gated by `ALLOS_MIN_TRADES`, fallen line never looser than structural `PILLAR_EDGE_FALL` | Fresh norms must not swing the pillar; allostasis may only tighten discipline, never loosen it | Un-gated EMA — first-trade noise moves the norm; unbounded dynamic threshold — could overturn the -0.10 safety floor |
| 2026-09-14 | Dyshomeostasis transient: violations reset when the EMA catches up to a persistent edge | A recurring negative edge becomes the *new normal* (Sterling) — the durable response is the tightened fallen line, not a permanent alarm | Permanent alarm on adapted norm — false distress; hysteresis-latched alarm — complexity without payoff |
| 2026-09-14 | Allostatic state published as its own shared-state dict, not merged into `pillar` | Pillar semantics stay realized-data-driven; telemetry readable by HALIM/webapp via separate keys | Merging into pillar — conflates geometry with bodily state; pillar-only reporting — loses dyshomeostasis to dynamics |
| 2026-09-14 | Somatic marker is asymmetric: pillar lean penalizes, tonic RPE mood is signed | A wounded pillar shouldn't produce "excitement"; RPE mood provides the bidirectional direction | Symmetric marker (same in both directions) — loses the loss-aversion asymmetry that Damasio's data requires |
| 2026-09-14 | Extinction signature = rounded-alpha bins with shared-dim-overlap neighbors | Coarse pattern keys survive intra-situation jitter and are cheap to test; scaling by `shared/len(tags)` gives graded, not binary, inhibition transfer | Exact k-NN inhibition (continuous distance) — noisier and more expensive for a learned gate |
| 2026-09-14 | Performance is a slow EWMA (0.7/0.3), not a loss-streak counter | A couple of losses shouldn't flip a reliable pattern; sustained degradation grows the weight, healthy shares decay it continuously | Loss-streak threshold — brittle to transient noise, needs a manual reset path |
| 2026-09-14 | Somatic marker bounded ±0.10 | Consistent with existing modulator bounds; prevents single module from dominating | Larger bound — risk of runaway; no bound — dangerous in live system |
| 2026-09-14 | Extinction via separate inhibition weights, not weight decay modification | Preserves original excitatory weight (CLS: extinction ≠ forgetting); allows reactivation | Increasing decay rate — loses memory; zeroing weights — destroys without context |
| 2026-09-14 | Sleep replay 3× loser weighting | Prevents overconfidence from replaying winners; matches biological "replay to learn" not "replay to enjoy" | Equal weighting — misses learning opportunity; loser-only — loses winner patterns |
| 2026-09-14 | `SleepScheduler.check` uses explicit `now` param, `_last_trigger` sentinel `-inf` | Tests inject synthetic time without monkeypatching; sentinel avoids spurious cooldown on first trigger | Monkeypatching `time.time` — fragile and shared-state leaking; sentinel `0.0` fails under synthetic timestamps |
| 2026-09-14 | Cooldown only after first trigger, shared between idle and session-close | One replay per cooldown window prevents double-runs if session-close follows an idle trigger | Cooldown on all calls — session-close never triggers; no cooldown — double replay in rapid succession |
| 2026-09-14 | `replay_weights` returns `[(pattern, drive)]` pairs | Caller (consolidation) and engine both see the same structure; no dict-to-list conversion needed | Return a dict — loses ordering; return drive array alone — loses pattern association |
| 2026-09-14 | Engine `select_patterns` / `run_cycle` accept `replay_list` override | Scheduler passes pre-weighted list; tests isolate weighting from attractor iteration | Always use attractors — no way to test or override weighting externally |
| 2026-09-14 | Auto-replay runs 5s, offline replay uses full 60s | 5s keeps the 30s S2 loop responsive; offline mode has no latency constraint | Both 60s — S2 loop blocked; both 5s — offline replay incomplete |
| 2026-09-14 | `_sleep_patterns` wraps attractor centers as `{alpha_i: v}` dicts | Matches the dict-key schema both the scheduler and `select_patterns` consume; no format conversion | Store attractors as dict directly — breaks the `Attractor(center=[float])` dataclass contract |
| 2026-09-14 | Metacognition as reliability score, not second-order Bayesian | Simpler to implement and validate; can upgrade later | Full Bayesian second-order — premature complexity |

---

## 6. Verification Checklist

### Pre-Implementation (Per Phase)
- [x] AWAKENED_BRAIN.md updated with design decisions
- [x] Tests written and passing: `pytest tests/test_<module>.py --no-cov -q`
- [x] Full suite passing: `pytest --no-cov -q` (1107 tests)
- [x] Pre-commit hooks pass (ruff, black, complexity, file-length, contracts)
- [ ] Webapp typecheck + build pass: `npm run typecheck && npm run build` *(Phase A: no webapp changes — RPE exposed via pillar.rpe_phasic/tonic, webapp panel reads it; no new component)*

### Post-Implementation (Per Phase)
- [x] Commit hash recorded in this doc under the relevant phase *(`9d11592` Phase A; `145e600` Phase B backend; `689f9b6` Phase B webapp; `8da15db` Phase C; `fd71504` Phase D; Phase E pending this commit)*
- [x] Module appears in `brain/__init__.py` exports *(rpe: MultiTimescaleRPE; orchestrator imports it — no top-level exports needed)*
- [x] Shared state keys documented in `brain/shared_state.py` *(rpe_phasic, rpe_tonic, rpe_meta, rpe_surprise + allostatic + somatic_marker + somatic_precision added to _state)* *(Phase D: net context-gated `episodic_bias` mirrors to state; extinction_size in brain snapshot)*
- [ ] HALIM evidence prompt updated (if applicable) *(Phase A: no HALIM prompt change — RPE available via state.)* *(Phase B: `pillar_fields` now emits `pillar_setpoint` + `pillar_deviation`, which flow into the evidence dict automatically — no halim_evidence.py edit needed)* *(Phase D: net episodic bias flows through existing `episodic_bias` state key — no prompt edit)*
- [x] Webapp panel updated (if applicable) *(Phase A: no webapp changes needed — existing pillar panel inherits new keys.)* *(Phase B: setpoint line + dyshomeostasis chip in PillarBalancePanel, hanoon-dash `a837644`)*
- [x] Design decisions logged *(6 design decisions under Phase A, 6 under Phase B, 6 under Phase C, 6 under Phase D, 6 under Phase E)*
- [x] This document updated with any deviations from plan *(deviation: on_trade_close/refactor to helper methods; R3 test contract does NOT skip orchestrator.py; Phase E: `_last_trigger` sentinel changed to `-inf`; auto-replay capped at 5s to avoid blocking S2; `sleep_patterns` builds attractor centers with `{alpha_i: v}` wrapper)*

### Final Verification (All Phases)
- [ ] All 6 modules implemented and wired
- [ ] All 6 test files passing
- [ ] Full suite green
- [ ] Webapp shows all new surfaces
- [ ] HALIM prompt includes all biological signals
- [ ] No regressions in existing pillar/learning behavior

---

## 7. Active Session Log

| Date | Session | What Changed | Commit | Notes |
|---|---|---|---|---|
| 2026-09-14 | Analysis & design | Full brain mapping, 6 biological gaps identified, 6-phase proposal | `d630f56` | Research: dopamine RPE, CLS, allostasis, somatic markers, extinction, metacognition |
| 2026-09-14 | Phase A — Multi-timescale RPE | New `brain/rpe.py` (3-channel dopamine RPE); wired into orchestrator, reflection, consolidation, emotion, pillar_awareness, shared_state; 17 tests; orchestrator refactored (3 new private methods) to respect R3 40-line contract; full suite 1020 passed | `9d11592` | Deviation: `_adapt_threshold` + `_update_rpe` + `_reflect_close` extracted from on_trade_close/_learn_from_real; RPE pytest contract does NOT skip orchestrator.py unlike the shell complexity check |
| 2026-09-14 | Phase B — Homeostatic setpoint + interoception | New `brain/allostasis.py` (AllostaticController, per-regime setpoint EMA, dyshomeostasis, atomic persistence); `pillar_awareness.py` dynamic fallen line + `_decorate` (setpoint keys for HALIM); `dynamics.adapt_threshold` gains dyshomeostatic tightening; `consolidation._update_pillar` publishes `allostatic`; orchestrator isinstance-guard (Class C watchdog); 21 tests; full suite 1040 passed | `145e600` | Deviation: 2 allostasis tests initially asserted non-transient dyshomeostasis — corrected to the transient-alarm design (violations reset as the EMA absorbs a persistent edge); `or {}` rejected for Class C compliance |
| 2026-09-14 | Phase B — webapp + inspection evidence | `inspection/pillar.py` emits `pillar_setpoint`/`pillar_deviation`/`below_setpoint` on the `pillar_balance` evidence (helper `_pillar_evidence` keeps `pillar_balance` ≤40 lines); PillarBalancePanel shows the allostatic setpoint line + dyshomeostasis chip; webapp typecheck + build green | `689f9b6` (+ hanoon-dash `a837644`) | Deviation: `pillar_balance` hit 43 lines after evidence addition — extracted `_pillar_evidence` helper to restore R3 compliance |
| 2026-09-14 | Phase C — Somatic markers | New `brain/somatic.py` (SomaticMarkerGenerator: bounded ±0.10 gut-feel bias from pillar lean + RPE mood + allostatic alarm; negative marker precision-dampens all learned modifiers, floored 0.6); wired via `_somatic_context` into `_score_pipeline` (raw bias + `_compute_mods` precision scale); published `somatic_marker`/`somatic_precision`; 18 tests; full suite 1058 passed | `8da15db` | Deviation: `_compute_mods` private helper extracted to keep `_score_pipeline` at exactly 40 lines after black re-wrapped the modifier sum into 9 lines (R3 contract); somatic is asymmetric (tilt only penalizes) per prospect-theory loss aversion |
| 2026-09-14 | Phase D — Extinction + context-tagged memory | New `brain/extinction.py` (ExtinctionTracker: context-gated inhibition via signature overlap; perf EWMA; regime renewal); `episodic.py` gains optional context tag on add/predict/modifier; orchestrator `_episodic_bias` returns net excitation−inhibition, reactivates on regime change; `_bounded_thinker()` helper to hold R3 at 40; `conf_bin_label`/`context_key` helpers; 27 tests; full suite 1085 passed | `fd71504` | Deviation: context tagging implemented inside `ExtinctionTracker` + `episodic` (not a parallel overlay); `_signature` uses coarse EPISODIC_KEYS rounding (1-dp) with shared-dim-overlap scaling, not exact k-NN inhibition; `_bounded_thinker()` consolidation to recover R3 when black wrapped the `_compute_mods` call to 3 lines |
| 2026-09-14 | Phase E — Sleep replay scheduler | New `brain/sleep_scheduler.py` (SleepScheduler: idle/session-close trigger with cooldown; loser 3× weighting; interleaved historical traces); `neurons/sleep.py` flipped `WIN_BIAS` 2.0→1.0, added `LOSS_BIAS=3.0`, `replay_list` override param on `select_patterns`/`run_cycle`; `consolidation.py` wires `_sleeper` into `_cycle` + `stop()` with 5s short replay + `_sleep_patterns` from attractor memory; 22 tests; full suite 1107 passed | `TBD` | Deviation: `_last_trigger` sentinel changed from `0.0` to `-inf` so synthetic-test timestamps don't spuriously hit the cooldown guard; engine merge-count starts at 0 (store=1 create, then 1+ additional), so tests store 3× to reach `trade_count≥2`; auto-replay runs 5s not 60s to avoid blocking the 30s S2 loop |

---

*This document is enforceable: no implementation commits without updating this doc first.*
*Every design decision, every deviation, every test result goes here.*
