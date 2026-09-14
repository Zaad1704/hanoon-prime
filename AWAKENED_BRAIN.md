# AWAKENED_BRAIN.md — Living Design & Implementation Contract

> **This document is the single source of truth for Juli's biological learning system.
> Every implementation session MUST update this document before committing.**

**Created:** 2026-09-14
**Status:** Design complete, Phase A ready to implement
**Last updated:** 2026-09-14 — initial creation

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

**File:** `src/hanoon_prime/brain/allostasis.py` (new, ~150 lines)
**Status:** `pending`
**Priority:** 2nd — pillar becomes dynamic
**Effort:** ~400 lines total (module + wiring + tests + webapp)
**Timeline:** Week 2-3

**Acceptance Criteria:**
- [ ] `AllostaticController` class with dynamic setpoints per regime
- [ ] Setpoints updated from recent outcome statistics (rolling window)
- [ ] `dyshomeostasis` detection: sustained deviation from setpoint for N trades
- [ ] `pillar_awareness.py` compares edge to dynamic setpoint, not fixed `PILLAR_EDGE_FALL`
- [ ] `dynamics.adapt_threshold` receives allostatic state → aggressive tightening on dyshomeostasis
- [ ] HALIM evidence prompt includes setpoint deviation info
- [ ] Webapp pillar panel shows setpoint and deviation
- [ ] Tests: `tests/test_allostasis.py` — setpoint tracking, regime shift response, dyshomeostasis detection
- [ ] Pre-commit hooks pass

**Biological Basis:** Sterling 2012 (allostasis), Keramati & Gutkin 2014 (homeostatic RL), Pezzulo et al. 2015 (active inference + interoception)

**Design:**

```python
class AllostaticController:
    """Dynamic reference points for pillar, win-rate, and pred-error.

    Each regime maintains its own setpoint. Setpoints shift based on
    recent outcome statistics (not fixed constants).
    """

    def update(self, pillar: dict, regime: str, recent_trades: list) -> dict:
        ...

    def setpoint(self, regime: str) -> dict: ...
    def deviation(self, pillar: dict, regime: str) -> float: ...
    def is_dyshomeostatic(self, regime: str) -> bool: ...

    def save(self) -> dict: ...
    def load(self, data: dict) -> None: ...
```

**Implementation Steps:**
1. Write `brain/allostasis.py` with `AllostaticController`
2. Wire into `consolidation._update_pillar` — controller updates setpoints each cycle
3. Modify `pillar_awareness.py` `compute_pillar_awareness` to accept dynamic setpoint
4. Wire dyshomeostasis into `dynamics.adapt_threshold` as aggression trigger
5. Extend HALIM evidence prompt with `setpoint_deviation`, `setpoint_duration`
6. Extend webapp PillarBalancePanel with setpoint line
7. Write `tests/test_allostasis.py`
8. Update this doc with commit hash

**Design Decisions:** (none yet)

---

### Phase C: Somatic Markers

**File:** `src/hanoon_prime/brain/somatic.py` (new, ~100 lines)
**Status:** `pending`
**Priority:** 3rd — rapid pre-scoring bias
**Effort:** ~250 lines total
**Timeline:** Week 3-4

**Acceptance Criteria:**
- [ ] `SomaticMarkerGenerator` class producing per-decision bias from full outcome trajectory
- [ ] Inputs: pillar state, RPE channels, allostatic state, episodic recall
- [ ] Output: single bounded bias (±0.10)
- [ ] Applied in `_score_pipeline` as `somatic_marker` modifier BEFORE detailed scoring
- [ ] When marker is negative, other modifiers are precision-dampened
- [ ] Published to shared state for webapp visualization
- [ ] Tests: `tests/test_somatic.py`
- [ ] Pre-commit hooks pass

**Biological Basis:** Damasio 1994 (somatic marker hypothesis), Bechara et al. 1996 (Iowa Gambling Task), Damasio 1996 (vmPFC + body loop / as-if body loop)

**Design:**

```python
class SomaticMarkerGenerator:
    """Gut-feel bias from the entire recent outcome trajectory.

    Combines pillar body-state, RPE mood, allostatic alarm, and
    episodic recall into a single rapid bias signal applied BEFORE
    detailed scoring — like Damasio's vmPFC markers that prune the
    search space before conscious reasoning.
    """

    def generate(self, alpha: dict, regime: str, pillar: dict,
                 rpe: dict, allostatic: dict) -> float:
        ...

    def precision_weight(self, marker: float) -> float:
        """Dampen other modifiers when marker is negative."""
        ...
```

**Implementation Steps:**
1. Write `brain/somatic.py`
2. Wire into `orchestrator._score_pipeline` after regime weights, before threshold
3. Wire precision dampening into modifier application
4. Publish to shared state
5. Write `tests/test_somatic.py`
6. Update this doc with commit hash

**Design Decisions:** (none yet)

---

### Phase D: Extinction + Context-Tagged Memory

**File:** `src/hanoon_prime/brain/extinction.py` (new, ~120 lines) + modifications to `brain/episodic.py`
**Status:** `pending`
**Priority:** 4th — regime transition handling
**Effort:** ~300 lines total
**Timeline:** Week 4-5

**Acceptance Criteria:**
- [ ] Context tags on episodic memories: (regime, confidence_bin, horizon)
- [ ] Inhibition weights separate from excitatory weights
- [ ] At retrieval: `modifier = excitation - inhibition` (both context-gated)
- [ ] On regime transition: reactivate suppressed patterns if regime matches
- [ ] Extinction signal when pattern performance degrades within context
- [ ] Tests: `tests/test_extinction.py`
- [ ] Pre-commit hooks pass

**Biological Basis:** Bouton 2004 (context-dependent extinction), KM Myers & Davis 2007 (extinction retrieval)

**Design:**

```python
class ExtinctionTracker:
    """Context-tagged inhibition weights for episodic memories.

    Every episodic memory gets a context tag (regime, conf_bin, horizon).
    When a pattern's recent performance degrades, an inhibition weight
    grows. At retrieval, modifier = excitation - inhibition (both
    context-gated).
    """

    def record(self, alpha: tuple, outcome: float, context: dict) -> None: ...
    def inhibition(self, alpha: tuple, context: dict) -> float: ...
    def reactivate(self, old_context: dict, new_context: dict) -> None: ...

    def save(self) -> dict: ...
    def load(self, data: dict) -> None: ...
```

**Implementation Steps:**
1. Write `brain/extinction.py`
2. Modify `brain/episodic.py` to accept context tags on `.add()` and `.modifier()`
3. Wire extinction into `orchestrator.on_trade_close` — record with context
4. Wire reactivation into regime transition detection
5. Write `tests/test_extinction.py`
6. Update this doc with commit hash

**Design Decisions:** (none yet)

---

### Phase E: Sleep Replay Scheduler

**File:** `src/hanoon_prime/brain/sleep_scheduler.py` (new, ~80 lines) + modifications to `brain/consolidation.py`
**Status:** `pending`
**Priority:** 5th — engine exists, needs scheduling
**Effort:** ~200 lines total
**Timeline:** Week 5-6

**Acceptance Criteria:**
- [ ] Auto-trigger `sleep_replay` after 30min inactivity or session close
- [ ] Replay losing trades 3× more often than winners (prevent overconfidence)
- [ ] Interleave with random historical trades (CLS interleaved learning)
- [ ] After replay: update regime_weights and episodic from consolidated patterns
- [ ] Tests: `tests/test_sleep_scheduler.py`
- [ ] Pre-commit hooks pass

**Biological Basis:** McClelland 1995 (CLS theory), Kumaran et al. 2016 (CLS updated), Singh et al. 2022 (bidirectional replay)

**Design:**

```python
class SleepScheduler:
    """Triggers offline replay during inactivity periods.

    Criteria: no trades for SLEEP_THRESHOLD_SEC (default 1800) or
    session close. Replay weights losing trades 3× higher.
    """

    def check(self, last_trade_time: float, session_close: bool) -> bool: ...
    def replay_weights(self, recent_trades: list) -> dict: ...
```

**Implementation Steps:**
1. Write `brain/sleep_scheduler.py`
2. Wire into `consolidation._cycle` — check trigger each cycle
3. Modify `sleep_replay` call to use weighted replay list
4. Write `tests/test_sleep_scheduler.py`
5. Update this doc with commit hash

**Design Decisions:** (none yet)

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
| `allostasis.py` | pillar, regime, recent_trades | {setpoint, deviation, is_dyshomeostatic} | pillar_awareness, dynamics, HALIM prompt | `state.json` |
| `somatic.py` | alpha, regime, pillar, rpe, allostatic | float (±0.10) | _score_pipeline | shared_state |
| `extinction.py` | alpha, outcome, context | inhibition weight | episodic.modifier | `juli_state.json` |
| `sleep_scheduler.py` | last_trade_time, session_close | trigger bool + replay weights | consolidation | — |
| `metacog.py` | conf_bin, outcome, alpha, regime | reliability, surprise, sizing_scalar | _score_pipeline, risk | `state.json` |

---

## 5. Design Decisions Log

| Date | Decision | Rationale | Alternatives Rejected |
|---|---|---|---|
| 2026-09-14 | Multi-timescale RPE as 3-channel scalar decomposition | Matches biological DA heterogeneity (Masset 2025); simpler than full TD(n) | Full TD(λ) — too complex for first pass; separate per-organ RPE — redundant |
| 2026-09-14 | Allostatic setpoints per-regime, not global | Market regimes have different normal distributions; global setpoint would trigger false alarms | Single global setpoint — loses regime specificity; fully adaptive (no setpoint) — loses reference |
| 2026-09-14 | Somatic marker bounded ±0.10 | Consistent with existing modulator bounds; prevents single module from dominating | Larger bound — risk of runaway; no bound — dangerous in live system |
| 2026-09-14 | Extinction via separate inhibition weights, not weight decay modification | Preserves original excitatory weight (CLS: extinction ≠ forgetting); allows reactivation | Increasing decay rate — loses memory; zeroing weights — destroys without context |
| 2026-09-14 | Sleep replay 3× loser weighting | Prevents overconfidence from replaying winners; matches biological "replay to learn" not "replay to enjoy" | Equal weighting — misses learning opportunity; loser-only — loses winner patterns |
| 2026-09-14 | Metacognition as reliability score, not second-order Bayesian | Simpler to implement and validate; can upgrade later | Full Bayesian second-order — premature complexity |

---

## 6. Verification Checklist

### Pre-Implementation (Per Phase)
- [x] AWAKENED_BRAIN.md updated with design decisions
- [x] Tests written and passing: `pytest tests/test_<module>.py --no-cov -q`
- [x] Full suite passing: `pytest --no-cov -q` (1020+ tests)
- [x] Pre-commit hooks pass (ruff, black, complexity, file-length, contracts)
- [ ] Webapp typecheck + build pass: `npm run typecheck && npm run build` *(Phase A: no webapp changes — RPE exposed via pillar.rpe_phasic/tonic, webapp panel reads it; no new component)*

### Post-Implementation (Per Phase)
- [ ] Commit hash recorded in this doc under the relevant phase *(fill on commit)*
- [x] Module appears in `brain/__init__.py` exports *(rpe: MultiTimescaleRPE; orchestrator imports it — no top-level exports needed)*
- [x] Shared state keys documented in `brain/shared_state.py` *(rpe_phasic, rpe_tonic, rpe_meta, rpe_surprise added to _state)*
- [ ] HALIM evidence prompt updated (if applicable) *(Phase A: no HALIM prompt change — RPE available via state)*
- [x] Webapp panel updated (if applicable) *(Phase A: no webapp changes needed — existing pillar panel inherits new keys)*
- [x] Design decisions logged *(6 design decisions documented under Phase A)*
- [x] This document updated with any deviations from plan *(deviation: on_trade_close/refactor to helper methods; R3 test contract does NOT skip orchestrator.py)*

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
| 2026-09-14 | Phase A — Multi-timescale RPE | New `brain/rpe.py` (3-channel dopamine RPE); wired into orchestrator, reflection, consolidation, emotion, pillar_awareness, shared_state; 17 tests; orchestrator refactored (3 new private methods) to respect R3 40-line contract | *pending* | Deviation: `_adapt_threshold` + `_update_rpe` + `_reflect_close` extracted from on_trade_close/_learn_from_real; RPE pytest contract does NOT skip orchestrator.py unlike the shell complexity check |

---

*This document is enforceable: no implementation commits without updating this doc first.*
*Every design decision, every deviation, every test result goes here.*
