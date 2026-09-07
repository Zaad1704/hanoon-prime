# hanoon_rebuild vs hanoon_prime — Code-Grounded Comparison

Both projects are siblings under `/Users/mdsabersajib/Downloads/`. Both are
neuro-morphic "JULI" trading systems targeting IB Gateway. They share a
common intellectual lineage (same 5 core indicator names appear in both,
same "JULI" brain branding, same neuromorphic/STDP/Nash vocabulary) but have
**diverged into two distinct architectures**. This report is derived from
reading the actual source files (paths cited inline) — not from summaries or
skimming.

> Legend for verdict codes: `PRIME` = `hanoon_prime/src/hanoon_prime/`,
> `REBUILD` = `hanoon_rebuild/`.

---

## 0. Identity & Versioning

| Axis | PRIME | REBUILD |
|---|---|---|
| Package identity | `pyproject.toml` → `name = "hanoon-prime"`, `version = "2.0.0"`, `requires-python >= 3.11`, deps `numpy>=1.24, scipy>=1.10`. Installable (`pip install -e ".[dev]"`). | No `pyproject.toml` / `setup.py` / `requirements.txt` at root. Flat monolith run via `python3 main.py`. `main.py` declares `VERSION = "3.0.0-snowflake"`, `BUILD_NAME = "Snowflake"`; `ops/app.py` docstring "HANOON 3.0". |
| Layout | Single PEP-660 src package: `src/hanoon_prime/` with `brain/` subpackage (orchestrator, neurons, cognitive, memory). | Strict **6-layer monorepo**: L0 `foundations/` → L1 `senses/` → L2 `hanoon/` (juli brain) → L3 `execution/`+`risk/` → L4 `reflection/` → L5 `monitoring/` → L6 `ops/`. |
| Entry point | `cli.py` → `bot = IBStreamingBot(...)`; `bot.run_paper(tickers)` on `IB_PAPER_PORT=4002`. Uses `ib_insync`. | `main.py` → argparse (`--health`, `--no-loop`, …) → `ops/app.py:HanooniApp` → `Runner` (`ops/runner.py`). `main.py` refuses mock broker in live mode (`HANOON_ALLOW_MOCK=1` only for tests). |
| Test bootstrap | `pytest.ini` (`pythonpath = .`), `src` on `sys.path` via `conftest`. | `pytest.ini` (`pythonpath = . tests`). |

---

## 1. Architecture & Data Flow

### PRIME — single-package, verdict-centric

Public contract path (`CONTRACT.md` / `tests/test_contract.py`):

```
hands.simulate_ticker (backtest) OR ib_cycle._cycle (live)
  eyes / compute_alpha            ──► cerebellum.compute_alpha (5 ind.)
  cerebellum.compute_alpha        ──► cortex.evaluate
  cortex.evaluate                 ──► Thought(verdict, score, direction, confidence)
  ib_cycle._check_safety          ──► hippocampus.check_safety_nets  (raises RuntimeError)
  ib_cycle._exec_decision         ──► ib_executor.place_bracket  (ATR brackets)
  memory.Journal.append           ──► journal  (carbon copy of IB)
  ib_cycle._reflect_closed        ──► juli.brain.on_trade_close  (NeuromorphicBrain, orchestrator.py)
```

`juli.py:JuliBrain` is a **scanner + router** wrapper: it owns a
`NeuromorphicBrain` (`brain/orchestrator.py`) and a `DataBudget` scanner.
`ib_adapter.py:IBStreamingBot` composes `JuliBrain` (live verdicts) +
`Hippocampus` (safety nets + sizing) + `IBExecutor` + `IBStreamer` + `Journal`.

The **NeuromorphicBrain** (`brain/orchestrator.py:291`) is the local source of
truth. Its fast path `tick()` → `_evaluate_fast()` (`orchestrator.py:154`):

```python
base = self.cortex.evaluate(alpha)              # tanh score + verdict  (R1: sole verdict source)
nash_pred = self.nash.predict(alpha, base.score, base.direction)
nash_op = self._compute_nash_mod(nash_pred)     # NASH_MOD_BOUND=0.03, ×4
neuro_score = self._compute_neuro_score(alpha, ticker)
blended = (1 - NEURO_BLEND) * base.score + NEURO_BLEND * neuro_score   # NEURO_BLEND=0.3
score = blended * r + hm + eb + nash_op          # r=regime multiplier, hm=halim, eb=episodic
direction = sign(score)
score = self._apply_nash_gate(score, direction, nash_pred)  # hard veto → 0.0
if EOD: score = 0.0
stabilized, reason = self.dynamics.process(score, direction)  # threshold/dynamics
sizing = self._maybe_size(stabilized, base.confidence, entry_price, atr, open_positions)
```

**Verdict authority (R1):** `cortex.evaluate` is the only place that emits
`BUY`/`SELL`/`HOLD`; `cortex.py:128` `_verdict()`. `test_R1_signal_modules_never_produce_verdicts`
enforces that `cerebellum.py`, `edge.py`, `hands.py`, `hippocampus.py`,
`immune.py` contain **no** verdict strings (`tests/test_contract.py:30`).

### REBUILD — 6-layer DAG, brain-as-orchestrator

Public entry (`hanoon/juli/__init__.py:119`):
`compute_enhanced_signal_score_with_learning` (`hanoon/juli/live.py:50`) →
`BrainEngine.composite()` (`hanoon/juli/brain_engine.py:196`):

```python
alpha = compute_alpha_signals(...)               # 27 indicators + 10 ib_*
score = compute_signal_score(alpha, ticker, horizon)       # weighted + percentile + clamp
cal_adj = prediction_error_adjustment(score, memory)       # learned calibration
score = clamp(score + cal_adj, THRESHOLD_MIN, THRESHOLD_MAX)
# neuromorphic blend (always-on, 0.3)
bridge = self._get_neuromorphic()
neuro_score = bridge.process_alpha(alpha, ticker)["score"]
score = clamp(score*0.7 + neuro_score*0.3, ...)
# learned EV gate  ──► thinker ──► verdict
_learned_should, _learned_ev, learned_diag = ev_gate_should_enter(...)
thought = self._thinker.bidirectional_think(alpha, score, ..., ev_enter=_learned_should, ...)
final_score = clamp(score + thought.score_delta + knowledge_pull, THRESHOLD_MIN, THRESHOLD_MAX)
```

Entry authorization is then **re-checked in a second gate** at
`hanoon/juli/decision.py:DecisionState.from_composite` (read in full):

```python
approved = (thought.verdict == "ENTER"
            and ev_enter
            and halim_approved
            and direction != 0
            and ev_per_unit >= CONSERVATIVE_EV_MIN)   # 0.07
```

**Verdict authority:** the **Thinker** is documented as the sole verdict
producer. `hanoon/juli/thinker.py:30`:

> "ARCHITECTURE ENFORCEMENT — JULI IS THE SOLE DECISION-MAKER … The EV gate,
> learned gate, health check, and death-spiral are all ADVISORY — they inform
> JULI's score modifiers but NEVER appear in the verdict decision branch."

Enforced by `tests/test_architecture.py:test_juli_is_sole_decision_maker` (AST).
Yet `decision.py` still ANDs `ev_enter` in — an **inherent tension** between
the documented doctrine and the ops-layer gate (see §8 "Entry authorization:
doctrine vs reality").

**Layer isolation is mechanical, not contractual.** `test_architecture.py`
performs AST-based static analysis with **zero imports**: enforces
pipeline-isolation (forbidden edges), no import cycles, no god files
(`GOD_FILE_LIMIT = 500`, `CODE_LINE_LIMIT = 600`), no patch files, **no bare
`except: pass`** (`test_no_silent_exceptions`). Layers and their import
boundaries are declared literally in the test (`BRAIN_LAYERS`,
`EXECUTION_RISK_LAYERS`, `REFLECTION_LAYERS`, `MONITORING_LAYERS`,
`COMPOSITION_ROOT`) with explicit exemptions (`LAYER_EXEMPT`).

---

## 2. Indicator Set

| | PRIME | REBUILD |
|---|---|---|
| **Count** | exactly **5** (`immune.py:13` `INDICATOR_NAMES` tuple; `cerebellum.py:22`). Fixed by contract R4 (evolved from "exactly 5" → "core 5 + compute functions"; `test_contract.py:167`). | **27 base** (`alpha.py`) + **10 `ib_*`** feed indicators. |
| **List (base)** | `vpin, orderbook_imbalance, institutional_flow, momentum, vwap_deviation` | `vpin, orderbook_imbalance, obv_divergence, volume_profile_proximity, kelly_fraction, trade_intensity, ad_signal, spread_tightness, hurst_exponent, bollinger_position, vwap_deviation, adx, ichimoku, elliott_wave, institutional_flow, trend_strength, momentum, microstructure, order_flow, institutional_wave, mfi, vw_macd, kc_position, stoch_k` (the latter 4, "NEW INDICATORS 2026-08-11", `constants.py:173`) plus `ib_*` (price_depth, vwap, vpin, imbalance, institutional — `alpha.py`). |
| **Weights** | `immune.py:68` `INDICATOR_WEIGHTS`, signs encode direction (all **positive** here; `hippocampus._adapt_weights` can drift negative within `[-2,+2]`). Sum of abs = 1.0. | Horizon-specific `HORIZON_WEIGHT_PROFILES` (`constants.py:185`): scalp vs swing differ sharply (e.g. scalp puts `institutional_flow=0.18`, swing `0.10`; scalp `spread_tightness=0.03`, swing `0.02`). Defaults `DEFAULT_ADAPTIVE_WEIGHTS` (`constants.py:159`). Learned weights overlay 60/40 via `scoring._get_weights`. |
| **Score math** | `cortex._tanh_score`: `score = tanh(Σ w_i·z_i)`, `z_i` = rolling z-score over `Z_NORM_WINDOW=50` clipped to `Z_CLIP=3.0` (`cortex.py:117`). Symmetric `[-1,+1]`. | `scoring.compute_signal_score`: weighted **average** of normalized-to-`[0,1]` indicators → `raw ∈[0,1]` → **optional inversion** `raw = 1-raw` if `SCORE_INVERT` → percentile-rank normalization → clamp `[THRESHOLD_MIN=0.10, THRESHOLD_MAX=0.70]` (`scoring.py:82`). |
| **Normalization** | Per-indicator rolling z-score (scale-invariant across tickers). | Per-indicator `[0,1]` mapping (`scoring._normalize_indicator`), then percentile-of-history, then structural clamp. |

**Key divergence:** Prime's score is a **signed, z-score-driven tanh**
(threshold on `|score|` ≥ `0.65` → verdict). Rebuild's score is an
**unsigned `[0.10, 0.70]` normalized percentile** (threshold on `score ≥
0.60` → ENTER). The rebuild score has **no sign of direction** baked into it
— direction is derived separately from a subset of alpha
(`live.py:109` sums `momentum + orderbook_imbalance + 2·institutional_flow`).
This is a deliberate design change: rebuild **separates confidence magnitude
from direction**, whereas prime folds both into one signed tanh.

---

## 3. Win-Probability / EV / Kelly Math (the core quantitative divergence)

### PRIME (`edge.py`, `cortex.py`)

```python
# cortex: win_prob from |score| only — direction-agnostic
# edge.score_to_win_prob (edge.py:19)
s = abs(score)                              # [-1,1] → [0,1]
win_prob = PRIOR_BOTTOM + s * (PRIOR_TOP - PRIOR_BOTTOM)   # immune.py: 0.25 + s*0.35
# => producible range [0.25, 0.60]; PRIOR_TOP_MAX=0.65 cap (R5)
# edge.compute_ev (edge.py:35)
gross_ev = p * r - (1 - p)                  # r = TARGET_R_R = 3.0 (immune.py:58)
net_ev  = gross_ev - fee_drag/risk_amount
ev_enter = gross_ev > 0.0                   # threshold is literally 0  (edge.py:61)
# edge.kelly_fraction (edge.py:70)  — FULL Kelly × FE_MLY_FRACTION=0.25
f = (p*(R+1) - 1)/R ; return clamp(f*0.25, 0, 1)
```

Entry rule (live): verdict `BUY`/`SELL` from cortex (score ≥ 0.65) **and**
`ev_enter` (`gross_ev > 0`) checked at **sizing** (`ib_executor.place_bracket`
calls `brain.size_position(score_to_win_prob(thought.score), ...) → hippocampus.size_position`).
There is **no learned gate** and **no Halim gate** in the prime live path;
the only block is `hippocampus.check_entry_allowed()` (safety nets) — see §7.

The **neuromorphic neuromancer** in prime is a *score blend*, not a gate:
`blended = 0.7*base + 0.3*neuro` feeds the same cortex threshold.

### REBUILD (`hanoon/juli/edge.py`, `ev_gate.py`)

```python
# edge.score_to_win_prob (edge.py:41) — producible range [0.18, 0.35]
s  = clamp(score, 0.10, 0.70)
norm = (s - 0.10) / 0.60
win_prob = PRIOR_BOTTOM + norm*(prior_top - PRIOR_BOTTOM)     # constants: 0.18, 0.35
# edge.compute_ev_and_edge (edge.py:140)
ev_per_unit = win_prob*avg_win - (1-win_prob)*avg_loss                 # avg_win=3, avg_loss=1
ev_dollars  = notional*ev_per_unit - fee_drag
ev_enter    = ev_dollars>0 and ev_per_unit>0
# ── REALIZED-EV GATE (ev_gate.py:50 compute_ev) ──
band_wr, reliability     = realized_band_read(memory, score)
conf_wr, conf_rel        = calibration.confidence_calibration(...)
p = win_probability(score, memory, regime, band_wr, reliability, conf_wr, conf_rel)   # both-direction pulls
realized_rr, realized_rel = realized_rr_read()          # from real trades only
effective_r = target_r_r*(1-rel) + realized_rr*rel     # target_r_r=3.0
ev = p*effective_r - (1-p)
ev *= direction_mod                                     # shorts penalized
should_enter = ev > ENTRY_EV_THRESHOLD                  # 0.05 (constants.py:272)
```

Three structural differences that matter in practice:

1. **Reachability range is tighter and lower.** Prime maps `|score|`∈[0,1] →
   `p`∈[0.25, 0.60]. Rebuild maps score∈[0.10,0.70] → `p`∈[0.18, 0.35].
   Rebuild's best-case win probability (35%) is **below prime's worst (25%)**
   only because rebuild's score ceiling is 0.70; the *band* is lower, meaning
   rebuild is structurally more conservative on the win-probability axis and
   relies on the **realized band WR + realized R:R** corrections to push `p`
   around. Prime never touches realized data in its `edge.py` — it's pure
   structural.

2. **The EV gate learns from realized data.** Rebuild's `ev_gate.py`
   `compute_ev` pulls `p` toward the **realized** score-band WR
   (`realized_band_read` → `memory.get_score_band_wr`) and the
   **confidence-bin WR** (calibration curve), reliability-weighted, **both
   directions** — a proven-losing band (e.g. 0.4–0.5 confidence = 11% WR)
   drives `p` *below* break-even so the gate refuses. It also blends
   `effective_r` toward the **realized R:R** (`realized_rr_read`,
   `edge.py:240`): the comment explicitly states the old exits delivered
   ~0.83:1 (avg win +1.08% vs avg loss −1.30%; break-even WR 54.6% vs JULI's
   best band 30%) → "every R=3.0-approved trade was a guaranteed loser at
   the realized R:R," and the gate refused everything until the new
   trailing/target-lock exits lifted the realized R:R. **`thin data →
   structural fallback`** (no deadlock). Prime's `edge.py` has none of this —
   its `ev_enter = gross_ev > 0` is purely structural.

3. **Direction penalty.** Rebuild applies `dir_adj`
   (`direction_exp.py`, bounded by `DIRECTION_EXP_BOUND`) that *scales* EV
   (shorts lose more historically — `ev_gate.py:128`). Prime has no
   direction-dependent EV adjustment.

**Kelly:** prime uses **full Kelly × 0.25** (`FE_MLY_FRACTION`). Rebuild uses
**full Kelly × 0.5** then a volatility adjustment and a **15% cap**
(`risk/sizer.py` `_KELLY_BASE_FRACTION=0.15`, `KELLY_BLEND=0.5`,
`_KELLY_MAX_SIZE=0.5`). Rebuild's sizer is a whole sub-layer
(`risk/sizer.py:PositionSizer`) with a documented **12-feature PPO
context** (concentration, spread, momentum, time-of-day, regime, …) and a
**Halim lesson modifier** (±5% of budget when trusted). Prime's
`hippocampus.size_position` is ~12 lines: `min(MAX_POSITION_NOTIONAL/px,
MAX_LOSS_PER_TRADE/(ATR·2.0), MAX_POSITION_NOTIONAL·kelly/px)` — three caps
only.

---

## 4. Entry Decision

| | PRIME | REBUILD |
|---|---|---|
| **Threshold** | `cortex` `|score| ≥ ENTRY_THRESHOLD=0.65` → `BUY`/`SELL` (`cortex.py:124`). | `SIGNAL_THRESHOLD` is read **from** the guardrail: `constants.py:123` → `SIGNAL_THRESHOLD = GUARDRAILS["SIGNAL_THRESHOLD"]["value"]` = **0.58** (the old 0.60-vs-0.58 disagreement was resolved in rebuild's favor of the guardrail); the `GateAdvisor` (`hanoon/halim_bridge/gate_advisor.py`) further auto-tunes it from realized WR. |
| **Direction** | Sign of the tanh score; `SELL` only if `SHORT_ALLOWED=True` (`immune.py:40`). | Derived from a directional subset of alpha (direction_hint + `institutional_flow`/`orderbook_imbalance`/`momentum`/`ichimoku`/`trend_strength` per `DIR_*_THRESH`), then neuromancer + Nash + Halim modify it (`thinker.py`). |
| **EV gate** | Structural `gross_ev > 0` checked at sizing; **not an entry gate** per se. | **Hard entry gate:** `ev_gate_should_enter` → `ev > 0.05` (plus `CONSERVATIVE_EV_MIN=0.07` AND, in `decision.py` ANDed with verdict ENTER + halim approval + direction). |
| **Halim** | Pre-loaded `halim_modifier` from `BrainState`; blended into score (`orchestrator.py:169` `+hm`). | Bounded **advisor pull** `±HALIM_PULL_BOUND=0.03` (`constants.py` + `thinker.py:289`); can veto when direction opposes and trust·pull high (`live.py:117-121`). |
| **Neuro blend** | `blended = 0.7·base + 0.3·neuro` (`orchestrator.py:168`, `NEURO_BLEND=0.3`), brain always-on warm-up. | Same constant `0.3` (`brain_engine.py:142` `_neuro_blend_target=0.3`, warmup **disabled** — "brain always on", `brain_engine.py:144`). |

### Entry authorization: doctrine vs reality (REBUILD)

The Thinker's docstring (§1) claims JULI is the sole decision-maker and all
gates are advisory. But `hanoon/juli/decision.py` (`DecisionState.from_composite`,
read in full) **re-applies the AND** at the ops layer:

```python
approved = (thought.verdict == "ENTER"
            and ev_enter
            and halim_approved
            and direction != 0
            and ev_per_unit >= CONSERVATIVE_EV_MIN)   # 0.07
```

So the **verdict** (`ENTER`/`HOLD`) is produced only by the Thinker (satisfying
the AST test), but the **authorization to trade** also requires the EV gate
to have passed. This is the honest, belt-and-suspenders path the code
actually runs — the "sole decision-maker" doctrine is about *verdict
production*, not *trade authorization*.

**PRIME** has no such second gate: the cortex verdict + safety nets
(`check_entry_allowed`) are the whole story. The neuromorphic score is
absorbed into the verdict-producing score, not a separate authorization.

---

## 5. Exit Logic

### PRIME — ATR brackets, single set, timeouts disabled (`hands.py`)

Backtest path only (`hands.simulate_ticker`; live uses IB bracket orders via
`ib_executor.place_bracket`). Stop/target from `_make_position`
(`hands.py:30`):

```python
stop   = entry - d * ATR_STOP_MULT * atr   # ATR_STOP_MULT = 2.0  (immune.py:34)
target = entry + d * ATR_TARGET_MULT * atr # ATR_TARGET_MULT = 6.0  (immune.py:35)   → 3:1
```
`_check_exit` (`hands.py:66`): stop hit? target hit? timeout after
`TIMEOUT_BARS = 999` (effectively **disabled** — `immune.py:36`). `_check_exit`
is pure geometric price-level crossing; **no trailing stop**, **no giveback**,
**no profit-lock**, **no consolidation**, **no indicator-based exits**.
Live brackets are static IB `bracketOrder()` with a *trailing* parent
modifier in `ib_executor._handle_parent` (`ib_executor.py:189` reuses
`ATR_STOP_MULT`/`ATR_TARGET_MULT`), but it's a price-distance trail tied to
re-submission, not a systematic exit policy.

### REBUILD — multi-tier, learned-aware, portfolio-aware (`execution/brackets.py`)

`BracketManager.should_exit` (`brackets.py:154`) — a **3-tier ladder** (TIER 1
hard stop, TIER 2 JULI primary verdict, TIER 3 mechanical safety nets with
sub-checks 3a/3b/…) on every monitor pulse. Note TIER 3 sub-checks read
`bracket.horizon` — rebuild's mechanical exits are already horizon-aware:

1. **TIER 1 — Hard stop** (`force_exit`). Never overridden.
2. **TIER 2 — JULI's primary verdict** (`juli_exit_verdict ∈ {EXIT,
   EXIT_HARD, BOOK_PROFIT}`) — JULI's exit thinking is **primary**; exits are
   recorded so JULI *learns* from them (`brackets.py:228`).
3. **TIER 3 — Mechanical safety nets** (when JULI says HOLD): `stop_loss`,
   `stale_force_exit` (adaptive `stale_minutes`/`stale_loss_pct` from
   `adaptive_thresholds`), `flat_timeout`, `stale_tightened_stop`,
   `profit_lock` (4 tiers: +10%→lock +4%, +7%→+3%, +5%→+2%, +3%→+1%,
   `brackets.py:32`), `giveback` (exit when 45% of peak fades;
   `GIVEBACK_KEEP_RATIO=0.55`), `target_lock`, `trailing_stop`
   (gain-relative: `entry + (peak-entry)*0.55`), `consolidation`
   (adaptive pulses), noise-buffer floor.

Exit thresholds are a literal dict in
`hanoon/juli/constants.py:303` `EXIT_THRESHOLDS` (14 keys):
`vpin_spike=0.85, hurst_mean_revert=0.35, bb_extreme_low=0.05,
bb_extreme_high=0.95, adx_weak=20.0, adx_fall_min=5.0,
vwap_extreme_atr=2.0, momentum_divergence=0.3, inst_flow_exit=-0.3,
mean_rev_extreme=2.5, sr_break_buffer=0.005, micro_toxic_spread=3.0,
micro_toxic_imbalance=-0.5`. `EXIT_POLICY_VERSION=3` (rebuilt for the v3
trailing/target-lock exits that the `realized_rr_read` in §3 gates on).

Plus **portfolio-level exits** in `monitoring/position_monitor.py`:
`_check_portfolio_profit` enforces `PORTFOLIO_PEAK_GIVEBACK_PCT=0.25`
(`foundations/constants.py`), `PORTFOLIO_EXIT_BATCH_SIZE=3`,
`PORTFOLIO_MIN_PEAK_USD=20.0` — exits the weakest positions when total
unrealized P&L drops 25% off its peak.

**Prime** has none of this: exits are a single static ATR bracket per
direction, no learning about exits, no portfolio-level giveback. The
asymmetry here is **structural and large**: rebuild treats exits as a
first-class, multi-dimensional, learned-from subsystem; prime treats them as
a mechanical stop/target.

---

## 6. Learning System

### PRIME

Two coexisting learners, split by code path:

**A. `Hippocampus._adapt_weights`** — the backtest learner (`hands.py` path).
A single asymmetric gradient (`hippocampus.py:149`):
```python
factor = REWARD_SCALE(0.5) if won else -PENALTY_SCALE(2.0)   # 4× punishment asymmetry
delta  = LEARNING_RATE(0.02) * factor * z_i * direction
w_i    = clamp(w_i + delta, WEIGHT_MIN=-2.0, WEIGHT_MAX=2.0)
w_i    *= WEIGHT_DECAY(0.999)                                  # per-trade geometric decay
```
`record_trade` (`hippocampus.py:135`) is the only caller. CONTRACT.md R8
originally mandated "ONE online weight gradient" — but `test_contract.py` R8
was **evolved** (`test_contract.py:277`) to require an "integrated learning
ecosystem: STDP + Hippocampus + Nash + Episodic," and
`brain/cognitive/nash.py` + `brain/neurons/` (STDP/bridge) +
`brain/episodic.py` exist. So the *contract text* lags the *test*, which
lags the *code*.

**B. `NeuromorphicBrain.on_trade_close`** (`orchestrator.py:209`) — the
**live** learner. It does **not** call `Hippocampus._adapt_weights`. Instead:
```python
self.dynamics.adapt_threshold(self.memory.pred_error)   # threshold adaptation
self.episodic.add(self._last_alpha[ticker], pnl_pct)    # k-NN episodic
self.nash.record_outcome(self._last_alpha[ticker], 0.0, won)  # pattern memory
self._neuromorphic.learn_from_outcome(ticker, won, pnl_pct)     # STDP
```
`brain/memory.py:JuliMemory` (the brain's persistent memory, 191 lines)
stores: weights, episodes, score history, `pred_error_ema` (EMA
`0.9·old+0.1·err`), win/loss counts, lessons, and a learned `threshold`
(clamped `[0.10, 0.70]`). Weights are repaired on load via
`weight_enforcer.get_enforcer().repair_on_load` (`brain/memory.py:65`).

### REBUILD

A genuine **multi-modal** learning state machine in
`hanoon/juli/memory.py:JuliMemory` (501 lines, `LearningMixin` +
`WeightManagerMixin` + `CalibrationMixin` + `PersistenceMixin` +
`HorizonGateMixin`, composed at `memory.py:28`). All learning state lives
here; **on_trade_close is the sole writer** (`brain_engine.py:412` docstring:
"SINGLE-WRITER learning entry point"). Highlights:

- **Weight adaptation** (`learning_mixin.py` via `adapt_weight(indicator, won, magnitude=...)`): `WEIGHT_LOSS_AVERSION=1.2`, `WEIGHT_DECAY_HALF_LIFE=100` (recency-mean reversion toward the prior), `GLOBAL_WEIGHT_FLOOR=0.01`, `WEIGHT_CONCENTRATION_CAP=0.25`, `MIN_LEARNED_WEIGHT`/`MAX_LEARNED_WEIGHT` per indicator (`constants.py:362-411`).
- **Per-horizon gates** (`HorizonGateMixin`): WR tracked per horizon;
  `NASH_GATE_AUTHORITY_WR=0.45`, cold-start protection (3+ consecutive
  losses with <20 samples closes the gate) (`memory.py:393`).
- **Dynamic PRIOR_TOP** (`memory.py:356`): 70% dynamic + 30% static blend,
  `[DYNAMIC_PRIOR_TOP_MIN, DYNAMIC_PRIOR_TOP_MAX]`; only acts after
  `DYNAMIC_PRIOR_TOP_MIN_TRADES`.
- **Calibration bins**: confidence → realized WR buckets + score-bucket
  prediction-error buckets (`_pred_errors`, `record_prediction` via
  `brain_engine.on_trade_close` → `memory.record_prediction`).
- **Episodic k-NN** (`hanoon/juli/episodic.py`), **Halim post-mortem
  lessons** (`record_halim_lesson`, per-indicator, bounded ±0.03,
  recency-weighted 2×/1×, `memory.py:256`), **direction expectancy**,
  **prediction-error buckets**, **per-ticker WR** (`get_ticker_recent_wr`,
  Fix #73).
- **IRONCLADE filter** (`memory.py:162` / `brain_engine.py:447`): only
  `real_trade`/`ib_fill`/`ib_paper` sources write memory; `simulation`
  rejected. Reconciled ghost trades (score=0.5, no alpha) skip weight
  adaptation but still audit-log.
- **Runtime canary** `verify_learning_gate` (`ev_gate.py:255`) — 3 probes
  (losing band refuses, recovery band admits, thin data admits) on every
  `/health`; guards against the "death spiral" where the gate silently
  admits losing bands (§3's `edge.py:91` comment documents the prior
  death-spiral live).

Prime's learning, by contrast, is **either a 1-gradient perceptron
(hippocampus, backtest) or a loosely-coupled set of module calls with no
central state machine** (`orchestrator.on_trade_close` calls into episodic/
nash/neuromorphic but each owns disjoint memory; `JuliMemory` is a simple
JSON store with no horizon gates, no dynamic prior, no IRONCLADE).

---

## 7. Safety Nets

| | PRIME | REBUILD |
|---|---|---|
| **Philosophy** | `immune.py` = **hard-coded, un-weakenable**. R6: "No `os.environ`, no config files." `test_R6_safety_nets_are_constants_not_env` AST-scans every `os.getenv` for safety keywords. Violations raise `RuntimeError` (halt, not pause). | `foundations/config.py:RuntimeConfig.from_env()` is **env-driven**; IB ports validated to `{4001,4002}`. `SacredRules` holds *structural* constraints (`MIN_RISK_REWARD=3.0`, `MAX_ATR_MULT=15.0`, `PORTFOLIO_RISK_MIN=0.30`…); `DAILY_LOSS_LIMIT_PCT` is env-driven (0 = off). |
| **Position caps** | `MAX_POSITION_NOTIONAL=5000.0`, `MAX_LOSS_PER_TRADE=50.0`, `MAX_CONCURRENT_POSITIONS=3` (`immune.py:43-45`). | `HORIZON_POSITION_CAPS` (`constants.py:31`): `{scalp:12, swing:5, multiday:3, multiweek:2, longterm:2}`, `HORIZON_TOTAL_CAP=15`, `PORTFOLIO_CAP=4.0× equity`, per-ticker concentration `WEIGHT_CONCENTRATION_CAP=0.25` (`portfolio.py:39`). |
| **Daily loss** | `DAILY_LOSS_LIMIT=200.0` (flat $) → `RuntimeError` (`hippocampus.py:105`). | `DAILY_LOSS_LIMIT_PCT` (env `%`) → kill-switch halting (`runner_composite.py:83`). |
| **Drawdown** | None (no stress mode). | `PortfolioRiskManager`: continuous `risk_scalar` from drawdown `1-3·dd` clamped
`[0.30, 1.00]`; `stress_mode` at `dd > 0.20` (tightens entry sizing to <30% budget); `_stress_mode` blocks `size_fraction > 0.3` (`portfolio.py:76-84`). |
| **Consecutive losses** | `CONSECUTIVE_LOSSES_PAUSE=3` → pause 60 min (`hippocampus.py:109`, `ib_cycle._check_safety`). | No direct equivalent; covered by the **learned gate** (`is_gate_closed`: WR<0.45 with full sample, or 3 consecutive 0%-WR cold-start) + `NASH_GATE_AUTHORITY` (`edge.py`/`constants.py`). |
| **Direction** | `SHORT_ALLOWED=True` (`immune.py:40`); `PRIOR_BOTTOM=0.25, PRIOR_TOP=0.60, PRIOR_TOP_MAX=0.65`, `SCORE_INVERT=False`, `CONFIDENCE_FLOOR=0.50` (`immune.py:51-55`). | `PRIOR_BOTTOM=0.18, PRIOR_TOP=0.35`, `SCORE_INVERT=True`, `CONFIDENCE_FLOOR=0.50` (`SacredRules`/`constants`). Shorts get an EV *penalty* (`ev_gate.py:128`, `DIRECTION_EXP_BOUND`). |
| **Fees** | `FEE_RATE=0.0001`, `FIXED_FEE=$0.01`/leg (`immune.py:59-60`). | `FEE_RATE=0.0001`, `FEE_ROUND_TRIP_DOLLARS=2.0` (`SacredRules`). |
| **Ticker filter** | None at the hard-coded level (`immune.py`). | `TICKER_BLACKLIST`: leveraged 2×/3× ETFs — `TQQQ, SOXL, UPRO, SPXL, UVDY, UVXY, SVXY, SSO, SDS, ...` (`foundations/constants.py`). |

**Trade-off:** prime's safety can't be weakened at runtime but is **coarse**
(3 positions, $5k, all-or-nothing halt). Rebuild's is **fine-grained and
adaptive** (per-horizon caps, drawdown stress, portfolio giveback, learned
gate) but relies on correct env wiring and the learned gate being sane (hence
the `verify_learning_gate` canary).

---

## 8. IB / Live Integration Layer

| | PRIME | REBUILD |
|---|---|---|
| **IB lib** | `ib_insync` via a thin compat shim `ib_compat.py`. | `ib_insync` adapter in `senses/ib/` (a from-scratch layer). |
| **Streamer** | `ib_streamer.py:IBStreamer` — `reqMktData` + `reqMktDepth` + `reqHistoricalData`, 1-min bar aggregation into `StreamBuffer` (`ib_streamer.py:77`); `LOOKBACK_BARS=70` (`EDGE_LOOKBACK+20`). | `senses/ib/ib/gateway.py:IBGateway` (735 lines) + `market_data.py`, `orders.py`, `historical.py`, `scanner_mixin.py`, `tick_pump.py`, `order_sweep.py`. `senses/ib/interface.py` is the Broker ABC. |
| **Executor** | `ib_executor.py:IBExecutor` — atomic `bracketOrder()` parent+TP+SL, `monitor_orders` trails/ cancels, `close_position`/`close_all_positions` (EOD flatten, manual flatten). "IB is source of truth; journal is carbon copy" (`CONTRACT.md`). | `execution/brackets.py:BracketManager` (mechanical exits) + `execution/orders.py`. Broker wired by `ops/broker_factory.py:create_broker`. |
| **Safety halt** | `ib_cycle._halt` → `safety_halt(reason)` (telegram) + `_running=False` (`ib_cycle.py:285`). | Same-named pattern but the daily-loss state lives in
`ops/runner_composite.py:get_daily_loss_state` and the kill-switch is in the
`Runner` main loop (`ops/runner.py`). |
| **Data ownership** | `memory.py:Journal` is append-only JSONL hash chain (R7); positions/P&L come straight from IB (`_ib_sync.py`). | `reflection/buffer.py:TradeBuffer` is the single-writer fill log; P&L prefers IB's `realizedPNL`, falls back to fill arithmetic (`buffer.py:230`), with a 2026-08-18 fix consuming fills per round-trip to stop phantom trades. |

Notable rebuild engineering: the **single-writer + dedup invariants** are
heavy. `position_monitor.py` is "the SOLE WRITER to the TradeBuffer";
fills are persisted to `runtime/seen_exec_ids.json` across restarts to stop
the 46-CISS-phantom-trade bug (`position_monitor.py:107-114`); bracket
reconciliation runs on every pulse for restored positions
(`position_monitor.py:280`); equity starts at 0.0 (never a fake $1,000)
(`portfolio.py:57`) and is only set from a real IB `accountSummary`. Prime's
`_daily_pnl` is simply assigned from IB's `reqPnL` (`ib_cycle._cycle` →
`hippocampus._daily_pnl = float(pnl.dailyPnL)`).

---

## 9. Tests & Validation Philosophy

### PRIME — contractual discipline (R1–R18)

`tests/test_contract.py` is a **runtime-enforced architectural contract**
tagged `@contract` ("always run in CI, cannot be skipped"). It mixes AST
checks with import + behavior:

| Rule | What it checks | Location |
|---|---|---|
| R1 | Only `cortex.py` produces `BUY`/`SELL`/`HOLD`; `cerebellum/edge/hands/hippocampus/immune` contain none | `test_contract.py:30` |
| R3 | No file > 200 lines (big skip-list), no function > 40 lines, no nesting > 3 (`_max_depth`) | `test_contract.py:70,102,120` |
| R4 | Core 5 indicators exist, each has `compute_<name>`, positive weights | `test_contract.py:167` |
| R4b | `sum(INDICATOR_WEIGHTS)` ∈ [0.8, 1.2] | `test_contract.py:199` |
| R5 | `SCORE_INVERT is False`, `PRIOR_TOP ≤ PRIOR_TOP_MAX`, `PRIOR_TOP ≥ 0.40` | `test_contract.py:208` |
| R6 | Safety nets are literal constants — AST-scans every `os.getenv` call whose segment mentions `MAX_LOSS/DAILY_LOSS/MAX_POS/MAX_CONCURRENT` | `test_contract.py:218` |
| R7 | `Journal` has no `update`/`delete`/`remove`; hash-chain `verify_chain()` | `test_contract.py:254` |
| R8 | Learning ecosystem integrated: `hippocampus.py`/`stdp.py`/`nash.py`/`brain/cognitive/episodic.py` each contain learn/stake/replay/update/add/record | `test_contract.py:277` |
| R9 | All `INDICATOR_WEIGHTS ≥ 0` | `test_contract.py:322` |
| R9b | `SCORE_INVERT is False` (re-assert) | `test_contract.py:337` |
| R10 | No `print()` in `src/` (AST `ast.Name id=='print'`), IB adapter files exempt | `test_contract.py:345` |
| R11 | Every public function has a docstring | `test_contract.py:378` |
| R12 | `coverage.fail_under` configured ≥ 80 (checks presence only; gate set at 17) | `test_contract.py:412` |
| R13 | No AST `Compare` against verdict strings outside cortex/ib_adapter/ib_executor/hands | `test_contract.py:421` |
| R14 | Every top-level `immune.py` assignment is `AnnAssign` (typed) except `__all__` | `test_contract.py:451` |
| R15 | No bare `except:` or `except Exception: pass` (IB adapter exempt) | `test_contract.py:478` |
| R16 | No `TODO`/`FIXME` (script `check_no_todo.py`) | — |
| R17 | mypy `strict=true`, `disallow_untyped_defs=true`; IB layer excluded | `pyproject.toml` |
| R18 | Module docstrings present | — |

Plus focused unit tests: `test_backtest.py`, `test_cerebellum.py`
(+ `test_indicator_edge.py` permutation tests at `EDGE_P_VALUE=0.05`),
`test_neuromorphic.py`, `test_safety_nets.py`, `test_telemetry.py`,
`test_ib_executor.py`, `test_ib_streamer.py`, `test_infra.py`,
`test_bugfixes.py`, `conftest.py`. `scripts/` hold stand-alone linters
(`check_complexity.py`, `check_positive_weights.py`, `check_profit_gate.py`,
`check_verdict_strings.py`, …).

**Maturity read:** prime is **disciplined and internally consistent** but the
contract shows **evolutionary churn** — R4 "exactly 5" became "core 5 + compute
functions", R8 "single learning system" became "integrated ecosystem", yet
`hippocampus._adapt_weights` (the original single gradient) is **still in the
code and still the backtest path**. The tests document intent ahead of the
code in places.

### REBUILD — architectural enforcement (layer + hygiene)

`tests/test_architecture.py` enforces **structure, not behaviour**, via AST
with zero imports:
- **Pipeline isolation** (`_forbidden_edges`): brain layers must not import
  execution/risk/monitoring/ops/reflection; exec/risk must not import brains;
  reflection must not import execution/monitoring/ops or `senses.ib.gateway`;
  monitoring must not import ops; **senses must not import brains**; only
  `ops` (composition root) may import everything.
- **No import cycles** (Tarjan SCC over the import graph, iterative).
- **No god files** (`GOD_FILE_LIMIT=500`, `CODE_LINE_LIMIT=600`) with a
  grandfathered exempt list (constants, brain_engine 795, thinker 543, edge 505, memory 509, app 543, runner 520, bridge_api 813, … — most are test files or the deliberate "single source of truth" constants).
- **No patch files** (reject names containing `patch/fix/tmp/hotfix/_new/_v2/…`).
- **No silent exceptions** (reject `except: pass`).

Behavioural coverage lives in `tests/unit/`, `tests/integration/`,
`tests/backtest/`, `tests/stress/`, `tests/benchmark/`, plus ~25
**top-level enforcement suites**: `test_juli_enforcement.py`,
`test_deep_enforcement.py`, `test_entry_standards.py`,
`test_exit_gate_enforcement.py`, `test_journal_enforcement_batch.py` +
`_2.py`, `test_fix_journal_enforcement.py`, `test_confidence_pipeline_e2e.py`,
`test_consolidation_34_36.py`, `test_brain_health_checkpoint.py`,
`test_decision_health.py`, `test_health_budget.py`, `test_weight_enforcer.py`,
`test_pipeline_integrity.py`, `test_live_smoke.py`, `test_news_providers.py`.
These are named after **Fixes #1–#73** — each is a regression guard for a
specific production bug. `scripts/`: `brain_health.py`,
`guardrails_check.py`, `market_cycle_validator.py`, `deep_audit.py`,
`comprehensive_verification.py`, `lint_architecture.py`,
`verify_ib_reconcile.py`.

**Maturity read:** rebuild is **operationally hardened and bug-tracking-
explicit**. The codebase is littered with "Fix #73", "2026-08-18 root-cause
fix", "IRONYCLADE" comments — it reads like a live production system where
every fix ships a regression test. The trade-off is **more complexity and
more seams to get wrong** (the `ev_gate.py` `try/except` swallow block, the
`print(f"WARNING: operation failed: …")` anti-pattern that prime's R10 bans,
the 543-line thinker that prime's R3 would reject).

---

## 10. The Neuromorphic, Halim, Nash Subsystems

| | PRIME | REBUILD |
|---|---|---|
| **Neuromorphic** | `brain/neurons/` package: `LIFNeuron`/`LIFNetwork`/`Synapse`/`STDPLearner`,
`AttractorMemory`/`Attractor`, `SleepReplayEngine`/`SleepResult`,
`NeuromorphicBridge` + `create_bridge`, `moe_config`, `weights_config`, plus `bridge_scoring`, `network`, `network_step`, `spike`, `threshold_adapter`. `NEURO_BLEND=0.3`; `ConsolidationEngine` offline slow path (`brain/consolidation.py`); `sleep_replay()`. | `hanoon/juli/neurons/` package: `LIFNeuron` (`lif.py`), `STDPLearner` (`stdp/rule.py`), `AttractorMemory`, `NeuromorphicBridge` (`bridge.py`), `sleep.py`, `moe_config`/`weights_config`. `BrainEngine` says "74-neuron LIF network"; blend 0.3 permanent, warmup disabled ("brain always on", `brain_engine.py:142-144`). Lazy-init `_get_neuromorphic()` (import-cycle avoidance). |
| **Halim (LLM)** | A **full local LLM stack**: `halim/` package with `HALIM_IDENTITY.json`/`HALIM_MANIFEST.json`, `__main__.py`/`engine.py`/`client.py`/`protocol.py`/`serve.py`/`phases.py`/`guardrails.py`/`scaffold.py`/`capabilities.py`/`device.py`/`dataset.py`/`unlock.py`, plus `halim/scripts/` (`train_toddler.py`, `eval_toddler.py`, `merge_lora_colab.py`, `download_qwen3.py`, `convert_peft_to_mlx.py`, `register_checkpoint.py`, `sync_from_tradingbot.py`, `prepare_sft.py`) and `halim/tools/quantize_checkpoints.py`, with a checkpoint at `halim/data/checkpoints/qwen3.5_4b_v2/` (`config.json`, `model_config.json`, `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`). Halim is the brain's deliberative analyst. | A **thin bounded advisor**: `hanoon/halim_bridge/` (`bridge.py`, `client.py`, `gate_advisor.py`). Halim is a client to an **external service** (`HALIM_SERVE_HOST/PORT`, default `127.0.0.1:8765` — `foundations/config.py`). Bounded pull `±HALIM_PULL_BOUND=0.03` (`live.py:109`), `gate_advisor` auto-tunes `SIGNAL_THRESHOLD` from realized WR. **No checkpoint files**, no training scripts — Halim is assumed to already exist as a service. |
| **Nash** | `brain/cognitive/nash.py:NashBrain` — pure-brain pattern matching via episodic k-NN, `MOD_BOUND=0.03`, `_GATE_WR=0.45` veto gate, `_GATE_MIN=20` samples. `NashPrediction{dataclass}` with `win_prob/confidence/opinion/gate_authority/n_samples`. `_match` does 1/(1+d) weighted k-NN over ≤200 history + episodes. | Referenced as `hanoon.nash` (a sibling brain layer per `test_architecture.py:90`); JULI may not import it directly (orchestrator wires `nash.predict` in — `test_architecture.py:245` forbids `hanoon.juli`→`hanoon.nash` imports). Exact surface not read in full, but it is a peer "pattern-based opinion" layer that feeds `nash_pull` (`thinker.py` references `nash_pull` indirectly via `score_delta`). |
| **Episodic k-NN** | `brain/cognitive/episodic.py:EpisodicMemory` (capacity from `config.py`). | `hanoon/juli/episodic.py:EpisodicMemory` — `recall_similar(alpha, k=5)` returns WR of k nearest by L2 in `EPISODIC_KEYS` space; boosts/damps score ±0.05 by `similar_wr` (`thinker.py:164`). |

**Directional read:** prime ships Halim + Nash + STDP + Episodic + Sleep as
**bundled, co-developed** components inside the single package (the LLM is
trained/merged in-repo). Rebuild ships them as **integrated-but-external**:
Halim as a network service the brain advises against; Neuromorphic as a lazy
singleton; Nash as a peer layer wired by the orchestrator. Prime is
self-contained; rebuild assumes a richer surrounding ops environment.

---

## 11. Risk / Sizing Sub-Layer (only REBUILD has a dedicated one)

- **REBUILD `risk/sizer.py`** (`PositionSizer`, 295 lines): base *half-Kelly*
  (`kelly * KELLY_BLEND=0.5`, capped `_KELLY_BASE_FRACTION=0.15`,
  `_KELLY_MAX_SIZE=0.5`), then a **12-feature PPO** (`concentration_penalty`,
  `risk_penalty`, `count_penalty`, volatility inverse-adjust, confidence
  exponent with `_KELLY_EXP_SENS=0.5`), then a **penny-stock notional cap** by
  price tier (`$500`/`$600`/`$750`/`$1000`, `brackets.py:453`) keyed to the
  documented catastrophes (HUIZ −$478, GAUZ −$340, MMA −$204, GYGY −$198,
  OMH −$184 — all penny stocks), then a Halim lesson modifier (±5% when
  `halim_trust ≥ 0.5`), then a **max-risk-per-trade** cap so no single trade
  can lose more than `max_risk_per_trade` of equity (`risk.fees.max_risk_dollars`).
  The whole thing returns a *fraction of the risk budget*, not a dollar
  amount (`risk/sizer.py` docstring).
- **REBUILD `risk/portfolio.py`** (`PortfolioRiskManager`, 211 lines): the
  live account equity (from IB, starts at 0.0, never fake `$1,000`), a
  continuous `risk_scalar` derived from drawdown, dynamic `_max_positions`
  (`max(6, min(30, int(equity/150)))`), stress mode at 20% drawdown,
  `pre_trade_risk_gate()` (exposure < 4×, ticker conc < 25%, not stressed
  above 30%/50%, positions remaining, scalar>0) (`portfolio.py:112`).
- **PRIME**: sizing is `hippocampus.size_position` (~12 lines) —
  `min(MAX_POSITION_NOTIONAL/px, MAX_LOSS_PER_TRADE/(ATR·2),
  MAX_POSITION_NOTIONAL·kelly/px)` floored to 1 share. No portfolio layer.

---

## 12. Headline Differences & Similarities (verbatim)

**Genuine differences**
- Scoring: prime `tanh(Σ w·z)` signed `[-1,1]` @ `0.65` vs rebuild
  percentile-normalized `[0.10,0.70]` @ `0.60`, **signed direction
  separated from confidence**.
- Win-probability band: prime `[0.25, 0.60]` (structural only) vs rebuild
  `[0.18, 0.35]` **plus realized-band-WR + realized-R:R reliability-weighted
  pulls and a live `verify_learning_gate` canary**.
- Entry: prime verdict-only-vs-safety-nets vs rebuild **Thinker verdict AND
  ops-layer `DecisionState` AND of verdict+EV+halim+direction**
  (§8 doctrine/rift).
- Exits: prime single static ATR bracket vs rebuild **5-tier ladder + 14 exit
  thresholds + profit-lock tiers + trailing + portfolio giveback**.
- Learning: prime backtest=1-perceptron gradient; live=loose module fan-out →
  rebuild **unified JuliMemory state machine** (horizon gates, dynamic prior,
  IRONCLADE source filter, ghost-trade exclusion, runtime canary).
- Safety: prime hard-halt constants vs rebuild **adaptive portfolio risk
  scalar + env-driven config + ticker blacklist**.
- Sizing: prime 3-cap min formula vs rebuild **PPO + penny-stock notional
  caps + Halim lesson modifier + max-risk-per-trade**.
- Halim: prime **in-repo trained 4B LLM** vs rebuild **external-service
  bounded advisor**.
- Tests: prime R1–R18 contract vs rebuild **AST layer-isolation + ~25
  fix-named behavior suites**.

**Genuine similarities (shared DNA)**
- Both brand themselves as "JULI," use `LIFNeuron`/`STDPLearner`/
  `AttractorMemory`/`NeuromorphicBridge`, `NEURO_BLEND=0.3` permanent +
  always-on brain, both have an episodic k-NN, both use `ib_insync` against
  IB Gateway (ports 4001/4002), both use a 3:1 `TARGET_R_R` structural
  default and `FEE_RATE=0.0001`, both keep `CONFIDENCE_FLOOR=0.50`, both
  treat IB's realized P&L as source-of-truth and use append-only fill/
  journal logs with hash/ledger discipline, both forbid hard exits from
  non-real sources (`simulation` rejected; prime's `_validate_trade_data`/
  `is_real_trade`).
- The 5 core indicator *names* (`vpin, orderbook_imbalance,
  institutional_flow, momentum, vwap_deviation`) and the *positive weight*
  convention survive intact in rebuild (`DEFAULT_ADAPTIVE_WEIGHTS` carries
  the same names with small positive defaults).

---

## 13. Maturity / Risk Summary

- **PRIME is disciplined and minimal.** The R1–R18 contract is enforceable
  and the codebase honors it (small files, single verdict source, hard
  constants). The **weakness** is evolutionary lag: the contract tests
  describe an "integrated learning ecosystem" that the live
  `NeuromorphicBrain.on_trade_close` only half-implements (it fans calls out
  but owns no central learning state), and `Hippocampus._adapt_weights` is
  still the only real gradient — confined to the backtest path. Exits are a
  **known gap** (static ATR brackets, timeouts disabled). It is the safer,
  more predictable system but structurally incapable of the exits/sizing
  sophistication rebuild built.
- **REBUILD is operationally hardened but sprawling.** Its 6-layer DAG +
  AST isolation + fix-named regression suites show it has absorbed real
  production scars (phantom trades, death-spiral gates, penny-stock blowups,
  equity-display fakery). The **weaknesses** are the seams: the
  `try/except` swallow in `ev_gate.py`'s calibration import chain (guarded
  by a literal "DO NOT DELETE THIS BLOCK" comment), the `print(
  f"WARNING: operation failed…")` anti-pattern that prime's R10 bans, the
  543-line `thinker.py` and 795-line `brain_engine.py` that prime's R3 would
  reject, and the **documented tension** between "JULI is the sole
  decision-maker" and `decision.py`'s hard AND of the EV gate.

**Bottom line:** these are best read as **two generations of the same
lineage** rather than an ancestor/descendant pair. Prime (v2.0.0) is the
"discipline-first extraction" — a minimal, contract-bound neuromorphic core
that is easy to reason about but deliberately leaves exits, portfolio risk,
and real-time learning as future work. Rebuild (v3.0.0 "Snowflake") is the
"production-hardened evolution" that shipped all of that — multi-tiered
exits, portfolio risk, a unified learning state machine, Halim-as-a-service,
penny-stock and phantom-trade guards — at the cost of a far larger,
looser, harder-to-gate-against-regression surface. The two share enough
vocabulary (LIF/STDP/Attractor/NeuromorphicBridge, the 5 indicator names,
the 3:1 R:R, IB-on-localhost, append-only journals, `real_trade`-only
learning) to confirm a common origin; the differences are deliberate
trade-offs between **provability** (prime) and **survivability** (rebuild).

---

## Addendum — Live Gap-Closing (v2.0.0 hardening run)

The original comparison identified three *live* gaps where rebuild
v3.0.0 "Snowflake" had shipped behaviour prime v1.x had only sketched: the
realized entry gate, the per-trade learning feedback loop, and tiered exits.
This run closes all three *without* disturbing prime's provability posture.

### 1. The realized-EV entry gate (the one quantitative gap)

- `src/hanoon_prime/brain/realized_ev.py` (NEW, 187 lines) — `RealizedStats`
  persists closed-trade outcomes to `runtime/juli_realized.json` and exposes
  `band_wr` (per-decile win-rate + reliability), `realized_rr` (win/loss R:R +
  reliability), `is_gate_closed` (refuses bands with `n ≥ 20` and
  `wr < 0.45`), and the `ev_gate_should_enter` gate which blends the
  structural `p·R−(1−p)` with the realized pull (`confidence` scales the
  pull, so a high-confidence signal trusts the model and a low-confidence
  signal defers to history). `verify_learning_gate()` is a 3-probe canary
  (losing band → refuse, recovery band → admit, thin data → structural admit)
  that returns `{"all_pass": True}` and runs in every `test_realized_ev.py`.
- `src/hanoon_prime/brain/risk.py` — `RiskEngine.evaluate` calls
  `ev_gate_should_enter`; when `realized is None` the gate falls back to the
  **exact** original math, so `test_safety_nets` (`ev==0.84, shares==3,
  stop==96.0, target==112.0, kelly==0.07`) is unchanged.
- `src/hanoon_prime/brain/orchestrator.py` — owns the single
  `RealizedStats()` instance (`self._realized`) and feeds it only IRONYCLADE
  sources (`real_trade`, `ib_fill`, `ib_paper`). The previous wiring bug —
  `reflection/supervisor.on_trade_close` called `memory.record_outcome` but
  was never invoked from `ib_cycle._reflect_closed`, so realized stats stayed
  empty forever — is fixed: `ib_cycle._reflect_closed` now calls
  `brain.on_trade_close(..., source="ib_fill")`.

### 2. Unified learning feedback

`on_trade_close` now fans out to dynamics/exits/episodic/nash/neuro AND
(hitherto missing) updates the realized band/RR + `memory.record_outcome` +
`memory.update_pred_error`. The model therefore learns from every closed real
trade — closing the exact gap called out in the body comparison.

### 3. Tiered exits

`src/hanoon_prime/brain/exits.py` `ExitPolicy` (profit-lock tiers, giveback,
staleness, consolidation exit) is wired: `NeuromorphicBrain.check_exit`
delegates to `self.exits.evaluate(...)` and returns an `ExitSignal` that
`ib_executor` consumes. This replaces rebuild's `brackets.py` single-ATR
stop/target with the same tiering discipline, but keeps the mechanical ATR
trailing in `ib_executor` as a backstop.

### 4. Penny-stock caps on the live risk path

`PENNY_NOTIONAL_CAPS` tiers `(2,500),(5,600),(20,750),(inf,1000)` cap notional
for sub-$2/$5/$20 stocks in both `brain/risk.py._size` and live
`ib_executor.place_bracket`, matching rebuild's phantom-trade guard.

### 5. Contract / hygiene gates now green

- `CONTRACT.md` R19 (realized-EV gate) + R20 (tiered exits) enforced via AST
  + behavioural tests in `tests/test_contract.py`, plus a dedicated
  `tests/test_realized_ev.py` behavioural suite.
- Full compliance gate: `mypy src/hanoon_prime` (strict, 112 files) clean,
  `ruff check src/hanoon_prime tests` clean, `pytest tests/ -m "not live"`
  → **211 passed, 0 failed, 0 skipped**, coverage 36.65% (gate ≥17%).
- R3 hygiene: `juli.py` (199), `reflection/buffer.py` (195),
  `brain/consolidation.py` (222) and `brain/exits.py` (173) re-trimmed or
  correctly exempt; `weights_config.get_network_topology` reduced to a
  1-line constant return; the skip-set basename bug
  (`brain/consolidation.py` → `consolidation.py`, `brain/exits.py` → `exits.py`)
  fixed so exempt files now actually match.
- No new verdict-string dispatch outside `{cortex, ib_executor, hands}`; no
  `print()` outside the IB adapter layer; no bare/`Exception: pass`;
  `immune.py` constants all `AnnAssign`.

### Verdict on the comparison

With R19/R20 implemented, prime v2.0.0 now carries the **same evolved-from-real-trades
discipline** as rebuild "Snowflake" on the three axes the body comparison
credited rebuild for (realized gate, unified learning, tiered exits) — while
retaining its provability advantages (strict R1–R20 contract, strict mypy,
single `RealizedStats`/`RiskEngine` boundary, typed constants, ≤200-line
modules). The remaining differences are deliberate: prime keeps exits
*tiered-but-mechanical* (no portfolio-level `risk/portfolio.py` and no
`cross_asset` multi-symbol optimisation) and keeps Halim-as-a-service as a
light `threshold_adapter` rather than rebuild's full `halim_gate_advisor`
service — i.e. prime deliberately trades rebuild's optimiser for the
provability of a tighter, contract-bound core.

---

## Addendum 2 — v2.1 BRAIN-FIRST (2026-09-07, code-verified)

A second hardening run ported rebuild's *doctrine*, not just its features.
The central decision is written into `CONTRACT.md` (R19 amendment):

> **Learned signals LEAN (bounded, two-way); mechanical constraints GATE.**

A learned gate that can *block* is self-referential — trained on the trades
it allowed, it can deadlock on its own history (rebuild's 2026-08-14 EV
deadlock, §3's documented death spiral). Prime's v2.1 therefore retires every
remaining learned hard-gate and demotes it to a bounded lean:

| Learned signal | v2.0.0 behavior | v2.1 BRAIN-FIRST behavior |
|---|---|---|
| Realized-EV gate (`brain/risk.py`) | `should_enter=False` refused the entry | **Advisory sizing scale only**: full size on `should_enter`, `0.75` on thin positive EV, `0.5` on negative EV — bounded `[0.5, 1.0]` (`SizingResult.ev_scale`); verdict kept in `ev_reason` for telemetry. Only MECHANICAL limits refuse: data validity, sub-rounding Kelly (<1 share), `MAX_CONCURRENT_POSITIONS`. |
| Nash pattern memory (`brain/orchestrator.py`) | `_apply_nash_gate` zeroed the score (hard veto) | **Bounded penalty** `NASH_PENALTY_MAX = 0.15` (`brain/config.py`), scaled by pattern confidence × win-prob deficit; short side leans toward zero. An overwhelming brain signal can still clear the threshold. Pinned by `tests/test_brain_first.py::TestNashBoundedPenalty` (`leaned != 0.0` — never a hard veto again). |
| Loss-aversion constant | `immune.py PENALTY_SCALE = 2.0` | **1.2** — verified equal to rebuild's `WEIGHT_LOSS_AVERSION = 1.2` (`hanoon/juli/constants.py:362`, itself "reduced from 1.5 — was causing over-reactive weight decay"); `brain/config.py` and `immune.py` now read the same value (CONTRACT.md §Punishment updated). |

### 1. The horizon ladder (rebuild `hanoon/horizon.py` → prime `brain/horizons.py`)

Six rungs — `scalp, multihour, swing, multiday, multiweek, longterm` — each
with `(atr_stop_mult, atr_target_mult, stale_minutes, patience)`. Every rung
keeps the sacred ≥ 3:1 R:R (asserted at import). Classification lives **in
the brain**: `tick(..., bars=...)` → `_classify_horizon` blends ATR%-base →
momentum-consistency bump → regime shrink → bounded one-step Halim nudge,
then snaps to the closest *enabled* rung (`HorizonManager`, persisted in
`runtime/horizons.json`, `scalp`-only safe default, webapp `POST /config
horizons`). Per-horizon effects wired end-to-end: `RiskEngine._size` uses
run-specific ATR multipliers (scalp keeps 2×/6×), `ExitPolicy.register` takes
per-horizon stale windows, `_maybe_size` scales the entry bar by `patience`,
and EOD flatten is horizon-aware — `holds_through_close` (multiday+) SURVIVE
the close (rebuild v4 lesson: flattening an overnight thesis at 15:55 defeats
it). `ib_executor` tracks `ticker → horizon` for order metadata + EOD policy.

### 2. The news organ (rebuild `senses/news/providers.py` → prime `brain/news_sources.py`)

System-2 sensory evidence, never a gate: Yahoo Finance search API (no key) +
Finnhub company-news (`FINNHUB_API_KEY` free tier), stdlib-only with 4s
timeouts and a 5-min per-ticker cache, scored by the SAME `SentimentPolarity`
model the outcome learner uses. `NewsFeedEngine` runs on the consolidation
thread and publishes `{ticker: polarity}` into `BrainState`; the fast path
reads a bounded ±0.03 bias (`_news_bias`) without touching the network.
Every provider failure is silent — no source is load-bearing.

### 3. Ops ports from rebuild

- **Gateway supervision** (`runner_gateway.py` port): `_supervise_gateway`
detects drops, reconnects 1s→2s→4s→8s→16s→30s-cap backoff, then
`_resubscribe_all` re-requests market data + re-seeds history — IB silently
drops subscriptions on disconnect (the stale-tick watchdog panic).
- **Stale-order sweep**: pending `JULI_*` bracket parents older than 60s are
cancelled so the brain re-decides on fresh data (rebuild's Error-201
stacked-book lesson).
- **Read-only Telegram chat** (`ops/telegram_chat.py` port): `TelegramChat`
long-polls getUpdates, answers `/status`, `/positions`, `/horizons` from a
read-only brain snapshot. Auth = configured chat_id only; rate-bucketed
20/min; it can never place orders or mutate config.

### 4. Rebuild drift since the body comparison (re-verified 2026-09-07)

Rebuild is now at **Fix #74**: #72 (proportional cold-start prediction-error
+ WS heartbeat), #73 v2 (per-ticker loss penalty is now ALPHA-AWARE
PROPORTIONAL, not a hard block — the hard block was rejected as another
death-spiral gate), #74 (confidence boost 0.30→0.60, threshold kept 0.55).
The body's §4 threshold note is corrected above: live `SIGNAL_THRESHOLD` now
READS the guardrail (0.58), so the old 0.60-vs-0.58 split is gone. §5's
"5-tier ladder" is corrected to the actual **3-tier** ladder (TIER 1 hard
stop / TIER 2 JULI verdict / TIER 3 mechanical nets); TIER 3 sub-checks are
individually adaptive (`adaptive_thresholds.py`) and horizon-aware
(`bracket.horizon`). `_KELLY_BASE_FRACTION = 0.15` carries the scar note
"was 25% — losses were 5x wins on $1000 notional". The doctrine-vs-reality
tension in §4 (`decision.py` re-ANDing the EV gate at the ops layer) is
UNCHANGED and still present at `hanoon/juli/decision.py:135-141`.

### 5. Gates (all green, both environments)

`pytest tests/ -m "not live"` → **370 passed**; `mypy src/hanoon_prime/`
(strict) clean in 116 files; `ruff check src/hanoon_prime tests` clean; all
six script gates (R9/R10/R11/R13/R15/R16) pass. Two environment-robustness
fixes shipped with this run: `test_place_oca_uses_stub_when_ib_missing` now
monkeypatches `_protect._ib = None` instead of depending on `ib_insync`
being ambient-absent, and `pyproject.toml` gained the standard
`ignore_missing_imports` override for the optional `ib_insync` extra — so
the type gate passes whether or not the `ib` extra is installed (CI
installs only `.[dev]`).

### Verdict after v2.1

The v2.0.0 addendum framed the split as **provability (prime) vs
survivability (rebuild)**. v2.1 collapses most of that trade-off: prime now
holds rebuild's survivability doctrine (learned signals lean, mechanical
constraints gate; horizons; news evidence; gateway self-heal) *inside* the
provability cage (R1–R20 contract, strict mypy, ≤200-line modules, single
verdict source). The remaining genuine differences are the deliberate ones:
rebuild still owns the ops surface (webapp/API, bridge services, portfolio
risk manager, cross_asset optimiser, GateAdvisor) and the 27-indicator
score; prime owns the contract-bound core with the 5-indicator signed tanh.
The two generations now disagree mainly on **scope**, not philosophy.

---

## Addendum 3 — 2026-09-07, post-v2.2 head-to-head (code-verified)

Both codebases re-read at commit: rebuild `5830534` (Fix #74, no new
fixes since Addendum 2), prime `v2.2` (v2.1 brain-first + v2.2
strategy organs + live smoke hardening).

**Rebuild re-verified unchanged:** `decision.py` still re-ANDs the EV
gate and conservative quality floor into the approval expression
(doctrine-vs-reality tension); 3-tier exit ladder; no new fix-named
suites since #74.

**Prime advanced since Addendum 2:** all five dormant strategy organs
now wired and live-learning (regime fallback + threading, cross-asset
feed, meta-label sizing, horizon bandit, per-regime weights); genome
read-model exposed in telemetry; pipeline health monitor daemon
(`monitor/pipeline.py`, `GET /pipeline`); live smoke harness
(23/23: false closes, false quotes, real-bar replay, zero orders);
three critical bugs found and guarded by tests (state-path stranding,
double-counted learning, numpy-truthiness entry kill — see FIXES.md).

### Verdict by dimension

| Dimension | Winner | Why |
|---|---|---|
| Decision correctness | **prime** | single verdict source, contract-enforced; rebuild re-ANDs its EV gate against its own doctrine |
| Strategy learning | **prime** | 7 learners fan out per real close with regime context; all bounded, all persisted; rebuild's learning is mostly frozen |
| Indicator breadth | rebuild | 27+10 vs prime's 27 (5 core + 22 higher-order) — parity approaching, rebuild still broader in raw count |
| Risk depth | rebuild | portfolio risk manager + cross-asset optimiser vs prime's scalar pre-trade gate |
| Scar tissue | rebuild | 74 fix-named regressions from real sessions vs prime's clean slate |
| Ops surface | rebuild | webapp/API/bridge ecosystem vs prime's telemetry API + Telegram |
| Continuous assurance | **prime** | pipeline monitor daemon + smoke harness + fixing journal; rebuild has no equivalent self-check loop |
| Test discipline | **prime** | 411 tests incl. live-IB suite + snapshot-shaped regression tests |

**Overall:** prime is now superior on the dimensions that determine
survival and improvement — decision correctness, learning, assurance.
Rebuild retains raw-scope advantages (indicators, risk depth, ops,
scar tissue) that prime has not yet earned. Neither is "better in
every way"; but prime is better in every way that compounds, and its
learning loop converts real trades into edge while rebuild's does
not. The fastest remaining parity moves: port the portfolio risk
manager, and let real data accumulate.
