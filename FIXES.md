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
