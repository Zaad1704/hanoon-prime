# Brain-First Decision Pipeline — Design

**Date:** 2026-09-09
**Status:** Approved for implementation planning
**Companion:** `ace4da7` (race fix) — this change supersedes the layered gates that accumulated on `ib_cycle` after that fix.

## 1. Problem

Today, the answer to "should we buy NVD?" is produced by **two authorities**:

1. `juli.tick()` — the brain (`.brain.tick` → `_evaluate_fast`) emits an entry decision with score, direction, sizing.
2. `ib_cycle._finish_cycle` — re-gates that decision through `_can_trade`, `_entry_throttled`, `_entry_safety_cleared`, `_portfolio_gate_and_size`, and a fully silent return at `_exec_decision:658` when the ticker lacks a valid bid/ask.

Only authority #1 logs its reasoning (`THINK %s ... score=...`). Authority #2 logs at INFO for some skips (`SKIP %s open`, `SKIP %s portfolio_risk`) but is **debug-only or fully silent** for others (`sizing=0`, `THROTTLE`, raw `return`). That asymmetry produced a class of production incident where the tape flows, `THINK NVD BUY score=1.000` fires, `d=2` decisions are produced, and **zero brackets** are placed, with no observable reason anywhere in the INFO log.

Beyond observability, the layered gates violate the intended architecture: top-level decision "organs" (throttle governor, safety nets, halt flag, portfolio-risk gate, direction/session filter) live outside `brain/`.

## 2. Target architecture

```
IB (streams, fills, account)  ───────────────────────┐
                                                    ▼
        ┌──────────────────────────────────────────────────┐
        │  juli.py — JuliBrain (the ONLY decision maker)   │
        │  ┌────────────────────────────────────────────┐  │
        │  │  FAST CORTEX  (per-ticker, per-call)       │  │
        │  │  · consult policy_state                     │  │
        │  │  · NaN / no-data validity                   │  │
        │  │  · penny bar, direction/session             │  │
        │  │  · governor (cycle cap + cooldown)          │  │
        │  │  · score/direction/sizing → Verdict         │  │
        │  └────────────────────────────────────────────┘  │
        │  ┌────────────────────────────────────────────┐  │
        │  │  SLOW CORTEX  (ConsolidationEngine loop)   │  │
        │  │  · regime (existing)                        │  │
        │  │  · portfolio risk (equity/drawdown/heat)    │  │
        │  │  · safety nets + halt                       │  │
        │  │  → publishes policy_state → BrainState      │  │
        │  └────────────────────────────────────────────┘  │
        └──────────────────────────────────────────────────┘
                    │
                    ▼  juli.tick() → list[Verdict]
        ┌──────────────────────────────────────────────────┐
        │  ib_cycle — pure orchestration (NO policy)       │
        │  · assemble context                              │
        │  · execute ENTER verdicts via executor            │
        │  · reflect closed → journal → learning            │
        │  · EOD / manual flatten (user commands only)     │
        └──────────────────────────────────────────────────┘
```

**Principle:** `IB → JULI (all processing/decisions) → execution → learning`. No new top-level organs; everything decision-relevant goes inside `brain/`.

## 3. Core contract

`JuliBrain.tick(...)` returns `list[Verdict]` — one verdict for **every evaluated ticker**, never a silent omission:

- `Verdict(ticker, action="ENTER", sizing, stop, target, horizon, reason)` — may place a bracket.
- `Verdict(ticker, action="HOLD", reason)` — no edge, not sized.
- `Verdict(ticker, action="VETOED", reason)` — edge existed but a policy blocked it (cooldown, safety, portfolio risk, direction, penny, no-data).

Exits continue to be produced by the brain (`juli._evaluate_exits`) and are returned in the same `tick` call.

Every verdict is logged at INFO and journaled. A veto is **data**, not absence: the log line includes the stage that vetoed (e.g., `VETOED NVD reason=reuse_cooldown stage=governor`).

## 4. New `brain/policy/` package

All new modules under `src/hanoon_prime/brain/policy/` (R3 caps: ≤200 lines/file, ≤40 lines/function).

### 4.1 `verdict.py`
`Verdict` dataclass: `ticker`, `action` (`ENTER|HOLD|VETOED`), `reason`, `stage`, `sizing: SizingResult | None`, `stop`, `target`, `horizon`, `score`, `direction`. Serializes for journal/telemetry.

### 4.2 `governor.py`
Pacing, not thinking. Owns:
- per-cycle entry budget (cap = `MAX_ENTRIES_PER_CYCLE`), reset at the start of every `juli.tick()`.
- per-ticker reuse cooldown (`ENTRY_REUSE_COOLDOWN_SEC`), keyed by ticker → last-entry timestamp.

API: `Governor.begin_cycle()`, `Governor.may_enter(ticker) -> (bool, reason)`, `Governor.note_entry(ticker)`.

Thread-affinity: called on the fast path only. Holds no locks; single-threaded per loop.

### 4.3 `trading_policy.py`
Decision parts of `TRADING_CONFIG`, brain-owned:
- direction mode (`long_only|short_only|both`) → `is_direction_allowed(side)`.
- session enablement (`pre_market|rth|post_market|overnight`) → `is_session_active(session)`.
- penny bar (`PENNY_PRICE`, `PENNY_SCORE_BAR`) → `is_penny_bar_cleared(ticker, price, score) -> (bool, reason)`.

Telemetry `POST /config` mutates this object (same surface as today, new home).

### 4.4 `portfolio_risk.py`
Absorbed from `monitor/portfolio_risk.py` **verbatim behavior** (equity sync, drawdown → risk_scalar, exposure cap, max-positions, giveback, `pre_trade_risk_gate`, `adjust_size`). Moved file; ownership transferred to the slow cortex, which calls it on its background cadence from the account feed. Fast cortex never mutates it — only reads its published snapshot.

### 4.5 `safety.py`
- Safety nets: daily-pnl loss limit, consecutive-loss pause, max-open-positions.
- Owns the halt flag (replaces `ib_cycle._halted` / `_halt`).
- Emits the halt via existing `safety_halt` telegram path (moved call site) and journals the halt event.
- Slow-cortex driven; lifecycle: `begin_call()`, `on_daily_pnl(pnl)`, `on_consecutive_losses(n)`, `authorized() -> (bool, reason)`.

## 5. Cortex split of policy

| Policy | Cortex | Why |
|---|---|---|
| regime / consolidation / HALIM | slow | existing ownership |
| portfolio risk (equity/drawdown/heat) | slow | reacts to account feeds, not per-tick |
| safety nets + halt flag | slow | portfolio-level, low cadence |
| NaN / no-data validity | fast | per-ticker, per-call |
| penny bar | fast | per-ticker mechanical |
| direction / session | fast | cheap config read per call |
| governor (cycle cap + cooldown) | fast | pacing on the entry path |
| score / direction / sizing | fast | existing fast path |

The slow cortex publishes a single `policy_state` dict into `BrainState` (e.g. key `policy_state`): `{authorized: bool, halted: bool, pause_reason: str, risk_scalar: float, equity_synced: bool, drawdown: float, exposure: float, position_count: int, daily_pnl: float, consecutive_losses: int}`.

The fast cortex reads `policy_state` once per `tick()` and applies per-ticker gates after it.

## 6. `ib_cycle` after the change

Removed entirely:
- `_can_trade` (gates move to fast cortex / trading_policy)
- `_entry_throttled` (→ governor)
- `_entry_safety_cleared`, `_entry_safety_cleared` probe recovery (→ safety producer under slow cortex; probe-recovery decision stays brain-owned)
- `_portfolio_gate_and_size` (→ slow-cortex portfolio_risk + fast sizing)
- `_exec_decision` (replaced by `_execute_verdict`)
- `_check_safety`, `_halt`, `_halted` (→ brain/policy/safety)
- `_sync_portfolio_risk` (→ slow cortex account feed)

Kept (pure I/O / user commands / learning):
- `_cycle` loop, `_finish_cycle` (slim: consume verdicts, execute ENTERs)
- snapshot assembly (`_snapshot`) — context collection, moved/kept as feed into juli
- `executor.sync_from_ib`, `place_bracket` calls, reconcile, position watchers
- `_reflect_closed`, journal writes, `_drain_event_exits`
- EOD flatten, manual flatten, `_check_eod_flatten`, `_check_manual_flatten` (user commands, not decisions)
- `_heartbeat`, pipeline monitor

The account feed (day P&L from `pnl.dailyPnL`, NetLiquidation from accountSummary) is routed into the slow cortex instead of `_sync_portfolio_risk`.

`_execute_verdict(verdict)` executes only `action="ENTER"` verdicts through `self.executor.place_bracket(...)`, registers the position brain-side (`register_position`), attaches watchers, and feeds `governor.note_entry(ticker)`.

## 7. Exits

Unchanged in behavior: `juli._evaluate_exits` uses `brain.check_exit`. Exit signals join the same `tick()` return. `ib_cycle` closes positions for exit verdicts and records the exit reason. No gate relocated for exits — the brain already owns them.

## 8. Telemetry

Endpoints stay. Data source changes:
- `/health` — reads `policy_state` (`halted`) instead of `bot._halted`.
- `/safety-net` — reads `safety` producer state; `POST` toggle mutates brain policy (as today).
- `/config` — reads/mutates `trading_policy` (same surface).
- `/risk` — reads slow-cortex-published portfolio-risk snapshot.
- `/brain` — unchanged.
- **New `/verdicts`** — last N verdicts (ticker/action/reason/stage) to make any silent-stop immediately visible.

Telemetry performs no policy logic; it is a read-only observer of brain state.

## 9. Data flow (one cycle)

1. `ib_cycle` assembles context: positions, pos_info, snapshots, account day-P&L, equity.
2. `juli.tick(context)`:
   a. `Governor.begin_cycle()`
   b. read `policy_state` from `BrainState`
   c. per evaluated ticker: snapshot → fast cortex `_evaluate_fast` → per-ticker gates (NaN → `VETOED no_data`; penny/direction/session via `trading_policy`; governor) → sizing → `Verdict`
   d. `_evaluate_exits` runs in parallel (existing)
   e. return `Verdict` list + exit list
3. `ib_cycle` executes `ENTER` verdicts via executor; closes exits; reflects; journals; learns.
4. Slow cortex (background loop, seconds cadence): regime + portfolio-risk feed + safety → writes `policy_state`.

## 10. Error handling

- NaN/inf in snapshots → `VETOED no_data` (never a crash, never silent).
- Account feed failures → slow cortex keeps last known `policy_state`; `equity_synced: False` quiesces portfolio-risk authorization (same semantics as today's `update_equity` refusal), logged once.
- ConsolidationEngine loop exceptions → existing pipeline monitor alerts; `policy_state` stays frozen on last good values.
- Blocking calls stay off the fast path; the slow cortex owns all account/network-touching policy.

## 11. Testing

Ports (existing tests → new home):
- `test_flow_throttle` → governor unit tests (cycle cap, cooldown, reset-on-`begin_cycle`).
- `test_probe_recovery` → safety producer tests.
- `test_bugfixes` portfolio-risk portions → `portfolio_risk` unit tests unchanged behavior.
- `test_telemetry` → updated for new data sources; new `/verdicts` route test.

New:
- **Silent-block regression test**: build a ticker with valid edge (score above dynamic threshold) but `policy_state.authorized=False` (halted) and assert `tick()` returns `VETOED reason=halted stage=safety`; and with a fresh ticker (no cooldown, authorized) assert `ENTER` is returned with sizing/stop/target.
- Fast-cortex no-data test: NaN bid/ask snapshot → `VETOED no_data`.
- Governor integration: 3 entries in one cycle → third `VETOED reason=cycle_budget`.
- End-to-end: `ib_cycle._execute_verdict` places a bracket only for `ENTER`.
- Full suite: `pytest`, mypy strict, R3 contract (`tests/test_contract.py`), pre-commit hooks, and R8 black (NOT on `ib_cycle.py`).

## 12. Deliverables (non-goals excluded)

- Move `monitor/portfolio_risk.py` → `brain/policy/portfolio_risk.py` (behavior-preserving).
- Add `brain/policy/{verdict,governor,trading_policy,safety}.py`.
- Re-wire `juli.tick` signature/return to `list[Verdict]`; update `ib_cycle` orchestration.
- Move halt flag/safety/account feeds under slow-cortex ownership.
- Add `/verdicts` telemetry route; re-point existing routes.
- Update `docs/ARCHITECTURE.md` to match.

Out of scope: `sleep_manager` stays a top-level clock/data provider (it answers "what session is it now?" — a fact, not a decision); `trading_policy.is_session_active` applies the enablement decision against that fact. Out of scope: learning-pipeline changes, scanner behavior.