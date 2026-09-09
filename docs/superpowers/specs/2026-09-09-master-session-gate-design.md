# Master Session Gate + Pre-Market Activation

Date: 2026-09-09
Status: Approved design (implementation follows via writing-plans)
Related: `docs/superpowers/plans/2026-09-09-brain-first-decision-pipeline.md`

## Problem

1. **Execution is hard-coded to RTH.** `monitor/sleep_manager.py:18-21` defines the
   trading day as 09:30–16:00 ET, and `ib_cycle.py:382-385` gates every ENTER
   execution on that flag. Pre-market (and post-market/overnight) sees no orders,
   no matter how strong the brain's signal.
2. **Session flags are decorative.** `TRADING_CONFIG.session_*` (trading_policy.py:21-24)
   only veto individual verdicts in `orchestrator._apply_fast_gates`, and even that is
   broken by a session-id mismatch: SleepManager returns `RTH` / `pre_post` /
   `overnight`, but TradingConfig keys are `pre_market` / `rth` / `post_market` /
   `overnight`. `.is_session_active("RTH")` silently falls back to `True`.
3. **Nothing sleeps.** The brain scores 24/5 and HALIM serves 24/5. Deactivating a
   session does not stop the rest of the system.

## Goals

- **Pre-market + RTH are the only trading sessions** (04:00–16:00 ET, weekdays).
  Post-market, overnight, and weekends: the entire system sleeps.
- **One gate governs everything.** When the current session is disabled, the bot
  stops all brain cycles (scoring/verdicts/orders) and stops consulting HALIM; the
  HALIM server stops inference; only telemetry stays up for operators.
- **Pre-market entries actually size.** Equity must be resolvable pre-market so
  ENTER executions get non-zero sizing.
- **Fail-safe semantics.** Any ambiguity pushes the system toward sleep, never
  toward trading.

## Design

### 1. Unified session identity (`monitor/sleep_manager.py`)

Classify every ET window with exactly TradingConfig's keys:

| window | time (ET, weekday basis) |
|---|---|
| `pre_market` | 04:00–09:30 |
| `rth` | 09:30–16:00 |
| `post_market` | 16:00–20:00 |
| `overnight` | 20:00–04:00 |
| `weekend` | Sat + Sun (never active) |

- `get_state(now: datetime | None = None) -> SleepState` gains an injectable clock
  so window classification is unit-testable. `SleepState` keeps
  `active / session / reason` fields.
- Add `effective_state(enabled, now=None) -> SleepState`:
  `active = time_window_active AND enabled.is_session_active(session)`.
  This is the single gate used by every consumer. `enabled` is anything exposing
  `is_session_active(session)` (TradingConfig satisfies it).
- Harden the identity contract: SleepManager's `session` values must equal
  TradingConfig's `session_*` attribute suffixes exactly (`pre_market`, `rth`,
  `post_market`, `overnight`). A module-level constant `SESSION_IDS` documents the
  canonical set (no capitalized `RTH`, no legacy `pre_post`) to prevent
  the silent-fallback bug from returning.

### 2. The pivot — enabled-set controls the whole system

- **ib_adapter.run loop** (the `while self._running` spin at ib_adapter.py:112):
  each iteration computes `effective_state(TRADING_CONFIG)`.
  - **Active:** run `self._cycle(poll, pnl)` exactly as today.
  - **Asleep:** do NOT call `_cycle`. Instead: keep gateway supervision live (crash
    watchdog), keep subscriptions/tickers flowing (bars stay current so wake at 04:00
    is instant), update the telemetry heartbeat, and sleep the normal poll gap. Log
    the active→asleep / asleep→active transition once each direction.
- **`ib_cycle._run_brain_cycle`** passes `session=mkt_state.session` (already wired);
  with unified ids, `_apply_fast_gates` `session_disabled` veto now fires correctly
  if a session somehow runs while disabled.
- **`ib_cycle._finish_cycle` line 385** — `market_open` becomes
  `effective_state(...).active`. Pre-market ENTER verdicts now reach `_execute_verdict`,
  which already guards on live bid/ask (`tk.hasBidAsk`), open positions, and sizing>0.
- **Exit behavior unchanged:** `_finish_cycle` closes positions regardless of session
  (gap protection remains 24/5).

### 3. Deep sleep behavior (summary of "asleep")

While asleep the bot performs: no brain cycles, no scoring, no verdicts, no orders,
no broker sync/backfill, no HALIM consultation. It performs: gateway supervision,
subscription keep-alive, telemetry heartbeat, poll-gap sleep. This makes deactivated
sessions a real sleep instead of a gated run.

### 4. Pre-market sizing — equity fallback chain

`BotCycleMixin._publish_account_feed` resolves equity in order:

1. IB `accountSummary` `NetLiquidation` when > 0 → persist `{ts, equity}` to
   `runtime/equity_cache.json` (atomic write).
2. Local sum: `CashBalance` from accountSummary + Σ `read_portfolio` `marketValue`.
3. Cache file (return last known good).
4. Otherwise equity stays unknown → `equity_synced=False` → sizing veto persists
   (unchanged safety).

PnL remains clamped (nan→0.0, already committed in `ee93dc9`). `/risk` keeps the
`equity` / `equity_synced` fields; semantics: `equity_synced=True` for any real
value (chains 1–3: live NetLiq, local cash+portfolio mark, or last-good cache).
Only chain 4 (truly unknown) stays unsynced, so the sizing veto still owns the
no-data case.

### 5. HALIM sleep — polled gate (`halim/halim/serve.py`, separate process)

- New `SessionPoller` thread: every `HALIM_SESSION_POLL_SEC` (default 20) fetches
  `HALIM_SESSION_URL` (default `http://127.0.0.1:8080/session`) with stdlib urllib,
  parses `{active, session}` into a thread-safe shared state.
  **Fail-safe:** any HTTP error / timeout / unparseable body → `active=False` (asleep).
- Handler behavior while asleep:
  - `/health` → adds `asleep: true` and `session`.
  - Inference routes `/v1/complete`, `/v1/complete-thinking`, `/v1/chat`,
    `/v1/generate`, `/v1/evolve`, `/v1/record` → HTTP 503
    `{"ok": false, "reason": "system_asleep"}`.
  - Read-only `/v1/status`, `/v1/runtime`, `/v1/manifest`, `/v1/stats`,
    `/v1/unlock` stay available for ops.
- Worker threads idle during sleep; model stays loaded (fast wake). No new work
  enqueued while asleep.
- Gate is polled, not pushed: HALIM's sleep can lag the session edge by up to
  2×poll (≤40s). Acceptable; documented.

### 6. Telemetry

- New `GET /session` → `{session, active, enabled, reason, ts}`. Source of truth for
  HALIM + operators; `enabled` = the live TradingConfig sessions dict.
- `/health` and `/pipeline` gain `session` and `session_active` fields.
- `/config` unchanged — POST `{"sessions": {...}}` already flips the flags live
  (telemetry.py:106-111), and the whole stack now reacts by sleeping/waking.

## Testing

- `sleep_manager`: window classification at boundaries (03:59/04:00, 09:29/09:30,
  15:59/16:00, 19:59/20:00), weekend, `effective_state` with each session flag off.
- Loop gate: inactive state → `_cycle` never called; transition logged once.
- Equity fallback: all four branches + cache round-trip + bad cache file.
- Telemetry `/session` shape and `/health` fields.
- HALIM: poller (ok, timeout, garbage, connection-refused → asleep); handler 503 on
  inference routes while asleep; 200 when awake.
- Contract R-checks (ib_cycle ≤40-line functions, no print, etc.), mypy `--strict`
  via pre-commit, full suite green. Everything shipped through the repo's TDD flow.
- Live restart (paper): at 04:00–09:30 ET the bot must emit + execute pre-market
  ENTERs with sized brackets; after 16:00 ET the entire stack (incl. HALIM) must be
  observably asleep; at 04:00 it wakes unaided.

## Non-goals

- No strategy/parameter changes (signals, thresholds, horizons).
- No HALIM unload-on-sleep (model stays warm).
- No overnight/post-market trading activation (explicitly disabled).