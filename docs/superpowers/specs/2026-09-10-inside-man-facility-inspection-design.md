# Inside Man — Facility Inspection Design

**Date:** 2026-09-10
**Status:** Draft (pending user review)
**Type:** Architectural — new `hanoon_prime.inspection` subsystem

## 1. Problem

The operator is blind while the system runs. There is no single place that
answers: *"is every component, joint, and mechanism working as designed, and
is nothing broken?"* Current tooling ships partial answers spread across
several places:

- `scripts/production_monitor.py` — guardian hard-rules + bug-catcher
  anomalies, HALIM diagnosis, Telegram notify, exit-code verdict.
- `Journal.verify_chain()` — journal tamper-evidence (hash chain).
- `telemetry` `/health`, `/snapshot`, `/positions` — live process surfaces.
- ad-hoc `pytest` runs — code-vs-spec correctness.

These are disjoint, differing in what they measure, hard to assemble by hand,
and they leave joints unverified (process identity, execution truth,
session-vs-clock expectation, digest of everything).

**Goal:** one library that verifies every joint, consumed by (a) the existing
background daemon for continuous detection + healing, (b) a daily digest to
Telegram, and (c) an on-demand command that prints the same manifest a human
can read and trust. Same checks, three faces — the operator can always confirm
what the daemon saw.

## 2. Design constraints (from the gate)

- **Read-only by default.** Checks only inspect. Nothing in this subsystem may
  mutate bot state, trades, weights, or the journal.
- **Gated auto-heal only.** Auto-actions are limited to mechanical,
  non-trading services (halim serve, cloudflared, gateway watchdog). Everything
  touching trades, money, or learning is report-only. `tldr`: eyes + gated
  mechanical heal.
- **No `print()` in `src/`.** CLI output lives behind `__main__`.
- Reuse existing machinery: `hanoon_prime._telegram.send`, the `_alert`/ledger
  dedupe pattern in `production_monitor.py`, `scripts/launch_detached.py`.
- Preserve `production_monitor.py` exit codes (0 PASS, 2 FAIL, 3 halim down,
  4 anomalies) and its streak/gate1 bookkeeping.
- Monorepo hygiene: gitignored `scripts/production_state.json`, `runtime/`,
  `logs/` stay ignored; `.planning/` never staged; push to both remotes.

## 3. Architecture

```
hanoon_prime.inspection/
  __init__.py   # public exports
  checks.py     # CheckSpec, CheckResult, run_check (harness)
  ctx.py        # InspectionContext — paths, urls, env, git head, pidfiles
  joints.py     # JOINT_* check functions + run_all(ctx) -> Manifest
  digest.py     # build_digest(manifest, ledger) -> str
  heal.py       # heal registry + gated runner (dry-run capable)
  __main__.py   # `python3 -m hanoon_prime.inspection manifest [--json]`
```

Consumers:

- `scripts/production_monitor.py` imports `run_all` and maps results →
  hard-fail set (exit 2 + streak reset), halim-down (exit 3), anomaly feed
  (exit 4 + HALIM + notify). Daemon loop and slot scheduling stay as-is; the
  metrics dict is replaced by the manifest.
- `start.command` gains one optional step: re-anchor the journal chain during
  stop→start (the only writer-safe moment). `stop.command` unchanged.
- Daily digest: sent by the daemon at the 23:50 slot, once per day.

### Data flow

A probe tick: `run_all(ctx)` → list of `CheckResult` → `Manifest{ts, git_head,
pid, results}`. Daemon aggregates: any `FAIL` in the hard set → violation;
any `WARN`/`FAIL` on anomaly-flagged checks → bug-catcher feed; digest
composer turns the manifest into one Telegram message; `heal.py` reviews the
FAIL set against the gated cabinet.

## 4. Check model

```python
@dataclass(frozen=True)
class CheckResult:
    joint: str          # e.g. "processes"
    name: str           # e.g. "bot_alive"
    status: str         # OK | WARN | FAIL | UNVERIFIABLE
    detail: str = ""
    evidence: dict = field(default_factory=dict)

@dataclass(frozen=True)
class CheckSpec:
    joint: str
    name: str
    fn: Callable[[InspectionContext], CheckResult]
    hard: bool = False      # FAIL => guardian hard violation (exit 2, streak reset)
    report: bool = False    # non-hard failure => bug-catcher anomaly feed (exit 4)
```

**Status semantics (critical, prevents false alarms):**

- `OK` — verified working.
- `FAIL` — *verified broken*. Strong evidence (hard) or reportable finding.
- `WARN` — degraded / anomaly-grade: something is off but not conclusively a
  guard violation (e.g. equity not yet synced post-restart, an oracle
  reconcile gap, one unanchorable window).
- `UNVERIFIABLE` — the check's *instrument* failed (e.g. `/health` unreachable
  because the bot is genuinely down, a needed file absent). The harness wraps
  every `fn` in try/except and converts raised exceptions to `UNVERIFIABLE`
  with the error in evidence. **A broken instrument must never read as
  "working", and a loss of view must never read as a false FAIL.**

Aggregation: manifest status = `FAIL` if any hard FAIL; else `WARN` if any
WARN or UNVERIFIABLE; else `OK`.

## 5. Joint inventory

| Joint | Checks (name → what it verifies) | hard / report |
|---|---|---|
| `processes` | `bot_alive`, `halim_alive`, `monitor_alive`, `cloudflared_alive`, `gateway_watchdog_alive` — pidfile exists, pid alive via `kill -0`, cmdline sane, no stale pidfile claiming a dead pid, exactly one bot process | process death: report |
| `identity` | `bot_from_trusted_checkout` — `git rev-parse HEAD` recorded at boot matches repo HEAD via start marker meta; pidfile owner is the `.venv/bin/python -m hanoon_prime.cli` we launched, not a stray copy | identity break: hard |
| `telemetry` | `health_ok` (status ok + connected), `snapshot_fresh` (non-empty, no `_unreachable`, cache age bounded), `positions_surface` (position_count present) | health/snapshot: hard |
| `pipeline` | `heartbeat_fresh` (last `ib_cycle HEARTBEAT` < 5 min in every session state), `cycle_flows_when_active` (session_active ⇒ recent `CYCLE`), `sleep_is_expected` (post-market ⇒ `session_active=False`, heartbeat-only, zero orders/cycles after `SESSION SLEEP`) | heartbeat: hard; sleep: report |
| `session` | `state_matches_clock` (session name sane for local time; overnight state accepted), `positions_reconciled` (IB `/positions` count vs `positions_open` within tolerance; residual pre-adoption IB positions tolerated and explained, not flagged as trading) | report |
| `safety` | `no_netting_guard`, `no_traceback`, `no_safety_halt`, `no_learn_blocked` (since last `ib_adapter Starting`), `policy_flags` (enabled/halted/authorized sane), `drawdown_bound` (daily_pnl ≥ −1.0% of equity, **armed only when equity > 0** so it re-arms after the first equity sync post-restart) | guard/traceback/drawdown/halt: hard; others: report |
| `memory` | `journal_grows` (count monotonic since last tick), `seq_forward` (no gaps in recent window), `verdicts_valid` (sample 60: actions in `{BUY,SELL,HOLD,VETOED,PASS,OPEN,CLOSE,ENTER}`, no NaN scores), `chain_intact_from_anchor` (**anchor = last `chain_reseed` entry, else the last hash-break found by a backward scan; verify forward from it** — today that anchor is the purge fingerprint at seq 94214, so post-purge data verifies clean immediately; a re-anchor advances the anchor to the `chain_reseed` entry so the full chain can be re-verified fresh) | verdicts/chain: hard |
| `purity` | `no_test_episodes` (no `T`/`TEST` in `juli_state.episodes` or journal), `weights_finite_in_band` (no NaN/inf, within [−2,2], ≥10 values), `brain_fields_bounded` (threshold ∈ [0.45,0.70], pred_error numeric∈[0,1] when present, risk_ceiling>0, positions_open non-neg int) | purity/weights: hard |
| `execution_oracle` | `enters_minted` (every recent `ENTER` verdict has a matching open/position event for the ticker), `closes_reconciled` (recent `position_closed`/`exit` events have fills/P&L), `equity_synced` (**equity/equity_synced true after restart; None equity → WARN until first sync**, never FAIL) | report (reconcile gaps are anomalies, not guard violations) |
| `halim` | `halim_state_matches_clock` (`/v1/complete` probe → ok/asleep/down; asleep expected post-market; down is degraded), `no_serve_crash_spam` (serve.log not error-bombing since start) | halim down → exit 3 (not hard/streak reset) |
| `notify` | `telegram_configured` (token + chat present), `send_healthy` (a lightweight probe send succeeds). (Digest delivery itself is recorded under `ledger["digest"]`, not re-checked in the manifest — otherwise the manifest would sit WARN all day before the EOD slot.) | report (telegram unconfigured is WARN, not FAIL) |

Each check keeps a `detail` and an `evidence` dict sized for a 4096-char
Telegram chunk and a human CLI read. No check shells out to `kill`, `ps`, or
git — it calls `os.kill(pid, 0)`, the pidfile/cmdline map from `ctx`,
`subprocess` only for `git rev-parse` (cached in ctx per tick).

## 6. InspectionContext (ctx)

Resolved once per tick from the same constants `production_monitor.py` uses:

- paths: `LOG_PATH`, `STATE_PATH`, `REALIZED_PATH`, `JULI_STATE_PATH`,
  `JOURNAL_PATH`, `STATE_FILE`, pidfile dir, `launch_detached.py` path.
- urls: telemetry `http://127.0.0.1:8080`, HALIM `http://127.0.0.1:8765`
  (env `HALIM_URL` override).
- env: `HANOON_HEAL` (default `on`), Telegram vars via `_telegram`.
- pidfile → expected cmdline map (the booted stack's contract).
- monotonic journal count tracker (owned by the daemon; persisted in
  `scripts/production_state.json` so restarts don't reset it).

## 7. CLI manifest

```
python3 -m hanoon_prime.inspection manifest          # text: joint-by-joint, OK/WARN/FAIL + detail + key evidence
python3 -m hanoon_prime.inspection manifest --json   # machine-readable manifest (same data as daemon tick)
python3 -m hanoon_prime.inspection manifest --heal --dry-run  # simulate gated heals, no side effects
```

Exit code mirrors the daemon's verdict (0/2/3/4) so the manifest command
doubles as a CI gate. No `print()` outside `__main__`.

## 8. Digest (Telegram, once per day, 23:50 slot)

One message composed from the manifest:

```
🛠 INSIDE-MAN DAILY | <date> | HEAD <short>
joints: 11 OK, 0 WARN, 0 FAIL
✅ processes · ✅ identity · ✅ telemetry · ✅ pipeline · ✅ session
✅ safety · ✅ memory · ✅ purity · ✅ execution_oracle · ✅ halim · ✅ notify
heal: <none | list of auto-actions taken + ts>
trades: <closes today> · P&L <$> · equity <$> · streak <n> · gate1 <status>
watch: <top findings, else 'nothing unusual'>
```

Sent via `_telegram.send`; recorded under `ledger["digest"]` with `last_day` +
`ts` to enforce once-per-day. On policy alert events it stays quiet — the
digest is the daily temperature, findings already notified individually are
not re-spammed unless still red at EOD.

## 9. Healing cabinet (gated, non-trading only)

`heal.py` registry: each entry is `{name, condition(ctx)→bool, action(ctx),
max_per_day=2}`. Runner:

- Config gate: `HANOON_HEAL=off` disables everything. Default: `on` for the
  cabinet below only.
- Condition evaluated only against `FAIL`/`UNVERIFIABLE` of joint `processes`
  for the specific service. No heal ever fires on trades, weights, P&L, or the
  journal.
- Action uses `scripts/launch_detached.py` with the service's pidfile + log,
  mirroring `start.command`'s launch lines.
- Every action is logged to the monitor log AND `ledger["heals"]` AND a
  one-line Telegram via `_notify`. Counted per day; exceeding `max_per_day`
  disables that heal until the next day (flap prevention).
- `--dry-run` reports what would run; tests exercise decisions, never actions.

**Cabinet:**
1. `restart_halim_serve` — halim pidfile dead / `/health` down longer than 2
   consecutive ticks while session is expected-active.
2. `restart_cloudflared` — cloudflared pidfile dead (tunnel is operator-facing
   reading, not trading).
3. `restart_gateway_watchdog` — ib_gateway_watchdog daemon dead.

**Not in the cabinet:** bot restart (trading), journal writes (exclusive bot
writer), any money/weight action.

## 10. Journal chain re-anchor (start.command step)

`verify_chain()` can only be `False` after a legitimate out-of-band edit (the
purge/reset performed 2026-09-09, fingerprint: exactly 2 breaks at
seq 93748/94214, everything after re-chains with 0 gaps). Re-anchoring *while
the bot is live* would corrupt seq/hash state the running `Journal` caches in
memory — **one writer only**. Therefore re-anchor happens in `start.command`,
after `stop.command` has torn the stack down and before the bot launches:

1. fresh `Journal(path)` is opened (`_seed` reads the current tail: count+hash)
2. if `verify_chain()` is `False`: append one `chain_reseed` entry
   `{event:"chain_reseed", note, prev_break:seq, from_seq, to_seq}` via the
   same `Journal.append` so seq/hash continue from the real tail
3. the new bot seeds from that tail next launch → future appends chain cleanly
4. daemon's `chain_intact_since_anchor` then verifies from the reseed forward
   (and a full `verify_chain()` pass is achievable after the next re-anchor)

If the chain is `True`, the step is a no-op. If `verify_chain()` cannot be
computed, log it and proceed (never block startup on an integrity check
uncertainty).

## 11. Alerting & dedupe

Reuse the existing `_alert(day, key, text, ledger)` pattern: one Telegram per
signature per calendar day, recorded in `ledger["alerted"][day][key]`. New
signatures: heal actions (`heal:<service>`), digest (`digest:<day>`),
oracle reconcile (`oracle:<ticker>`), unverifiable instruments
(`unverifiable:<joint>`). Same 4096-chunking via `_telegram.send`.

## 12. Error handling

- Every `fn` wrapped by `run_check`; any exception → `UNVERIFIABLE` + evidence
  `{"error": str, "trace": last-3}`. Manifest never raises.
- Top-level `run_all` catches per-check exceptions (already handled by harness)
  and per-*joint* surprises: an entire joint failing to run → each of its
  checks, if it produced no result, emits `UNVERIFIABLE` (never a raise).
- A dead bot: `telemetry` checks → `UNVERIFIABLE`/`FAIL` on health (hard),
  pipeline heartbeat check → `FAIL` after 5 min, everything downstream that
  reads `/health` degrades to `UNVERIFIABLE` — the daemon keeps its existing
  down-cycle behavior (≤ ~4h then self-exit), no FAIL flood, no notify storm.

## 13. Testing

New `tests/test_inspection_*.py` (pytest, existing suite conventions):

- **checks unit tests** with fixtures: temp journal file, fake pidfiles +
  dead-pid case, sample log lines (sleep marker, heartbeat, per-session),
  monkeypatched `urllib`/`os.kill`, crafted `runtime/state.json` variants
  (equity=0, equity None, threshold out-of-band, polluted episodes, NaN
  weights, valid set).
- **session-boundary tests**: the wellbeing of `sleep_is_expected` and
  `cycle_flows_when_active` across pre/regular/post-market fixtures, plus the
  session-bounding marker logic ported from `_bot_start_lines()`, incl. the
  "log has no date" trap.
- **aggregation**: one hard FAIL → manifest FAIL; only WARN → WARN; all OK →
  OK; instrument broken → UNVERIFIABLE ≠ FAIL.
- **oracle**: matching ENTER↔open, reconcile gap → WARN not FAIL.
- **digest**: golden render with/without findings, chunk ≤ 4096, once-per-day
  ledger guard.
- **heal**: dry-run only; condition true → would-run list exact; max_per_day
  enforced; `HANOON_HEAL=off` → empty.
- **monitor refactor regression**: `production_monitor.py --status`, `--json`,
  exit codes, streak roll, gate1 update — preserved via existing test style
  (TDD: write/port tests, then refactor to consume the library).

## 14. Files touched

- new: `src/hanoon_prime/inspection/{__init__,checks,ctx,joints,digest,heal,__main__}.py`
- new: `tests/test_inspection_*.py`
- edit: `scripts/production_monitor.py` — consume `inspection`, keep exit
  codes/roles/slots/ledger
- edit: `scripts/start.command` — chain re-anchor step
- edit: `PRODUCTION_READINESS.md` — document Inside Man + how to use the
  manifest command
- docs: this spec, committed

## 15. Rollout

1. Build `inspection` library + tests; internal TDD loop (tests pass).
2. Refactor monitor to consume it; `--status`/`--json`/exit codes regression
   green.
3. Live dry run: `manifest --json` against the running stack — compare against
   today's evidence (overnight PASS profile, HEARTBEAT age, equity gap, chain
   anchor at the purge fingerprint seq 94214 → `chain_intact_from_anchor` OK).
4. Enable digest at EOD slot; first digest reviewed by operator.
5. Keep `HANOON_HEAL=off` for a 48h soak; then set `on` (cabinet limited).

## 16. Non-goals / deferred

- **Decision replay equivalence** (feed the same tape, diff decisions vs
  simulcast) — separate spec; explicitly out of scope here.
- Full web dashboard — operator chose digest + on-demand manifest.
- Auto-restart of the trading bot, and any autonomous flatten/rollback — held
  by the authority decision (Eyes + gated auto-heal).
- Altering bot/runtime behavior or the journal's append-only writer rule —
  the chain re-anchor is a scripted boot step, not a daemon write.