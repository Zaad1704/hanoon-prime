# Hanoon Prime — Production Readiness

Purpose: turn "fixing again and again" into a defined, evidence-based exit
to real money. Nothing in this file is a promise that the system is bug-free;
it is a checklist whose boxes require *evidence* before we claim readiness.

Status of this document: tracking. Last updated 2026-09-10.

---

## 1. Current posture

- Account: **IBKR PAPER** (real money NOT yet connected).
- Deployed commit: `788a000` (on `main`; also pushed to both remotes).
- Runtime: PID restarted 01:05 local; telemetry `GET /health` →
  `{"status":"ok","connected":true}`; SSE + snapshot working.
- Safety net: **disabled by design** (`enabled=false`, telemetry
  `GET /safety-net`). Do NOT flip it on without a decision recorded in §5.
- Yesterday's paper P&L: **−$2,627 of ~$240k equity** (−1.1%).
  Non-trivial for one day; must not repeat while soaked.

## 2. Verified baseline (why the recent bug-loop ended)

Checked and green as of 2026-09-10:

- [x] 678 tests pass; total coverage 64.79% (floor 17%).
- [x] pre-commit: all 17 hooks green (ruff/black/isort/mypy/complexity/file-
      length + R1/R4/R5/R7/R8/R9/R10/R11/R13/R15/R16).
- [x] Test hermeticity: conftest `HANOO_MEMORY_FILE` per-test tmp path — test
      runs can no longer write learning episodes/scores into `runtime/`.
- [x] Netting-reversal guard (`bb21b4c`, `_guard.py`): long_only can no longer
      flip a position into an accidental short; verified live (OLB flattened,
      P&L 0.00, account flat after).
- [x] `adapt_threshold` boundaries fixed (`788a000`): raises at `≥0.50`,
      lowers at `<0.45`, both previously unreachable.
- [x] Memory threshold setter clamps to `[THRESHOLD_MIN=0.45, THRESHOLD_MAX=0.70]`.
- [x] Contamination purged: 523 test episodes + 6 polluted score keys removed
      from `juli_state.json`; `juli_realized.json` reset to empty (was ~62%
      fake tallies). Weights vector was never directionally touched (empty
      alpha → zero gradient).
- [x] Security defaults: telemetry binds `127.0.0.1:8080` only.

## 3. Safety toggle — how YOU do it via telemetry (verified working)

No code edits needed. The machine already exposes it:

- Status: `GET http://127.0.0.1:8080/safety-net`
- Enable:  `POST /safety-net` body `{"action":"enable"}`
- Disable: `POST /safety-net` body `{"action":"disable"}`
- Resume from halt: `POST /safety-net` body `{"action":"resume"}`

Semantics you must internalize:

- Enabling is **not** a kill switch. `SafetyProducer.authorized()` only vetoes
  **NEW entries** when a limit trips; it never liquidates open positions and
  never stops the bot.
- Trip limits: daily P&L `< -$200` (`DAILY_LOSS_LIMIT`), `≥3` consecutive
  losses (`CONSECUTIVE_LOSSES_PAUSE`), `>3` open positions
  (`MAX_CONCURRENT_POSITIONS`).
- **Enabling does not survive a restart.** On boot `SafetyProducer` starts
  disabled and re-publishes `enabled=false` on the first policy pulse. If you
  ever want it on, re-apply it after every restart.
- To *freeze* the bot right now (emergency): kill the process
  (`kill <pid>`); optionally enable safety so re-entry on restart is blocked
  while limits are tripped. There is no remote dry-shutdown verb; don't assume
  one exists.

Round-trip proven on the live paper bot 2026-09-10: disable → enable →
disable, `halted` stayed `false`, final state `enabled=false`.

## 4. Open risks (accepted, tracked, or must-fix)

| # | Risk | Severity | Status |
|---|------|----------|--------|
| 1 | Safety off — no automated stop on a losing day or stack | High | Accepted by decision (§5 R1). Re-enable is one POST (§3). |
| 2 | Tunnel `api.hanoonweb.xyz` may expose `:8080` to the public — `/safety-net` has no auth token | High | Must verify (Gate 3). If exposed: add auth or kill tunnel before real money. |
| 3 | Paper fills on penny/OTC-heavy book (NOK, SOXS, OLB, SUNE, VVOS…) mis-represent slippage | Medium | Accepted for soak; real entry must be size-reduced (Gate 2). |
| 4 | Paper daily −1.1% shows the book can bleed one weak day even with healthy hygiene | Medium | Gate 1 requires a bounded-P&L streak to clear. |
| 5 | No watchdog/alerting outside my checks | Medium | Gate 3 deliverable. |
| 6 | Weights absorbed test-age data via counters before the conftest fix (win/loss counts slightly off; realized store reset) | Low | Confined to display stats; decision path is clean post-reset. |
| 7 | Telemetry v2 refresher stall seen earlier (health said `starting` while bot ran) | Low | Not reproduced on latest boot; re-check under Gate 1 soak. |

## 5. Decision log

| Date | Decision | By |
|------|----------|-----|
| 2026-09-10 | Stay on PAPER; do not activate the safety net; user retains the one-POST toggle (§3) | user |
| 2026-09-10 | Reset `juli_realized.json` rather than reconstruct from journal | user |
| 2026-09-10 | Start tracking readiness in this file; Gate 1 soak begins now | user |

## 6. Go-live gates (all must be checked with evidence)

### Gate 0 — Hygiene (done, §2)
- [x] All §2 baseline items green on the exact commit you'll run.

### Gate 1 — Automated runtime ledger (HALIM-verified guardian + bug catcher)

An enforceable, always-on monitor keeps Gate 1 honest **and** catches bugs. It
runs as part of the stack lifecycle — **`scripts/start.command`** brings up
HALIM serve → trading bot → guardian monitor; **`scripts/stop.command`** tears
them down in reverse (monitor → bot → HALIM). The monitor only ever runs while
the bot is up; when the bot is unreachable it idles and, after ~4h of
sustained downtime, exits on its own. No launchd involved (launchd is
TCC-blocked on `~/Downloads`).

    # lifecycle (double-clickable from Finder, or from a terminal):
    scripts/start.command
    scripts/stop.command

The monitor executes `scripts/production_monitor.py --daemon` — checks at
`:00`/`:30` local plus a 23:50 day-finalize. Daemons are pidfile-tracked in
`runtime/pids/` (`hanoon_prime.pid`, `production_monitor.pid`).

**Hierarchy of truth (read-only, never trades):**
1. **Deterministic role → `PASS`/`FAIL` (Cannot be overridden by an LLM):**
   telemetry `/health` must be `ok`+`connected`; `/snapshot` must be non-empty
   (telemetry-stall catch); a fresh `ib_cycle CYCLE` line within 5 min
   (pipeline-stall catch); zero `NETTING GUARD` triggers since last restart;
   zero `Traceback` lines; daily P&L must never break `-1.0%` of equity;
   learning state must stay hermetically test-free. Any violation FAILs (exit 2)
   and resets the clean-day streak.
2. **Deterministic role → bug catcher (`anomalies`, streak-neutral):**
   decision path sanity — weighted carriers within `[-2,2]` and non-sparse;
   brain-state fields typed & bounded (`threshold`∈[0.45,0.70],
   `pred_error`∈[0,1], `risk_ceiling`>0); last journal verdicts have finite
   scores and valid actions; non-noise `ERROR` lines since restart →
   flagged problems; `pipeline_incident` rows; `SAFETY HALT` /
   `LEARN BLOCKED` markers. These do NOT reset the streak but DO escalate.
3. **HALIM** is itself inspected: `halim_state_matches_clock` verifies HALIM
   is awake during market hours and `asleep` post-market (a `down` HALIM maps
   to monitor exit 3 and dominates). HALIM does **not** steer the verdict —
   the manifest is fully deterministic.
4. **User notification:** every hard FAIL, HALIM-down, and FAIL-grade anomaly
   is sent to the Telegram chat via the bot's own `_telegram.send()` (token/
   chat from `.env`) — deduped to one message per signature per calendar day
   in `ledger["alerted"]`. WARN-grade anomalies are recorded in the ledger
   `findings` and surface in the 23:50 EOD digest, never paged. If Telegram
   is not configured it is recorded in the ledger instead, never crashed on.
   See the Inside Man section under Gate 3 for the full manifest contract
   (exit codes, heal cabinet, digest, re-anchor).

**State & reads:**
- Ledger (daemon-owned, gitignored): `scripts/production_state.json`
- Inspect: `scripts/production_monitor.py --status`
- Metrics only: `scripts/production_monitor.py --json`
- Manual check: `.venv/bin/python scripts/production_monitor.py`
- Log: `logs/production_monitor.log`
- Exit codes: 0 PASS · 2 deterministic FAIL · 3 HALIM degraded · 4 anomalies reported

**Current state (as of doc update):** `gate_status=pending`, streak `0/10`,
closes `0/200` — today is FAILing the drawdown rule (paper `-1.09%`), so it
does not count toward the soak. This is the expected, enforced behavior: the
streak resumes only after 10 consecutive clean trading days (Mon-Fri counting).

### Gate 1 — Paper soak (IN PROGRESS start date 2026-09-10)
- [ ] **10 consecutive trading-days** on PAPER with:
  - [ ] zero `NETTING GUARD` triggers,
  - [ ] zero `Traceback` lines in `logs/hanoon_prime.log` attributed to the session,
  - [ ] `GET /health` → `status:"ok"` continuously (no `starting` stall),
  - [ ] daily P&L never below **−1.0%** of equity,
  - [ ] `GATE`/guard events absent from the journal verdict stream.
- [ ] At least **200 real closes** recorded (currently ~587 since journal start;
      the realized store is rebuilding from 0).
- [ ] The safety toggle exercised ≥1× per week to keep the habit + prove the
      code path alive. Always return to `disabled`.

### Gate 2 — Real-money step-down (only after Gate 1)
- [ ] Enable the safety net via telemetry BEFORE the first real-money session.
- [ ] Start at **≤ 1/4 paper size** and a hard daily loss cap of 0.5% equity.
- [ ] Keep it running until it matches Gate 1's stats (or 30 paper-equivalent
      trades) before scaling up.
- [ ] A human reviews every overnight `position_closed` row for slippage vs
      paper expectations.

### Gate 3 — Monitoring & access (can run in parallel with Gate 1)
- [ ] Confirm whether `api.hanoonweb.xyz` proxies `:8080`; if yes, either
      add `Bearer` auth to `/safety-net` + `/config` or turn the tunnel off
      before real money.
- [ ] Hook an alerting watchdog to `GET /health` and `GET /safety-net`
      (status != ok, or safety tripped) that pages the operator, not just a log.
- [ ] Verify telemetry survives a bot restart (no `starting` refresh stall)
      at least 3×.

#### Inside Man — facility inspection (live, evidence-based)

One command runs every joint check against the running stack and returns the
whole picture; the guardian daemon already uses it as its single probe source.
This supersedes the bespoke per-check block described under Gate 1.

    # full picture, every joint, one command:
    .venv/bin/python -m hanoon_prime.inspection manifest        # human text
    .venv/bin/python -m hanoon_prime.inspection manifest --json # machine JSON

**Joints (12):** processes, identity, telemetry, pipeline, session, safety,
memory, purity, execution_oracle, halim, notify — each check is one of the
decision roles below.

**Severity semantics:** `OK` (green) · `WARN` (baseline/idle deviations that
do **not** stop anything) · `FAIL` · `UNVERIFIABLE` (probe could not run — the
manifest never raises). **Manifest FAILs with urgency: any hard FAIL → `FAIL`;
only `WARN`/`UNVERIFIABLE` → `WARN`; all `OK` → `OK`.** Known benign baselines
read WARN until the day's first event: `equity_synced` (before the first IB
equity sync after restart), `enters_minted` (unminted ENTERs), `send_healthy`
(before the first probe send of the day), `sleep_is_expected`/`asleep` HALIM
matches the post-market clock.

**Monitor exit codes (daemon and single pass):**

| Code | Meaning | Effect |
|------|---------|--------|
| 0 | PASS | streak continues |
| 2 | hard FAIL | **streak reset**, gate paused |
| 3 | HALIM down | paged; deterministic checks still ran (down dominates) |
| 4 | FAIL-grade anomalies | paged once per signature/day; streak-neutral |

WARN-grade anomalies never page (they are folded into the ledger `findings`
and appear in the EOD digest), preserving the old "no alert on pre-sync/idle"
behavior. Manifest source of truth: `src/hanoon_prime/inspection/joints.py`.

**Heal cabinet (gated auto-heal, default `HANOON_HEAL` = on, set `0` to
disable):**

| Trigger (report check) | Action |
|------------------------|--------|
| `halim_alive` FAIL | `scripts/halim_start.sh` (launch_detached) |
| `cloudflared_alive` FAIL | `cloudflared tunnel --config ~/.cloudflared/config.yml run` (launch_detached) |
| `gateway_watchdog_alive` FAIL | `scripts/ib_gateway_watchdog.py` (launch_detached) |

Gated to **2 heal actions/day** (`MAX_PER_DAY`); every heal is recorded in the
ledger (`heals.{today,count,events}`). Dry-run first:

    .venv/bin/python -m hanoon_prime.inspection heal --dry-run
    .venv/bin/python -m hanoon_prime.inspection heal            # apply (gated)

**Alerting & daily digest:** hard FAILs, HALIM-down, and FAIL-grade anomalies
page via Telegram (the bot's own `_telegram.send`, deduped one message per
signature per calendar day in `ledger["alerted"]`; unconfigured → ledger only,
never crashed on). Once per day the daemon sends the **EOD digest** at the
23:50 slot (`digest_send`), a single Telegram message with the day's findings;
a freshness-gated `send_healthy` probe (one lightweight reference send/day)
keeps the notify path itself checked.

**Boot re-anchor:** `scripts/start.command` runs `reanchor` between stopping
and starting the bot (logs to `logs/reanchor.log`), re-seeding the journal
chain anchor when it is broken. Chain history before an intentional purge is
**expected** to show breaks; `chain_intact_from_anchor` verifies 0 gaps after
the last `chain_reseed`/purge anchor (currently anchored at the purge break,
seq 94214). Manual fix: `.venv/bin/python -m hanoon_prime.inspection reanchor`.

**Runtime artifacts (gitignored):** `runtime/inspection_ledger.json` is the
daemon/CLI ledger (`scripts/production_state.json` for the monitor ledger);
`logs/reanchor.log` is the re-anchor audit trail.

### Gate 4 — Incident runbook
- [ ] Documented, rehearsed steps (not just this §3 note) for: unexpected
      short, mass stop-run, tunnel exposure, runaway sizing. Rehearsed once
      against PAPER deliberately (e.g. force a halt by POST-ing then resume).

## 7. Rules of engagement (non-negotiable)

1. **Never** set `safety_enabled`/`policy_state.enabled` to true except by the
   user's recorded decision (§5) or an explicit command from the user.
2. Do not "repair" live runtime state without stopping the bot first.
3. Do not disable R7 (append-only journal) or R13-guarded code to "unblock"
   a fix.
4. Re-run Gate 0 in full after any code change to `src/`.
5. The `.planning/` scratch dir is never committed; this readiness doc is the
   canonical tracking artifact.