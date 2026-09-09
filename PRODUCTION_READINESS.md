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