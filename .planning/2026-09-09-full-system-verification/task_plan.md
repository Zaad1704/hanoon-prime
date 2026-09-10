# Full System Verification — hanoon_prime (JULI)

## Goal
Deep, evidence-based audit of every subsystem of the live trading system (bot, brain, IB, HALIM, Telegram, webapp/tunnel, watchdogs, persistence/learning, safety) — confirming each is genuinely working, not just present. No guessing: every conclusion must be backed by log evidence, process state, or code wiring. Then monitor continuously during RTH and report.

## Phases
### Phase 1 — Process & Inventory (complete)
- All expected processes alive: bot (90253), IB Gateway (39696), halim serve (63509), cloudflared (63642), ib_gateway_watchdog (63325), overnight_monitor (63695), monitor.command.
- Logs dir inventoried; primary live log = logs/hanoon_prime.log (15.9MB / live @20:00).

### Phase 2 — Log deep-dive since deploy (19:56) — in_progress
- Categorize every ERROR/WARNING raised since restart; classify benign vs real.
- Verify core loop cadence (CYCLE/SCREEN), THINK/verdicts, brackets, fills, reconcile.
- Confirm no repeated exceptions/timeouts/connection loss.
- Verify HALIM modifier actually pulling live values (not zeros / not fallback).
- Verify telegram delivery clean.
- Verify journal append activity (journal_entries growing: 63152→63601).

### Phase 3 — Subsystem-by-subsystem evidence
- IB connectivity / fills / orders / brackets / sweep / reconcile / fill-confirmed accounting (executor hook).
- Brain: orchestrator, consolidation, thinker, halim_adapter, meta subsystems all firing.
- Safety/policy state (disabled-by-design, togglable).
- Feed health (latency spikes, gaps).
- Telegram notifier + chat.
- HALIM serve process + its own log.
- Webapp/tunnel (health route, cloudflared metrics endpoint).
- Watchdogs/overnight_monitor logs.
- Learning/persistence (journal, exits-tracker, memory) writes occurring.

### Phase 4 — Cross-state consistency
- Health positions vs bot book/live plan.
- Heartbeat advances; no stale beat.
- Exits-tracker content sanity vs open positions (fill-confirm accounting live).
- Daily P&L tracking active (safety disabled).

### Phase 5 — Continuous monitoring window
- Sample metrics over ~10 min: cycle errors=0, heartbeat advances, position count stable ±, telegram send failures=0, feed gaps=0.
- Watch for anything new breaking.

### Phase 6 — Report
- Per-subsystem verdicts + evidence, any anomalies found, recommended follow-ups.

## Next Step
Run Phase 2 log categorization.