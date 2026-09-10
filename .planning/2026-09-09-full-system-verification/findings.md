# Findings — Full System Verification

## Evidence key
- All log references since deploy (commit 1ea03c4, bot PID 90253, restarted 19:56 +06, RTH active).

## Subsystem verdicts
1. Bot process — OK. Single instance (90253), deployed commit = HEAD (1ea03c4), 0.6% CPU / 46MB, CYCLE every ~1s, HEARTBEAT every ~60s advancing (open count exactly matches /health: open=8 == position_count=8 at 20:07).
2. IB Gateway/connectivity — OK. Gateway 39696 up; watchdog (63325) no restarts since 18:05 (no events = stable); /health connected:true.
3. Core loop / decisions — OK. Screens every ~1s, decisions ~20+/min (THINK/NASH/LEARN), 69 orchestrator lines in a 3-min sample; risk.evaluate + nash.predict wired in verdict path (orchestrator 438/750/817).
4. Fill-confirmed accounting (new fix) — VERIFIED LIVE. Exits: EXIT SIGNAL (consolidation Np) → bracket close → IB reconcile release (e.g., 20:01:29.627 burst) → Telegram WIN/LOSS → REFLECT src=ib_fill → LEARN. Notifications fire only at fill, not verdict.
5. Order hygiene — OK. ib_order_sweep cancelled 1 stale BUY (NOK LMT 36.0) at 20:01:18; zero rejects/duplicates/insufficient-margin since restart.
6. Protection — OK. Continuous BRACKET BUY placement on new positions (20:02–20:04: GPRO, IRD, ITUB, MIMI, SOXL, SOXS, TSLL, YMAT, SPCX, INTC, NOK, SOXL); ADOPTs on reopen; zero ADOPT fails. Protective/OCA logic validated in _protect.py + ib_order_sweep.py.
7. Brain wiring — OK. orchestrator constructs Hippocampus(safety_enabled=False), NashBrain, RiskEngine, Deliberator, ConsolidationEngine, RegimeDetector/RegimeWeights, CrossAsset; all used in decision/sizing path. S2 consolidation ticks every ~35-40s with regime/halim/thinker/news.
8. HALIM — OK. halim serve PID 63509, model ready (mlx), modifiers streaming (sr_proximity 1.200, bollinger_position 0.500, institutional_flow/trade_intensity rotating every ~30-60s). Startup qwen3_5 type-mismatch warning is cosmetic.
9. Feed — OK. juli_feed continuous; no gaps/resent/resync/reconnect; only transient latency-spike warnings (max 22ms); counts rising/falling with tick activity.
10. Telegram — OK (post-fix). 24 WIN/3 LOSS/7 BREAKEVEN delivered since restart; zero "send failed"/"not configured" after deploy.
11. Webapp/tunnel — OK. api.hanoonweb.xyz/health 200 (also confirms .env fallback path works for the tunneled web UI); cloudflared QUIC connections registered; recent ERRs are transient QUIC/DNS churn (last DNS err 13:11Z), reconnect fine.
12. Watchdogs/monitors — OK. ib_gateway_watchdog alive (no gateway restarts, stable), overnight_monitor alive. NO alert sent today (nothing alarming under its local health probe).
13. Persistence/learning — OK. runtime/*.json all stamped <1min ago (state.json, juli_realized.json total=352 samples, equity_cache 242782.93, juli_state 469KB); journal +~180-200 entries/min (to 64995); R5/R7 invariants = refactor-safe append writes.
14. Safety — OK per user decision. Hippocampus.check_safety_nets early-returns when safety disabled; SafetyProducer enabled=False; /safety-net shows enabled:false halted:false authorized:true; daily_pnl -554 (IB dailyPnL, authoritative) NOT triggering halt (intended).

## Anomalies
A. Day P&L = -$554 (IB authoritative) with safety intentionally OFF. Biggest single risk to user's capital today; most losses pre-date this deploy (mass consolidation-exit RTH-open behavior designed but costly). Flag, not break.
B. News organ alive (Yahoo fetch + polarity works — SUNE scored −0.125) but has contributed news=0.00 in every S2 line (851 lines Sep 8-9). Root cause: snapshot only published when top-5 alpha tickers have non-zero sentiment at 2-min refresh cadence; in practice they don't → news_sentiment never set → 0.00. Tuning/sampling gap, not dead code (proven scoring works).
C. overnight_monitor false alarm: its tunnel probe (urllib → api.hanoonweb.xyz/health) is bot-blocked by Cloudflare (HTTP 403) while real clients (curl 200) — monitor's UA flagged. Its log prints "Tunnel still down" every 10 min. Tunnel actually fine.
D. Pre-fix artifacts (not live): ~590 ERROR today — 569 cancelMktData "No reqId" from the 19:55:30 SIGTERM teardown of old bot; 20× "Cycle error: market_open not defined" at 17:12 (fixed by 363a5ed, confirmed ancestor of running commit). Zero errors since 19:56.

## Continuous sample (20:08–20:10)
0 ERROR added; heartbeat advanced; open=8 stable; journal growing.

## Cognitive audit (Phase 7 — "alive like a human brain?")
Decision organs with REAL runtime data paths into the raw score (verified in _score_pipeline / tick()):
- cortex.evaluate (weights overlaid w/ regime vectors + learned memory weights), nash predictor + bounded penalty gate (writes nash_modifier), neuromorphic neuro score (NEURO_BLEND), calibration nudge (pred_error 0.4537), gate advisor threshold_delta (realized-WR-driven), news bias, regime multiplier + per-regime vectors, halim modifier, horizon classifier + Wilson/Thompson bandit override (trend_down→scalp counts 7.6:15.4), dynamics stabilization, RiskEngine sizing, meta-label size_scalar + advisor tightening, cross-asset modifier.

INERT organs (present, NOT functioning in the loop) — chr-encoded byte scan, definitive:
- Deliberator: constructed orchestrator.py:97; `self.deliberator.` AND `.deliberate(` appear ZERO times anywhere in src → threshold deliberation never consulted.
- episodic_bias: default 0.0 in shared_state; READ in orchestrator:`_get_regime_data` (line 512) but NEVER WRITTEN by any module (byte scan: only shared_state def + orchestrator read) → episodic term in raw score always 0.0. EpisodicMemory accumulates/replays but its bias output never flows in.
- news: functional (probe SUNE → −0.125) but publishes nothing in prod → 0.00 in all S2 lines (publishing gate samples only top-5 alpha with non-zero polarity at 2-min cadence).

Continuous learning (live evidence): meta-label trains per trade (juli_meta_label.json n=200, brier≈0.16, 12-dim weights), horizon bandit Thompson-updates, realized partitions fine-grained per band/conf/regime/horizon (total=200+), advisor/calibration WR-driven, nash record_outcome on close. Supervisor daily/weekly reviews fire once per restart (24h interval never elapses within a daily-restarted process); on_trade_close is observe-only by design (orchestrator is the single writer).

Behavioral profile (proper decisions?): two-sided (14,960 BUY vs 17,713 SELL THINKs today), broad score range (15% at ≥0.9, rest 0.0–0.8), 3,475 abstain markers, risk-sized entries. BUT 66.5% WR with tiny edges (median win R 0.0634 / loss R 0.0200), exits dominated by consolidation reason (285), day P&L −554 (safety off by user choice). Verdict: brain is real, layered, learning — but its dominant output is a mechanical thin-margin consolidation-exit policy, currently net-negative; deliberation + episodic + news organs add nothing to actual decisions today.

## Recommendations
1. Reconsider arming safety (or lower position churn) — day P&L −554 vs −200 limit while disabled.
2. News: lower refresh cadence or relax publish gate (publish even zero-sentiment snapshots and emit S2 from it), or whitelist ever-scored tickers.
3. overnight_monitor: set a browser User-Agent on its tunnel probe.
4. Optional: reduce feed latency-spike warning threshold noise OR treat sub-10ms spikes as info.