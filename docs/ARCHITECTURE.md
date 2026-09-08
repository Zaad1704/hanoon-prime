# HANOON PRIME — Architecture (Code-Verified)

> This document is a strict, code-verified map. Every arrow and box
> corresponds to an actual function call, import, or data path read
> from the source. No assumptions.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HANOON PRIME 3.0                            │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │   IB     │   │  Brain   │   │ Executor │   │Monitor/  │       │
│  │ Gateway  │──▶│ (Juli)   │──▶│ (Orders) │   │Telemetry │       │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘       │
│       ▲                                                │           │
│       │            ┌──────────┐                        │           │
│       └────────────│ Telegram │◀───────────────────────┘           │
│                    └──────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Map (Actual Files)

| Component | File(s) | Lines | Role |
|-----------|---------|-------|------|
| **Entry Point** | `cli.py` | 88 | Sets up logging, creates bot, calls `run()` |
| **Bot Core** | `ib_adapter.py` | 171 | `IBStreamingBot` class, owns all components |
| **Main Loop** | `ib_cycle.py` | 646 | `BotCycleMixin`: cycle, brain tick, verdict execution |
| **Data Layer** | `ib_streamer.py` | 334 | `IBStreamer`: market data, DOM, history seeding |
| **Brain Interface** | `juli.py` | 197 | `JuliBrain`: data preamble, voting window, verdict loop |
| **Brain Core** | `brain/orchestrator.py` | 1128 | `NeuromorphicBrain`: decide_entry gates, policy, sizing, portfolio |
| **Decision Policy** | `brain/policy/` | ≤200 ea | `TradingPolicy`, `SafetyProducer`, `ProbeRecovery`, `Governor`: entry gates |
| **Cortex** | `cortex.py` | ~200 | `Cortex.evaluate()`: alpha → score + direction |
| **Risk Engine** | `brain/risk.py` | ~320 | `RiskEngine.evaluate()`: sizing + risk gate |
| **Execution** | `ib_executor.py` | 377 | `IBExecutor`: bracket orders, position tracking |
| **Protection** | `_protect.py` | ~180 | `protect_position()`: OCA brackets (STP+LMT) |
| **System 2** | `brain/consolidation.py` | 369 | `ConsolidationEngine`: HALIM, thinker, portfolio/safety state |
| **Telemetry** | `telemetry.py` | 410 | `TelemetryAPI`: Flask app on :8080 |
| **Pipeline Monitor** | `monitor/pipeline.py` | ~160 | `PipelineMonitor`: health checks, alerts |
| **Immune System** | `immune.py` | ~80 | Hard limits: MAX_POSITIONS, LOSS_LIMIT, etc. (legacy reference) |
| **Learning Buffer** | `hippocampus.py` | 199 | `Hippocampus`: account PnL feed, open-position mirror (2nd instance here) |
| **Dynamics** | `brain/dynamics.py` | 119 | `Dynamics`: hysteresis, velocity, adaptive threshold |

---

## Data Flow: Main Loop (One Cycle)

```
CYCLE START (every ~1s)
│
├─── _supervise_gateway()
│    └── Check IB connection → reconnect + resubscribe if lost
│
├─── executor.sync_from_ib(streamer)
│    └── read_ib_positions() → reconcile tracked ↔ IB truth
│         └── Adopt orphans (reconciled positions)
│
├─── _sweep_stale_orders()
│    └── Cancel DAY-tif entry parents pending > 60s
│         (GTC protection orders NEVER swept)
│    (daily-loss / position-cap safety now lives in the slow-cortex
│     SafetyProducer — see decide_entry gates below; safety_enabled is
│     OFF by default, activated via the webapp /safety-net command)
│
├─── _sync_subs()
│    ├── Subscribe missing tickers (async reqMktData — no block)
│    ├── _gc_stale_subs() — unsubscribe tickers idle > 60s
│    └── seed_history() ONE ticker per cycle (blocking ~2.4s)
│         └── Open positions seeded first, then scanner candidates
│
├─── _check_manual_flatten() — webapp flatten request
├─── _check_eod_flatten() — force-close intraday positions near close
│
├─── juli.tick(watch, snapshot, streamer, held_positions, closing, pos_info, session)
│    │
│    │   ╔═══════════════════════════════════════════════════╗
│    │   ║              BRAIN EVALUATION PATH               ║
│    │   ╚═══════════════════════════════════════════════════╝
│    │   (NeuromorphicBrain.start() at JuliBrain construction
│    │    spawns the slow-cortex consolidation thread once)
│    │
│    ├─── _data_preamble(streamer, snapshot, held_positions)
│    │    ├─── feed.ensure_refs() — SPY/QQQ/VXX ref prices
│    │    ├─── _sync_and_scan() — sync scanner results
│    │    ├─── _maybe_screen(snapshot) — screen candidates
│    │    ├─── feed.fallback_regime() — local regime when HALIM stale
│    │    └─── _maybe_allocate() — cycle data budget
│    │
│    ├─── EXIT EVALUATION:
│    │    juli._evaluate_exits(positions, snapshot, closing)
│    │    ├─── brain.check_exit(ticker, last_price, direction)
│    │    │    └── exit_checks.py → exit_ladder.py → learned_exit.py
│    │    └─── policy_exits from brain.state["policy_exits"]
│    │         └── slow-cortex staged exits (consolidation publishes)
│    │
│    └─── ENTRY EVALUATION (verdict loop):
│         universe = sorted(watch ∪ held_positions)
│         For each ticker:
│              │
│              ├── juli._snap_for(ticker, snapshot) → window of bars
│              ├── juli._lock_held(ticker, held_positions)
│              ├── juli._eval_window(ticker, snap, held_positions, session)
│              │    │
│              │    ├── governor.begin_cycle() — MAX_ENTRIES_PER_CYCLE budget
│              │    ├── rotating EVAL_WINDOW=4 slice scored via
│              │    │    brain.decide_entry(alpha, bars, price, atr, ...)
│              │    │    │
│              │    │    │  ╔═══════════════════════════════════════════╗
│              │    │    │  ║      NEUROMORPHIC BRAIN (orchestrator)  ║
│              │    │    │  ╚═══════════════════════════════════════════╝
│              │    │    │
│              │    │    ├── Validity gates (data, regime)
│              │    │    ├── Signal: cortex.evaluate → stabilize → dynamics
│              │    │    ├── TradingPolicy verdict (approved/denied/reason)
│              │    │    ├── SafetyProducer (enabled/halted/authorized)
│              │    │    ├── ProbeRecovery (position cap, stress)
│              │    │    ├── Governor (halted? budget? reuse cooldown?)
│              │    │    ├── RiskEngine sizing (Kelly, EV gate, quality)
│              │    │    ├── Portfolio gate (position cap, exposure)
│              │    │    └── Scale (regime multiplier) → Verdict
│              │    │         Verdict(ticker, action, reason, sizing,
│              │    │                 horizon, thought, trace)
│              │    │         └── actions: ENTER / HOLD / VETOED
│              │    │
│              │    ├── Verdict appended to _recent_verdicts (maxlen 200)
│              │    │    └── served at GET /verdicts (telemetry)
│              │    │
│              │    └── governor.note_entry() stamps the reuse cooldown
│              │         AFTER execution confirms an entry
│              │
│              └── returns (exits, verdicts) — decisions only, no orders
│
├─── _finish_cycle(exit_s, verdicts, pnl, meta)
│    │
│    ├─── bars = count pendingTickers → update_bar (aggregate 1-min bars)
│    │
│    ├─── _drain_event_exits()
│    │    └── Drain signal_queue (event-driven exit signals from tick watchers)
│    │         └── If stop breach → close_position()
│    │
│    ├─── Process brain exits:
│    │    For each exit_s:
│    │         executor.close_position(ticker, streamer)
│    │
│    ├─── Journal verdicts (observability):
│    │    For each verdict:
│    │         journal {"event": "verdict", ts, **v.to_dict()}
│    │
│    ├─── Execute ENTER verdicts:
│    │    For each verdict, ONLY IF market_open AND v.action == ENTER:
│    │         _execute_verdict(v)
│    │         │
│    │         ├── Sizing guard: sizing is None or shares ≤ 0 → SKIP
│    │         ├── Live bid/ask (else "VERDICT UNEXECUTABLE")
│    │         ├── Skip if ticker already open
│    │         ├── price = (bid + ask) * 0.5
│    │         ├── executor.place_bracket(ticker, v.thought, price, streamer,
│    │         │                          sizing=v.sizing, horizon=v.horizon)
│    │         │   ├── IB order: LMT parent DAY + STP/LMT GTC
│    │         │   ├── _brackets[ticker] = (stop, target)
│    │         │   └── _protect.protect_position() → OCA group
│    │         ├── juli.last_thoughts[ticker] = v.thought
│    │         ├── juli.brain.register_position(ticker, price, horizon=v.horizon)
│    │         ├── _attach_position_watchers(ticker)
│    │         │    ├── streamer.attach_exit_watcher(ticker, check)
│    │         │    └── streamer.watch_pnl_single(ticker, account)
│    │         └── governor.note_entry(ticker) — reuse cooldown
│    │
│    ├─── _publish_account_feed(pnl)
│    │    └── Set juli._state["account_feed"] = {daily_pnl, ts}
│    │         └── Every RISK_SYNC_SECS: also equity + positions
│    │              └── Slow cortex _update_policy consumes it
│    │
│    ├─── _reflect_closed()
│    │    └── For each newly closed trade:
│    │         ├── juli.brain.on_trade_close(ticker, won, pnl_pct, direction, source)
│    │         │   └── ironclad gate → only ib_fill/real_trade/ib_paper/reconciled_exit pass
│    │         │        ├── episodic.add(alpha, outcome)
│    │         │        ├── nash.record_pattern(...)
│    │         │        ├── realized.record_outcome(...)
│    │         │        └── reflector.adapt(...)
│    │         ├── _closing.discard(ticker)
│    │         ├── _watched.discard(ticker)
│    │         └── streamer.unwatch_pnl_single(ticker)
│    │
│    ├─── monitor.record_cycle(market_open)
│    ├─── time.sleep(max(0.2, poll - elapsed))  ← minimum 0.2s gap
│    └─── _heartbeat() (every 60s)
│
CYCLE END
```

---

## Data Flow: Order Execution (Entry)

```
Decision passes all gates
│
├─── executor.place_bracket(ticker, thought, price, streamer, sizing, horizon)
│    │
│    ├─── Get bid/ask from streamer.get_last_price(ticker)
│    ├─── ATR stop/target from streamer.buffer_atr(ticker)
│    │    └── stop = price - ATR_STOP_MULT * ATR
│    │    └── target = price + ATR_TARGET_MULT * ATR
│    │
│    ├─── Place IB bracket order:
│    │    parent = LimitOrder(direction, shares, price)  [DAY]
│    │    stop_loss = StopOrder(-direction, shares, stop)  [GTC]
│    │    take_profit = LimitOrder(-direction, shares, target)  [GTC]
│    │    ib.placeOrder(parent)
│    │    ib.placeOrder(stop_loss)
│    │    ib.placeOrder(take_profit)
│    │
│    └─── _brackets[ticker] = (stop, target)
│
├─── _protect.protect_position(ticker, streamer)
│    ├─── OCA group: both STP + LMT are OCA-cancel linked
│    ├─── If one fills → other auto-cancels
│    └─── _oca_groups[ticker] = oca_group_id
│
└─── Register for monitoring:
     ├─── executor._open_positions[ticker] = {...}
     ├─── brain.register_position(ticker, price, horizon)
     └─── _attach_position_watchers(ticker)
          ├─── tick.updateEvent += exit_handler (stop breach check)
          └─── reqPnLSingle(account, ticker) (real-time PnL)
```

---

## Data Flow: Order Execution (Exit)

```
Exit signal arrives (one of):
├─── Brain exit: exit_checks.py → exit_ladder.py → learned_exit.py
│    └── should_exit = True, reason = "ladder_breach"
│
├─── Event-driven exit: tick watcher → signal_queue
│    └── reason = "hard_stop_breach"
│
├─── Manual flatten: webapp → _FLATTEN_REQUESTED
│    └── close_all_positions()
│
└─── EOD flatten: _check_eod_flatten()
     └── force-close intraday positions
│
▼
executor.close_position(ticker, streamer)
│
├─── Market order: MarketOrder(-direction, shares)
├─── ib.placeOrder(market_order)
│
├─── Wait for fill (execDetailsEvent callback)
│    └── streamer.record_execution(trade, fill)
│         └── Record: ticker, action, shares, price, timestamp
│
└─── _record_exit()
     ├─── Get realized P&L from fill
     ├─── journal_exit() → append to journal_live.jsonl
     ├─── trade_closed() → Telegram notification
     ├─── executor.get_newly_closed_trades() → queue for _reflect_closed
     └─── _closing.discard(ticker)
```

---

## Data Flow: System 2 (Background Thread)

```
ConsolidationEngine.start()  (every 30s)
│
├─── HALIM Poll:
│    ├─── halim.classify_regime(prices) → regime label + multiplier
│    ├─── halim.get_advisory(recent_trades) → risk modifier
│    └─── halim.get_recommendations() → structured JSON recs
│         └── Apply recs to dynamics.threshold, risk params
│
├─── Thinker:
│    ├─── thinker.analyze(alpha, recent_outcomes) → signal
│    └─── Publish to BrainState (thinker_modifier)
│
├─── News:
│    ├─── news.fetch_headlines() → sentiment
│    └─── Publish to BrainState (news_modifier)
│
├─── Regime Fallback:
│    └─── If HALIM unreachable → local RegimeDetector (numpy)
│
├─── Sleep Replay:
│    └─── replay_simulated_trades() → learn from what-if scenarios
│
├─── HALIM Postmortem (after trade close):
│    └─── halim.analyze_trade(trade_data) → insights, recommendations
│
└─── Persist State:
     ├─── brain_state.to_dict() → runtime/state.json
     └─── memory.save() → persisted weights/params

└─── _update_policy() (slow-cortex ownership of portfolio + safety)
     ├─── Begin safety call: enabled?/daily_pnl/consecutive_losses/position_count
     ├─── SafetyProducer.authorized() → (authorized, pause_reason)
     ├─── Build policy_state:
     │    {enabled, halted, authorized, pause_reason, equity, drawdown,
     │     exposure, stress_mode, risk_scalar, position_count, max_positions,
     │     daily_pnl, consecutive_losses}
     │    └── Published to BrainState → read by telemetry /health, /risk,
     │         /safety-net and by decide_entry gates (halt blocks entries)
     ├─── Stage policy_exits (e.g. stress-de-risking) → merged by juli
     └─── Journal snapshot → safety producer recall
```

---

## Data Flow: IB Gateway ↔ Bot

```
IB Gateway (port 4002 paper / 4001 live)
│
├─── TCP Socket (persistent)
│    │
│    ├─── Incoming to Bot:
│    │    ├─── MarketData → streamer.update_bar()
│    │    │    └── Ticker updates → StreamBuffer (1-min aggregation)
│    │    ├─── execDetailsEvent → streamer.record_execution()
│    │    ├─── commissionReportEvent → streamer.record_commission()
│    │    ├─── pnlUpdateEvent → hippocampus._daily_pnl
│    │    ├─── positionEvent → executor._open_positions sync
│    │    └─── Error events → logged, some suppressed
│    │
│    └─── Outgoing from Bot:
│         ├─── reqMktData(ticker) → subscribe market data
│         ├─── cancelMktData(ticker) → unsubscribe (GC)
│         ├─── reqHistoricalData(ticker) → seed 2D 1min bars
│         ├─── reqPnL(account) → account-level PnL stream
│         ├─── reqPnLSingle(account, ticker) → per-position PnL
│         ├─── reqPositions() → reconcile tracked ↔ IB
│         ├─── reqOpenOrders() → sweep stale entries
│         ├─── placeOrder(bracket) → entry execution
│         ├─── placeOrder(market) → exit execution
│         ├─── cancelOrder(protection) → cleanup on exit
│         └─── reqScannerSubscription() → live scanner
│
└─── Reconnection:
     ├───ponential backoff: 1s → 2s → 4s → 8s → 16s → 30s (cap)
     ├─── On reconnect: _seeded_subs.clear() → re-seed positions first
     └─── All subscriptions re-requested
```

---

## Key Data Structures

```
BrainState (shared_state.py)
├── regime_label, regime_multiplier, regime_risk
├── halim_modifier, episodic_bias, nash_modifier
├── thinker_modifier, news_modifier
├── ref_prices (SPY/QQQ/IWM/VXX)
├── latest_prices, latest_alpha
└── persist → runtime/state.json

SizingResult (brain/risk.py)
├── shares: int
├── stop_price: float
├── target_price: float
├── ev: float, kelly: float
├── risk_pass: bool
├── reason: str
├── ev_scale: float, ev_reason: str
└── quality_penalty: float

Verdict (brain/policy/verdict.py — returned by brain.decide_entry)
├── ticker: str
├── action: str (ENTER / HOLD / VETOED)
├── reason: str (e.g. "sized_to_zero", "security_halted", "admitted")
├── sizing: SizingResult | None
├── horizon: str (scalp/multihour/swing)
├── thought: SimpleNamespace(direction, score, confidence, ...)
└── trace: {base, neuro, nash, halim} (observer-only)
```

---

## Ironclad Gate (Learning Protection)

```
on_trade_close() source check:
│
├─── ALLOWED (pass through to learning):
│    ├── ib_fill          (bot placed order, IB filled)
│    ├── real_trade       (real market execution)
│    ├── ib_paper         (paper account fill)
│    └── reconciled_exit  (adopted position closed)
│
└─── BLOCKED (skip all learning):
     ├── synthetic        (reconciled entry, not real fill)
     ├── paper_replay     (sleep replay, what-if)
     ├── backtest         (historical simulation)
     └── any other source
```

---

## Module Dependency Graph (Key Paths)

```
cli.py
  └── ib_adapter.py (IBStreamingBot)
        ├── ib_cycle.py (BotCycleMixin) ←── MAIN LOOP
        │     ├── ib_executor.py (IBExecutor)
        │     │     ├── _protect.py (OCA brackets)
        │     │     ├── ib_bracket.py (order construction)
        │     │     └── _ib_sync.py (read_ib_positions, journal)
        │     ├── ib_streamer.py (IBStreamer)
        │     │     ├── ib_compat.py (ib_insync compat)
        │     │     └── eyes.py (rolling_atr, indicators)
        │     ├── juli.py (JuliBrain) ←── BRAIN INTERFACE
        │     │     ├── brain/orchestrator.py (NeuromorphicBrain)
        │     │     │     ├── cortex.py (Cortex)
        │     │     │     ├── brain/dynamics.py (Dynamics)
        │     │     │     ├── brain/risk.py (RiskEngine)
        │     │     │     ├── brain/episodic.py (EpisodicMemory)
        │     │     │     ├── brain/regime.py (RegimeDetector)
        │     │     │     ├── brain/cross_asset.py (CrossAsset)
        │     │     │     ├── brain/gate_advisor.py (GateAdvisor)
        │     │     │     ├── brain/neurons/ (NeuromorphicNetwork)
        │     │     │     └── brain/policy/  ←── DECISION GATES
        │     │     │          ├── trading_policy.py (TradingPolicy)
        │     │     │          ├── safety.py (SafetyProducer)
        │     │     │          ├── governor.py (Governor)
        │     │     │          └── probe_recovery.py (ProbeRecovery)
        │     │     ├── juli_feed.py (tick → snapshot conversion)
        │     │     └── brain/exit_checks.py → exit_ladder.py
        │     └── monitor/pipeline.py (PipelineMonitor)
        │
        ├── brain/consolidation.py (System 2 — slow cortex, every ~30s)
        │     ├── SafetyProducer (portfolio/safety policy_state owner)
        │     ├── brain/halim_adapter.py (HALIM)
        │     ├── brain/thinker.py (Thinker)
        │     ├── brain/news_sources.py (NewsFeed)
        │     ├── reflection/buffer.py (TradeBuffer)
        │     └── reflection/supervisor.py (LearningSupervisor)
        │
        ├── hippocampus.py (Hippocampus — account PnL feed + position mirror)
        ├── telemetry.py (TelemetryAPI — Flask :8080, observer of policy_state
        │                + /verdicts from juli._recent_verdicts)
        └── (no top-level decision organs: brain-first, all gates inside brain/)
```
