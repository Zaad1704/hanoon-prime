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
| **Bot Core** | `ib_adapter.py` | 170 | `IBStreamingBot` class, owns all components |
| **Main Loop** | `ib_cycle.py` | 623 | `BotCycleMixin`: cycle, sync, execution, safety |
| **Data Layer** | `ib_streamer.py` | 334 | `IBStreamer`: market data, DOM, history seeding |
| **Brain Interface** | `juli.py` | 179 | `JuliBrain`: evaluates tickers, routes to orchestrator |
| **Brain Core** | `brain/orchestrator.py` | 806 | `NeuromorphicBrain`: scoring, sizing, verdicts |
| **Cortex** | `cortex.py` | ~200 | `Cortex.evaluate()`: alpha → score + direction |
| **Risk Engine** | `brain/risk.py` | ~320 | `RiskEngine.evaluate()`: sizing + risk gate |
| **Execution** | `ib_executor.py` | 377 | `IBExecutor`: bracket orders, position tracking |
| **Protection** | `_protect.py` | ~180 | `protect_position()`: OCA brackets (STP+LMT) |
| **System 2** | `brain/consolidation.py` | 304 | `ConsolidationEngine`: HALIM, thinker, persistence |
| **Telemetry** | `telemetry.py` | ~200 | `TelemetryAPI`: Flask app on :8080 |
| **Pipeline Monitor** | `monitor/pipeline.py` | ~160 | `PipelineMonitor`: health checks, alerts |
| **Portfolio Risk** | `monitor/portfolio_risk.py` | ~120 | `PortfolioRiskManager`: drawdown → risk scalar |
| **Immune System** | `immune.py` | ~80 | Hard limits: MAX_POSITIONS, LOSS_LIMIT, etc. |
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
│
├─── _check_safety(pnl)
│    └── Check daily loss, consecutive losses → _halt() if breached
│         (safety_enabled=False by default; user activates via webapp)
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
├─── juli.tick(positions, snapshot, streamer, closing)
│    │
│    │   ╔═══════════════════════════════════════════════════╗
│    │   ║              BRAIN EVALUATION PATH               ║
│    │   ╚═══════════════════════════════════════════════════╝
│    │
│    ├─── juli._sync_and_scan() — sync scanner results
│    ├─── juli._maybe_screen(snapshot) — screen candidates
│    ├─── juli._maybe_allocate(positions) — data budget allocation
│    │
│    ├─── EXIT EVALUATION:
│    │    juli._evaluate_exits(positions, snapshot, closing)
│    │    └── For each open position:
│    │         brain.check_exit(ticker, last_price, direction)
│    │         └── exit_checks.py → exit_ladder.py → learned_exit.py
│    │
│    └─── ENTRY EVALUATION:
│         juli._evaluate_entries(positions, snapshot)
│         └── For each tracked ticker:
│              │
│              ├── juli._eval_one(ticker, snap, open_count)
│              │    │
│              │    ├── len(prices) < 20 → SKIP (no data)
│              │    │
│              │    ├── brain.tick(alpha, ticker, price, atr, open_positions, bars)
│              │    │    │
│              │    │    │  ╔═══════════════════════════════════════════╗
│              │    │    │  ║      NEUROMORPHIC BRAIN (orchestrator)  ║
│              │    │    │  ╚═══════════════════════════════════════════╝
│              │    │    │
│              │    │    ├── Refractory check → skip if cooling down
│              │    │    │
│              │    │    └── _evaluate_fast():
│              │    │         │
│              │    │         ├── Get regime data from BrainState
│              │    │         │   (HALIM modifier, episodic bias, nash mod)
│              │    │         │
│              │    │         ├── _local_regime_fallback()
│              │    │         │   └── If HALIM label unknown → numpy detector
│              │    │         │
│              │    │         ├── _cross_asset.update(ticker, price, ref_prices)
│              │    │         │   └── SPY/QQQ/IWM/VXX lead-lag modifier
│              │    │         │
│              │    │         ├── _classify_horizon(bars)
│              │    │         │   └── scalp / multihour / swing
│              │    │         │
│              │    │         └── _score_pipeline(ticker, alpha, regime_mul, halim, episodic, cross)
│              │    │              │
│              │    │              ├── base = cortex.evaluate(alpha)
│              │    │              │   └── ScoreResult(score, direction, confidence, verdict)
│              │    │              │
│              │    │              ├── nash_pred = nash.predict(alpha, base.score, base.direction)
│              │    │              ├── nash_op = _compute_nash_mod(nash_pred)
│              │    │              ├── neuro_score = _compute_neuro_score(alpha, ticker)
│              │    │              ├── cal_adj = _calibration_nudge(base.score)
│              │    │              │
│              │    │              ├── blended = (1-NEURO_BLEND)*(base.score+cal_adj) + NEURO_BLEND*neuro_score
│              │    │              ├── advisor_delta = _advisor.threshold_delta()
│              │    │              ├── news_bias = _news_bias(ticker)
│              │    │              │
│              │    │              ├── raw = blended * regime_mul + halim + episodic + nash_op + news_bias + cross - advisor_delta
│              │    │              │
│              │    │              └── _stabilize(raw, nash_pred)
│              │    │                   ├── direction = sign(raw)
│              │    │                   ├── score = _apply_nash_gate(raw, direction, nash_pred)
│              │    │                   ├── score = 0.0 if EOD penalty
│              │    │                   ├── stabilized = dynamics.process(score, direction)
│              │    │                   │   └── hysteresis + velocity + refractory + threshold adaptation
│              │    │                   └── final_dir = sign(stabilized)
│              │    │
│              │    │
│              │    │    └── _maybe_size(ctx, entry_price, atr, open_positions)
│              │    │         ├── abs(score) <= dynamics.threshold * patience → SizingResult() (empty)
│              │    │         └── risk.evaluate(score, confidence, price, atr, open_positions)
│              │    │              ├── NaN guard → reject
│              │    │              ├── Position cap → reject
│              │    │              ├── Win probability from edge.py
│              │    │              ├── Kelly sizing
│              │    │              ├── EV gate (minimum EV threshold)
│              │    │              ├── Quality penalty (spread, volume)
│              │    │              └── Returns SizingResult(shares, stop, target, risk_pass, ...)
│              │    │
│              │    │
│              │    └── Returns decision dict:
│              │         {ticker, direction, score, sizing, verdict, horizon, trace}
│              │
│              └── juli._build_decision(ticker, direction, result)
│                   └── decision dict with thought=SimpleNamespace(...)
│
├─── _finish_cycle(exit_s, decisions, pnl, meta)
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
│    ├─── Process entry decisions:
│    │    For each decision:
│    │         if market_open AND _can_trade(dec):
│    │              _exec_decision(dec)
│    │              │
│    │              ├── _halted check → skip
│    │              ├── _can_trade(): direction allowed, session active
│    │              ├── _portfolio_gate_and_size(): risk gate + size adjustment
│    │              ├── _check_safety(): position cap, loss limits
│    │              ├── executor.place_bracket(ticker, thought, price, streamer, sizing, horizon)
│    │              │   ├── IB order: MOC/LMT parent + STP + LMT GTC
│    │              │   ├── _brackets[ticker] = (stop, target)
│    │              │   └── _protect.protect_position() → OCA group
│    │              ├── juli.brain.register_position(ticker, price, horizon)
│    │              └── _attach_position_watchers(ticker)
│    │                   ├── streamer.attach_exit_watcher(ticker, check)
│    │                   │   └── tick.updateEvent → enqueue exit signal if stop breached
│    │                   └── streamer.watch_pnl_single(ticker, account)
│    │
│    ├─── _reflect_closed()
│    │    └── For each newly closed trade:
│    │         ├── juli.brain.on_trade_close(ticker, won, pnl_pct, direction, source)
│    │         │   └── ironclade gate → only ib_fill/real_trade/ib_paper/reconciled_exit pass
│    │         │        ├── episodic.add(alpha, outcome)
│    │         │        ├── nash.record_pattern(...)
│    │         │        ├── realized.record_outcome(...)
│    │         │        └── reflector.adapt(...)
│    │         ├── _closing.discard(ticker)
│    │         ├── _watched.discard(ticker)
│    │         └── streamer.unwatch_pnl_single(ticker)
│    │
│    ├─── monitor.record_cycle(market_open)
│    ├─── _sync_portfolio_risk() (throttled every 30s)
│    │    └── Read NetLiquidation → drawdown → risk_scalar → adjust entry size
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

Decision Dict (returned by brain.tick)
├── ticker: str
├── direction: int (+1/-1/0)
├── score: float (stabilized)
├── sizing: SizingResult
├── verdict: str (BUY/SELL/REFRACTORY)
├── horizon: str (scalp/multihour/swing)
├── confidence: float
├── trace: {base, neuro, nash, halim}
└── thought: SimpleNamespace(direction, score, verdict, confidence)
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
        │     │     │     └── brain/neurons/ (NeuromorphicNetwork)
        │     │     ├── juli_feed.py (tick → snapshot conversion)
        │     │     └── brain/exit_checks.py → exit_ladder.py
        │     └── monitor/pipeline.py (PipelineMonitor)
        │
        ├── brain/consolidation.py (System 2)
        │     ├── brain/halim_adapter.py (HALIM)
        │     ├── brain/thinker.py (Thinker)
        │     ├── brain/news_sources.py (NewsFeed)
        │     ├── reflection/buffer.py (TradeBuffer)
        │     └── reflection/supervisor.py (LearningSupervisor)
        │
        ├── hippocampus.py (Hippocampus — safety state)
        ├── telemetry.py (TelemetryAPI — Flask :8080)
        └── monitor/portfolio_risk.py (PortfolioRiskManager)
```
