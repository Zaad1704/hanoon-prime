# HANOON PRIME — Complete Runtime Map

> Every step below is traced from actual source code. Not a document.
> Not a guess. This IS the code's behavior.

---

## STEP 0: System Boot

```
$ python3 -m hanoon_prime.cli
│
├── cli.py:main()
│   ├── _setup_logging()
│   │   └── CleanFormatter strips "hanoon_prime." prefix from logger names
│   │   └── Suppresses ib_insync.wrapper, ib_insync.ib (CRITICAL)
│   │   └── Suppresses ib_insync.client (WARNING)
│   │
│   ├── tickers = sys.argv[1:] or LIQUID_US_SEED[:5]
│   │   └── Default: ['SPY','QQQ','NVDA','AAPL','MSFT'] (immune.py)
│   │
│   ├── repo_root = Path(__file__).resolve().parents[2]
│   ├── journal_path = repo_root / "runtime" / "journal_live.jsonl"
│   ├── journal_path.parent.mkdir(parents=True, exist_ok=True)
│   │
│   ├── bot = IBStreamingBot(account="PAPER")
│   │   │
│   │   ├── ib_adapter.py:IBStreamingBot.__init__()
│   │   │   ├── self.ib = ib.IB()                    # ib_insync client
│   │   │   ├── self.account = "PAPER"
│   │   │   ├── self.hippocampus = Hippocampus(safety_enabled=False)
│   │   │   │   └── safety_enabled=False → halts OFF by default
│   │   │   ├── self._halted = False
│   │   │   ├── self.brain_state = BrainState()
│   │   │   │   └── Shared dict: regime, modifiers, prices, alpha
│   │   │   ├── self.juli = JuliBrain(self.ib)
│   │   │   │   ├── JuliBrain.__init__()
│   │   │   │   │   ├── self.brain = NeuromorphicBrain()
│   │   │   │   │   │   ├── self.cortex = Cortex()       # alpha → score
│   │   │   │   │   │   ├── self.nash = NashBrain()       # pattern memory
│   │   │   │   │   │   ├── self.dynamics = Dynamics()    # hysteresis + threshold
│   │   │   │   │   │   │   └── _threshold = 0.58 (adaptive p90-based)
│   │   │   │   │   │   ├── self.risk = RiskEngine()     # sizing + risk gate
│   │   │   │   │   │   ├── self.episodic = EpisodicMemory()  # k-NN memory
│   │   │   │   │   │   ├── self._regime_detector = RegimeDetector()
│   │   │   │   │   │   ├── self._regime_weights = RegimeWeights()
│   │   │   │   │   │   ├── self._cross_asset = CrossAsset()
│   │   │   │   │   │   ├── self._advisor = GateAdvisor()
│   │   │   │   │   │   └── self._neuromorphic = NeuromorphicNetwork()
│   │   │   │   │   ├── self.feed = JuliFeed()
│   │   │   │   │   ├── self.scanner = Scanner()
│   │   │   │   │   └── self.budget = DataBudget()
│   │   │   │   └── self._state = BrainState()
│   │   │   │
│   │   │   ├── self.journal = Journal(runtime/journal_live.jsonl)
│   │   │   ├── self.streamer = IBStreamer(self.ib)
│   │   │   │   ├── self.buffers = {}           # StreamBuffer per ticker
│   │   │   │   ├── self.contracts = {}          # IB Contract per ticker
│   │   │   │   ├── self.ticker_subs = {}        # Ticker objects from reqMktData
│   │   │   │   ├── self.depth_subs = {}         # Level 2 subs (disabled)
│   │   │   │   ├── self.signal_queue = Queue()  # event-driven exits
│   │   │   │   └── self.last_seen = {}          # GC timestamps
│   │   │   │
│   │   │   ├── self.executor = IBExecutor(self.ib, self.hippocampus, self.journal)
│   │   │   │   ├── self._brackets = {}          # ticker → (stop, target)
│   │   │   │   ├── self._pending_parent = set() # awaiting fill
│   │   │   │   ├── self._synthetic = set()      # reconciled positions
│   │   │   │   ├── self._horizons = {}          # ticker → horizon
│   │   │   │   └── self._open_positions = {}    # tracked by brain
│   │   │   │
│   │   │   ├── self._chat = TelegramChat()
│   │   │   ├── self.monitor = PipelineMonitor(self, self.journal)
│   │   │   │   └── Thread: monitors pipeline health, sends alerts
│   │   │   ├── self._closing = set()
│   │   │   ├── self._exit_reasons = {}
│   │   │   ├── self._watched = set()            # tick watchers attached
│   │   │   └── self._seeded_subs = set()        # history seeded tickers
│   │   │
│   │   └── _setup_signals() → SIGINT/SIGTERM → self._running = False
│   │
│   ├── telemetry = TelemetryAPI(bot, journal_path)
│   │   └── TelemetryAPI.__init__()
│   │       └── Routes: /health, /positions, /brain, /safety-net, /journal, etc.
│   │
│   ├── telemetry.start()
│   │   └── threading.Thread(target=http_server) → port 8080
│   │
│   └── bot.run_paper(tickers)
│       │
│       ├── bot.connect(port=IB_PAPER_PORT)
│       │   └── try_connect(ib, host, port, clientId)
│       │       └── ib.connect(host, port, clientId=cid)
│       │       └── ib.execDetailsEvent += streamer.record_execution
│       │       └── ib.commissionReportEvent += streamer.record_commission
│       │
│       └── bot.run(tickers, poll=1.0)
│           │
│           ├── self._running = True
│           ├── startup(seed) → Telegram: bot started
│           ├── executor.tracked_tickers = set(seed)
│           │
│           ├── FOR EACH seed ticker:
│           │   ├── streamer.subscribe(t)
│           │   │   ├── ib.Stock(t, "SMART", "USD")
│           │   │   ├── ib.qualifyContracts(contract)
│           │   │   ├── self.contracts[t] = contract
│           │   │   ├── self.buffers[t] = StreamBuffer(t)
│           │   │   ├── self.ticker_subs[t] = ib.reqMktData(contract, "", False, False)
│           │   │   └── self.depth_subs[t] = None
│           │   │
│           │   └── streamer.seed_history(t)
│           │       ├── ib.reqHistoricalData(contract, "", "2 D", "1 min", "TRADES", True, 1)
│           │       ├── ib.sleep(1)
│           │       └── FOR EACH bar (reversed):
│           │           └── buffer.append(BarSeries(close, high, low, vol, buy_vol, bid, ask))
│           │
│           ├── pnl = self._start_pnl()
│           │   └── ib.reqPnL(account, "") → real-time account P&L
│           │
│           ├── self._chat.start() → TelegramChat thread
│           ├── self.monitor.start() → PipelineMonitor thread
│           │
│           ├── log.info("All streams active. Entering event loop...")
│           │
│           └── WHILE self._running:
│               └── self._cycle(poll, pnl)  ←── STEP 1
```

---

## STEP 1: Main Loop — One Cycle

```
_cycle(poll=1.0, pnl)
│
│ started = time.monotonic()
│
├── _supervise_gateway()
│   ├── ib.isConnected() → False?
│   │   ├── delay = min(30.0, 1.0 * 2^attempts)
│   │   ├── time.sleep(delay)
│   │   ├── self._reconnect()
│   │   │   └── self.connect(host, port, clientId)
│   │   │       └── ib.connect(host, port, clientId=cid)
│   │   │
│   │   └── self._resubscribe_all()
│   │       └── FOR EACH tracked ticker:
│   │           └── streamer.subscribe(t)
│   │               └── ib.reqMktData(contract, "", False, False)
│   │
│   └── ib.isConnected() → True? → continue
│
├── executor.sync_from_ib(streamer)
│   │
│   ├── ib.isConnected() → False? → return
│   │
│   ├── sweep_zombies(ib) → cancel zombie orders
│   │
│   ├── _adopt_orphan_positions(streamer)
│   │   └── FOR EACH IB position (qty != 0):
│   │       ├── sym not in last_thoughts?
│   │       │   ├── tracked_tickers.add(sym)
│   │       │   ├── streamer.subscribe(sym) → ib.reqMktData
│   │       │   ├── streamer.seed_history(sym) → ib.reqHistoricalData
│   │       │   └── log("RECONCILE: adopted %s")
│   │       │
│   │       └── last_thoughts[sym] = {direction, price, shares, synthetic=True}
│   │
│   ├── protect_position(ib, tracked, brackets, pending, streamer)
│   │   └── FOR EACH open order:
│   │       └── If parent filled but no protection → place OCA bracket
│   │
│   ├── _brackets_from_trades(ib, tracked, brackets)
│   │   └── Sync brackets from IB order state
│   │
│   ├── ib_positions = read_ib_positions(ib, tracked, brackets)
│   │
│   ├── GUARD: ib_positions empty AND brain has positions?
│   │   └── log.debug("IB returned 0 positions — skipping exit scan")
│   │   └── return (DON'T fire false exits)
│   │
│   ├── FOR EACH brain._open_positions NOT in ib_positions:
│   │   └── _record_exit(t, streamer)
│   │       ├── _brackets.pop(t)
│   │       ├── _horizons.pop(t)
│   │       ├── pos = brain._open_positions.pop(t)
│   │       ├── pnl = get_ib_pnl(ib, t, pos)
│   │       ├── is_synthetic = t in _synthetic
│   │       ├── IF NOT synthetic:
│   │       │   ├── trade_closed(t, "LONG"/"SHORT", pnl) → Telegram
│   │       │   ├── brain.record_trade(t, won, pnl, direction)
│   │       │   └── journal_exit(journal, t, pnl, pos)
│   │       │
│   │       └── _closed_trades.append({ticker, pnl, source, ...})
│   │
│   └── brain._open_positions = ib_positions
│
├── _sweep_stale_orders()
│   │
│   ├── FOR EACH ib.openTrades():
│   │   ├── order.parentId exists? → skip (protection leg)
│   │   ├── order.tif != "DAY"? → skip (GTC protection)
│   │   ├── order.status in (PendingSubmit, PreSubmitted)?
│   │   │   └── time - placed > 60s?
│   │   │       └── ib.cancelOrder(order) → log("SWEEP cancel %s")
│   │   │
│   │   └── ORDER is day-tif entry parent, pending > 60s → CANCEL
│   │       (GTC protection orders NEVER touched)
│   │
│   └── PROTECTION ORDERS (GTC) SURVIVE ALL SWEEPS
│
├── _check_safety(pnl)
│   │
│   ├── hippocampus.safety_enabled → False? → return
│   │   └── SAFETY OFF BY DEFAULT
│   │
│   ├── IF safety_enabled == True:
│   │   ├── n = count_open_positions(ib, tracked)
│   │   ├── n > MAX_CONCURRENT_POSITIONS (30)?
│   │   │   └── log.critical("SAFETY: %d > %d") → _halt("too_many_positions")
│   │   │
│   │   ├── pnl.dailyPnL < -DAILY_LOSS_LIMIT (-$500)?
│   │   │   └── log.critical("SAFETY: P&L $%.2f") → _halt("daily_loss_limit")
│   │   │
│   │   └── consecutive_losses > CONSECUTIVE_LOSSES_PAUSE (5)?
│   │       └── _halt("consecutive_losses")
│   │
│   └── _halt(reason):
│       ├── log.warning("HALT: %s (entries blocked)")
│       ├── self._halted = True
│       ├── telegram.hafety_halt(reason)
│       └── BOT CONTINUES RUNNING (never sets _running=False)
│
├── _sync_subs()
│   │
│   ├── tracked = juli.budget.get_all_tracked()
│   ├── scanner = {c.symbol for c in juli._candidates[:20]}
│   ├── needed = tracked | scanner | set(hippocampus._open_positions)
│   │
│   ├── executor.tracked_tickers = tracked
│   │
│   ├── streamer.touch(needed) → mark all as fresh
│   │   └── FOR EACH t: last_seen[t] = now.time()
│   │
│   ├── missing = [s for s in sorted(needed) if s not in ticker_subs]
│   │
│   ├── FOR EACH missing ticker (FAST — async reqMktData):
│   │   └── streamer.subscribe(s) → ib.reqMktData(contract, "", False, False)
│   │       └── IB sends back market data on TCP socket
│   │
│   ├── _gc_stale_subs()
│   │   └── FOR EACH ticker in last_seen:
│   │       ├── now - last_seen[t] > 60s?
│   │       ├── t in open_positions? → touch(t) → KEEP
│   │       └── streamer.unsubscribe(t)
│   │           ├── ib.cancelMktData(sub)
│   │           ├── Remove from ticker_subs, buffers, contracts, etc.
│   │           └── log("Unsubscribed %s (GC)")
│   │
│   └── SEED HISTORY (SLOW — blocking, ONE per cycle):
│       └── pending = [open positions first, then scanner candidates]
│           └── s = pending[0]  (positions prioritized)
│               └── streamer.seed_history(s)
│                   ├── ib.reqHistoricalData(contract, "", "2 D", "1 min", ...)
│                   ├── ib.sleep(1)
│                   └── FOR EACH bar → buffer.append(BarSeries(...))
│
├── IF monitor.pop_heal():  → force re-subscribe
│   └── _sync_subs() again
│
├── IF _check_manual_flatten():
│   └── _FLATTEN_REQUESTED.pop() → webapp requested flatten
│       ├── ib.positions() → count open
│       ├── executor.close_all_positions(streamer)
│       │   └── FOR EACH position:
│       │       └── MarketOrder(-direction, shares) → ib.placeOrder
│       └── _finish_cycle([], [], pnl, market_open=False) → return
│
├── IF _check_eod_flatten():
│   └── minutes_to_close() <= 5?
│       ├── executor.close_all_positions(streamer, only=intraday_horizons)
│       └── _finish_cycle([], [], pnl, market_open=False) → return
│
├── positions = set(hippocampus._open_positions.keys())
│
├── market_open = _SLEEP_MGR.get_state().active
│   └── SleepManager.get_state()
│       ├── _force_active? → SleepState(active=True, session="forced")
│       ├── weekday >= 5? → active=False (weekend)
│       ├── RTH: 9:30-16:00 ET → active=True, session="RTH"
│       ├── 8:00 ET → active=False, session="pre_post"
│       └── Otherwise → active=False, session="overnight"
│
├── exit_s, decisions = juli.tick(positions, snapshot, streamer, closing)
│   │
│   │   ╔═══════════════════════════════════════════════════════╗
│   │   ║              STEP 2: BRAIN EVALUATION                ║
│   │   ╚═══════════════════════════════════════════════════════╝
│   │
│   └── (see STEP 2 below)
│
└── _finish_cycle(exit_s, decisions, pnl, meta)
    │
    │   ╔═══════════════════════════════════════════════════════╗
    │   ║              STEP 3: ORDER DISPATCH                  ║
    │   ╚═══════════════════════════════════════════════════════╝
    │
    └── (see STEP 3 below)
```

---

## STEP 2: Brain Evaluation (juli.tick)

```
juli.tick(positions, get_snapshot, streamer, closing)
│
├── self.feed.ensure_refs(streamer)
│   └── Subscribe to SPY/QQQ/IWM/VXX if not yet subscribed
│
├── _sync_and_scan()
│   ├── scanner.should_scan()?
│   │   └── scanner.scan("most_active") → ib.reqScannerSubscription
│   │       └── IB returns top gainers/losers/most-active
│   │
│   └── results := scanner.collect()
│       └── self._candidates = results
│
├── _maybe_screen(get_snapshot)
│   ├── IF candidates exist:
│   │   ├── FOR EACH candidate[:MAX_CANDIDATES]:
│   │   │   └── snap = get_snapshot(c.symbol) → streamer.get_snapshot(ticker)
│   │   │
│   │   ├── n = count snaps where last > 0
│   │   ├── log("SCREEN: %d/%d passed", n, len(candidates))
│   │   │
│   │   └── feed.publish_ref_prices(get_snapshot)
│   │       └── Publish SPY/QQQ/IWM/VXX to BrainState.ref_prices
│   │
│   └── IF no candidates → skip
│
├── feed.fallback_regime()
│   └── If BrainState.regime_label == "unknown" → local detector
│
├── _maybe_allocate(positions)
│   └── IF 5s since last allocation:
│       └── budget.allocate(positions, candidate_symbols)
│           └── Rank tickers by: open position > scanner > tracked
│               └── Set data budget priorities
│
├── EXIT EVALUATION:
│   _evaluate_exits(positions, get_snapshot, closing)
│   │
│   └── FOR EACH open position t:
│       ├── t in closing? → skip (already being closed)
│       ├── snap = get_snapshot(t) → None? → skip
│       ├── snap["last"] <= 0? → skip
│       │
│       └── sig = brain.check_exit(t, snap["last"], direction=1)
│           │
│           ├── brain/exit_checks.py → exit_ladder.py
│           │   ├── Check time-based exits (holding too long)
│           │   ├── Check profit-taking (target reached)
│           │   ├── Check stop-loss (stop breached)
│           │   └── Check learned exits (pattern-based)
│           │
│           └── IF sig.should_exit:
│               ├── exits.append({ticker, reason, type})
│               └── log("EXIT SIGNAL %s: %s", t, sig.reason)
│
├── ENTRY EVALUATION:
│   _evaluate_entries(positions, get_snapshot)
│   │
│   └── FOR EACH ticker in budget.get_all_tracked() | positions:
│       │
│       ├── snap = get_snapshot(ticker) → None? → skip
│       │
│       ├── _eval_one(ticker, snap, open_count)
│       │   │
│       │   ├── prices = snap.get("prices") or []
│       │   ├── len(prices) < 20? → return None (insufficient data)
│       │   │
│       │   ├── state.set_latest_prices(prices)
│       │   │
│       │   ├── t0 = time.perf_counter_ns()
│       │   │
│       │   ├── result = brain.tick(alpha, ticker, price, atr, open_positions, bars)
│       │   │   │
│       │   │   │  ╔═══════════════════════════════════════════╗
│       │   │   │  ║   NEUROMORPHIC BRAIN CORE (orchestrator) ║
│       │   │   │  ╚═══════════════════════════════════════════╝
│       │   │   │
│       │   │   ├── state.is_refractory()?
│       │   │   │   └── True → return {direction=0, verdict="REFRACTORY"}
│       │   │   │
│       │   │   └── _evaluate_fast(ticker, alpha, price, atr, open_positions, bars)
│       │   │       │
│       │   │       ├── _r, rl, rr, hm, eb, _ = _get_regime_data()
│       │   │       │   └── Read from BrainState:
│       │   │       │       regime_multiplier, regime_label, regime_risk
│       │   │       │       halim_modifier, episodic_bias, nash_modifier
│       │   │       │
│       │   │       ├── _r, rl = _local_regime_fallback(rl)
│       │   │       │   └── If HALIM label unknown → numpy RegimeDetector
│       │   │       │
│       │   │       ├── canon = _canonical_regime(rl)
│       │   │       │   └── "trending_bull"→"trend_up", "ranging"→"range", etc.
│       │   │       │
│       │   │       ├── cross = _cross_asset.update(ticker, price, ref_prices)
│       │   │       │   └── SPY/QQQ/IWM/VXX lead-lag modifier
│       │   │       │
│       │   │       ├── horizon = _classify_horizon(bars)
│       │   │       │   └── scalp / multihour / swing
│       │   │       │
│       │   │       ├── horizon, hz_reason = _bandit.select(canon, horizon)
│       │   │       │   └── Horizon bandit may override (bounded)
│       │   │       │
│       │   │       ├── _apply_regime_weights(canon)
│       │   │       │   └── Adjust weights by regime type
│       │   │       │
│       │   │       └── ctx = _score_pipeline(ticker, alpha, regime_mul, halim, episodic, cross)
│       │   │           │
│       │   │           ├── base = cortex.evaluate(alpha, prior_top=realized.dynamic_prior_top())
│       │   │           │   └── cortex.py:Evaluate()
│       │   │           │       ├── Convert alpha dict → indicator values
│       │   │           │       ├── Score each indicator
│       │   │           │       ├── Weighted average → raw score
│       │   │           │       ├── direction = sign(score)
│       │   │           │       ├── confidence = abs(score)
│       │   │           │       └── Return ScoreResult(score, direction, confidence, verdict)
│       │   │           │
│       │   │           ├── nash_pred = nash.predict(alpha, base.score, base.direction)
│       │   │           │   └── Pattern memory: has this alpha pattern been seen?
│       │   │           │       └── Returns NashPrediction(win_prob, confidence, gate_authority)
│       │   │           │
│       │   │           ├── nash_op = _compute_nash_mod(nash_pred)
│       │   │           │   └── Bounded modifier from win_prob vs 0.5
│       │   │           │
│       │   │           ├── neuro_score = _compute_neuro_score(alpha, ticker)
│       │   │           │   └── NeuromorphicNetwork.process_alpha(alpha, ticker)
│       │   │           │       └── Spike-based neural scoring
│       │   │           │
│       │   │           ├── cal_adj = _calibration_nudge(base.score)
│       │   │           │   └── Realized WR - predicted win prob
│       │   │           │
│       │   │           ├── blended = (1-NEURO_BLEND)*(base.score+cal_adj) + NEURO_BLEND*neuro_score
│       │   │           │   └── NEURO_BLEND = 0.3
│       │   │           │
│       │   │           ├── advisor_delta = _advisor.threshold_delta()
│       │   │           │   └── Learned threshold adjustment from realized WR
│       │   │           │
│       │   │           ├── news_bias = _news_bias(ticker)
│       │   │           │   └── Bounded ±0.03 from news sentiment
│       │   │           │
│       │   │           ├── raw = blended * regime_mul + halim + episodic + nash_op + news_bias + cross - advisor_delta
│       │   │           │
│       │   │           └── _stabilize(raw, nash_pred)
│       │   │               ├── direction = sign(raw)
│       │   │               ├── score = _apply_nash_gate(raw, direction, nash_pred)
│       │   │               │   └── Bounded penalty from pattern memory
│       │   │               │       (never zeros — just leans)
│       │   │               │
│       │   │               ├── IF EOD penalty: score = 0.0
│       │   │               │
│       │   │               ├── stabilized, dyn_reason = dynamics.process(score, direction)
│       │   │               │   └── dynamics.py:Dynamics.process()
│       │   │               │       ├── _score_history.append(raw_score)
│       │   │               │       ├── _quintile_history.append(raw_score)
│       │   │               │       ├── _update_quintile_threshold()
│       │   │               │       │   └── p90 of recent |scores| → adaptive threshold
│       │   │               │       │
│       │   │               │       ├── velocity = _compute_velocity()
│       │   │               │       │   └── Score momentum over window
│       │   │               │       │
│       │   │               │       ├── hysteresis_adj = _apply_hysteresis(score, direction)
│       │   │               │       │   └── Resist flipping BUY↔SELL
│       │   │               │       │
│       │   │               │       ├── refractory_adj = _apply_refractory()
│       │   │               │       │   └── Suppress after trade events
│       │   │               │       │
│       │   │               │       └── stabilized = score + hysteresis + velocity*0.1 + refractory
│       │   │               │           └── clamp(stabilized, -1.0, 1.0)
│       │   │               │
│       │   │               └── final_dir = sign(stabilized)
│       │   │
│       │   │
│       │   │   └── _maybe_size(ctx, entry_price, atr, open_positions)
│       │   │       │
│       │   │       ├── abs(score) <= dynamics.threshold * patience?
│       │   │       │   └── True → return SizingResult() (empty, risk_pass=False)
│       │   │       │       └── SCORE TOO LOW → NO ENTRY
│       │   │       │
│       │   │       └── risk.evaluate(score, confidence, entry_price, atr, open_positions)
│       │   │           │
│       │   │           ├── NaN check → reject
│       │   │           ├── Position cap (MAX_CONCURRENT_POSITIONS=30) → reject
│       │   │           ├── Win probability from edge.py:score_to_win_prob()
│       │   │           │   └── tanh(score) → win_prob in [0.4, 0.6]
│       │   │           ├── Kelly sizing: f* = (p*b - q) / b
│       │   │           ├── EV gate: if EV < min → reject
│       │   │           ├── Quality penalty: spread, volume adjustments
│       │   │           ├── Risk scalar from PortfolioRiskManager
│       │   │           │   └── Drawdown → reduced sizing
│       │   │           │
│       │   │           └── Return SizingResult(
│       │   │               shares, stop_price, target_price,
│       │   │               ev, kelly, risk_pass=True/False,
│       │   │               reason, ev_scale, ev_reason, quality_penalty
│       │   │           )
│       │   │
│       │   │
│       │   ├── check_tick_latency(t0, ticker)
│       │   │   └── If > 2000us → log WARNING "Tick latency spike: %d us"
│       │   │
│       │   └── IF direction != 0:
│       │       └── _build_decision(ticker, direction, result)
│       │           ├── log("THINK %s BUY/SELL score=%.3f regime=%s risk=%s hz=%s/%s")
│       │           └── Return {ticker, direction, score, sizing, thought=SimpleNamespace(...)}
│       │
│       └── IF exception: brain.note_eval_failure(ticker, e) → log WARNING
│
└── RETURN (exit_s, entries)
```

---

## STEP 3: Order Dispatch (_finish_cycle)

```
_finish_cycle(exit_s, decisions, pnl, meta)
│
├── bars = 0
│   FOR EACH ib.pendingTickers():
│       └── IF streamer.update_bar(tk.contract.symbol):
│           └── bars += 1
│               └── Aggregates ticks into 1-min OHLCV bars
│
├── _drain_event_exits()
│   └── FOR EACH sig in streamer.signal_queue (non-blocking):
│       ├── sig.type != "exit"? → skip
│       ├── sig.ticker in _closing? → skip
│       ├── sig.ticker not in open_positions? → skip
│       │
│       └── CLOSE POSITION:
│           ├── _closing.add(ticker)
│           ├── executor.close_position(ticker, streamer)
│           │   └── ib_executor.py:close_position()
│           │       ├── get_last_price(ticker) → current price
│           │       ├── buffer_atr(ticker) → current ATR
│           │       ├── Contract = streamer.contracts[ticker]
│           │       ├── shares = abs(pos.shares)
│           │       ├── action = "SELL" if long, "BUY" if short
│           │       ├── ib.placeOrder(contract, MarketOrder(action, shares))
│           │       └── log("CLOSE %s %s qty=%d price=%.2f")
│           │
│           ├── _exit_reasons[ticker] = sig.reason
│           └── log("EXIT %s (event): %s", ticker, reason)
│
├── FOR EACH es in exit_s (brain exits):
│   ├── es.ticker in _closing? → skip
│   ├── _closing.add(ticker)
│   ├── executor.close_position(ticker, streamer) → market order
│   ├── _exit_reasons[ticker] = es.type
│   └── log("EXIT %s: %s", ticker, reason)
│
├── FOR EACH dec in decisions (entry signals):
│   ├── meta.market_open? → False → skip
│   ├── _can_trade(dec)?
│   │   ├── self._halted? → return False
│   │   ├── dec.thought.verdict direction allowed?
│   │   │   └── TRADING_CONFIG.is_direction_allowed(side)
│   │   └── session active?
│   │       └── TRADING_CONFIG.is_session_active(state.session)
│   │
│   └── _exec_decision(dec)
│       │
│       ├── t = dec.ticker
│       │
│       ├── t in open_positions? → log("SKIP %s open") → return
│       │
│       ├── _portfolio_gate_and_size(t, tk, dec)
│       │   ├── sizing = dec.sizing
│       │   ├── sizing is None or shares <= 0? → return None
│       │   ├── price = (tk.bid + tk.ask) / 2
│       │   ├── allowed, reason = _PORTFOLIO_RISK.pre_trade_risk_gate(t, abs(shares*price))
│       │   │   └── IF drawdown too deep → block
│       │   ├── adj = _PORTFOLIO_RISK.adjust_size(shares, price)
│       │   │   └── risk_scalar reduces size in drawdown
│       │   └── IF adj <= 0 → return None
│       │
│       ├── CHECK SAFETY:
│       │   ├── probe_recovery check → skip if death spiral
│       │   └── safety_enabled? → check_entry_allowed()
│       │       └── Returns False if halted or over limits
│       │
│       ├── EXECUTE ORDER:
│       │   executor.place_bracket(t, dec.thought, price, streamer, sizing, horizon)
│       │   │
│       │   ├── atr = streamer.buffer_atr(ticker)
│       │   ├── atr <= 0 or NaN? → log WARNING → return
│       │   ├── price is NaN? → log WARNING → return
│       │   │
│       │   ├── IF sizing.risk_pass:
│       │   │   ├── shares = int(sizing.shares)
│       │   │   ├── stop = float(sizing.stop_price)
│       │   │   └── target = float(sizing.target_price)
│       │   │
│       │   ├── ELSE (legacy hippocampus sizing):
│       │   │   ├── shares = max(1, int(hippocampus.size_position(win_prob, price, atr)))
│       │   │   ├── stop = max(0.01, price - d * ATR_STOP_MULT * atr)
│       │   │   └── target = max(0.01, price + d * ATR_TARGET_MULT * atr)
│       │   │
│       │   ├── action = "BUY" if direction > 0 else "SELL"
│       │   ├── contract = streamer.contracts[ticker]
│       │   │
│       │   ├── FOR EACH order in ib.bracketOrder(action, shares, price, target, stop):
│       │   │   ├── order.tif = "DAY"
│       │   │   ├── order.outsideRth = ALLOW_EXTENDED_HOURS
│       │   │   └── ib.placeOrder(contract, order)
│       │   │       ├── Parent: LimitOrder/MOC, DAY
│       │   │       ├── Stop: StopOrder, GTC
│       │   │       └── Target: LimitOrder, GTC
│       │   │
│       │   ├── _brackets[ticker] = (stop, target)
│       │   ├── _pending_parent.add(ticker)
│       │   └── log("BRACKET %s %s @ %.2f stop=%.2f target=%.2f qty=%d")
│       │
│       ├── brain.register_position(ticker, price, horizon)
│       │   └── hippocampus._open_positions[ticker] = PositionRecord(...)
│       │
│       └── _attach_position_watchers(ticker)
│           ├── IF already watched → skip
│           ├── _watched.add(ticker)
│           │
│           ├── streamer.attach_exit_watcher(ticker, check)
│           │   └── tk = ticker_subs[ticker]
│           │   └── def _on_tick(_t):
│           │       ├── last = float(_t.last or _t.close or _t.bid or 0)
│           │       ├── IF last <= 0 → return
│           │       ├── reason = check(ticker, last, _t)
│           │       │   └── IF last <= brackets[ticker][0]: return "hard_stop_breach"
│           │       ├── IF reason:
│           │       │   └── signal_queue.put({type:"exit", ticker, reason})
│           │       │
│           │       └── tk.updateEvent += _on_tick  ←── IB TICK CALLBACK
│           │           └── IB pushes tick → this fires on socket thread
│           │               └── Only enqueues signal (no blocking work)
│           │
│           └── streamer.watch_pnl_single(ticker, account)
│               └── ib.reqPnLSingle(account, ticker)
│                   └── Real-time unrealized PnL pushed by IB
│
├── _reflect_closed()
│   └── FOR EACH trade in executor.get_newly_closed_trades():
│       ├── won = trade.pnl > 0
│       ├── source = trade.source ("ib_fill" or "reconciled_exit")
│       │
│       ├── brain.on_trade_close(ticker, won, pnl_pct, direction, source, exit_triggers)
│       │   │
│       │   ├── IRONCLADE GATE:
│       │   │   ├── source in {ib_fill, real_trade, ib_paper, reconciled_exit}?
│       │   │   │   └── ALLOW → proceed to learning
│       │   │   │
│       │   │   └── source NOT in allowed set?
│       │   │       └── BLOCK → log("LEARN BLOCKED %s src=%s") → return
│       │   │
│       │   ├── IF ALLOWED:
│       │   │   ├── episodic.add(alpha, pnl_pct) → k-NN memory
│       │   │   ├── nash.record_pattern(alpha, won, score)
│       │   │   ├── realized.record_outcome(pnl_pct, score, won)
│       │   │   │   └── Updates win rate, average PnL, prediction error
│       │   │   ├── reflector.adapt(won, score)
│       │   │   │   └── Adjusts weights based on outcome
│       │   │   ├── meta_label.record(...)
│       │   │   ├── horizon_bandit.record(...)
│       │   │   └── regime_weights.record(...)
│       │   │
│       │   └── dynamics.adapt_threshold(pred_error)
│       │       └── If error > 0.6 → threshold += 0.01
│       │       └── If error < 0.3 → threshold -= 0.005
│       │
│       ├── _closing.discard(ticker)
│       ├── _watched.discard(ticker)
│       ├── streamer.unwatch_pnl_single(ticker)
│       │   └── ib.cancelPnLSingle(sub)
│       │
│       └── log("REFLECT %s WIN/LOSS pnl=%.4f src=%s")
│
├── monitor.record_cycle(market_open)
│   └── PipelineMonitor: count decisions, failures, timing
│
├── IF pnl is not None:
│   └── daily = float(pnl.dailyPnL)
│       └── hippocampus._daily_pnl = daily
│
├── _sync_portfolio_risk() (throttled every 30s)
│   └── read_portfolio(ib) → net_liquidation
│       └── PortfolioRiskManager.update(net_liq)
│           ├── compute_drawdown() → current vs peak
│           ├── risk_scalar = f(drawdown) → 1.0 (normal) → 0.5 (deep DD)
│           └── pre_trade_risk_gate() uses risk_scalar to size down
│
├── gap = max(0.2, meta.poll - elapsed)
│   └── time.sleep(gap)  ←── MINIMUM 0.2s BETWEEN CYCLES
│
├── _heartbeat() (every 60s)
│   └── log("HEARTBEAT open=%d journal=%d")
│
└── log("CYCLE bars=%d open=%d d=%d x=%d")
    └── bars, open_positions, decisions_count, exits_count
```

---

## STEP 4: System 2 (Background Thread)

```
ConsolidationEngine.start() → threading.Thread(target=_loop, daemon=True)
│
└── _loop() → WHILE _running:
    │   time.sleep(interval)  # 30s
    │
    └── _cycle()
        │
        ├── _update_regime()
        │   ├── alpha = _get_latest_alpha() → BrainState.latest_alpha
        │   ├── regime = halim.get_regime(alpha, prices)
        │   │   └── HALIM service → HTTP → regime dict
        │   │
        │   ├── IF isinstance(regime, dict):
        │   │   └── state.update(regime_multiplier, regime_label, regime_confidence, ...)
        │   │
        │   └── IF label == "unknown":
        │       └── _local_regime()
        │           └── RegimeDetector.detect(prices) → numpy
        │               └── state.update(regime_label, regime_multiplier, regime_source="local_fallback")
        │
        ├── _update_halim()
        │   ├── ticker = max(alpha, key=abs)
        │   ├── mod = halim.get_modifier(ticker, alpha, 0.0, "SCAN")
        │   └── state.update(halim_modifier=mod)
        │
        ├── _run_thinker()
        │   ├── r = thinker.think(alpha, prices, regime, threshold)
        │   │   └── Deliberator: analyze alpha against regime
        │   │       └── Returns Signal(modifier, reasoning)
        │   └── state.update(thinker_modifier=r.modifier)
        │
        ├── _apply_halim_recommendations()
        │   ├── recs = fetch_recommendations(halim_url)
        │   │   └── HTTP POST → HALIM /v1/complete
        │   │       └── Returns structured JSON: [{action, param, value}]
        │   │
        │   ├── valid = [r for r in recs if validate_recommendation(r) is None]
        │   └── state.update(halim_recommendations=valid)
        │       └── Orchestrator reads these and applies to dynamics.threshold, risk params
        │
        ├── news.maybe_refresh()
        │   └── Fetch headlines → sentiment → state.update(news_sentiment)
        │
        ├── _persist_state()
        │   ├── state.to_dict() → runtime/state.json
        │   └── memory.save() → persisted weights
        │
        └── log("S2 | regime=%.2f halim=%.3f thinker=%.4f news=%.2f")
```

---

## STEP 5: Telemetry API (HTTP Server)

```
TelemetryAPI.start() → threading.Thread(target=http_server, daemon=True)
│
└── HTTPServer(("0.0.0.0", 8080), _H) → serve_forever()
    │
    ├── GET /health
    │   └── {status, connected, tickers, position_count, safety_net_enabled, halted, uptime}
    │
    ├── GET /positions
    │   └── [{ticker, direction, shares, entry_price, unrealized_pnl, horizon}, ...]
    │
    ├── GET /brain
    │   └── {threshold, decisions, episodic_count, pred_error, regime, ...}
    │
    ├── GET /safety-net
    │   └── {enabled, halted, daily_pnl, consecutive_losses, ...}
    │
    ├── POST /safety-net
    │   ├── {"action": "enable"} → hippocampus.safety_enabled = True
    │   ├── {"action": "disable"} → hippocampus.safety_enabled = False
    │   ├── {"action": "halt"} → bot._halt("manual")
    │   └── {"action": "resume"} → bot._halted = False
    │
    ├── GET /halim
    │   └── {regime, modifier, insights, recommendations}
    │
    ├── GET /journal
    │   └── [last 50 journal entries from journal_live.jsonl]
    │
    ├── GET /pipeline
    │   └── {healthy, eval_failures, cycle_latency_p95, ...}
    │
    └── GET /config
        └── {direction_mode, horizons, max_positions, ...}
```

---

## STEP 6: Event-Driven Exit Path

```
IB pushes tick update on TCP socket
│
└── ib_insync callback thread:
    │
    ├── ticker.updateEvent fires
    │   └── _on_tick(_t) [attached by _attach_position_watchers]
    │       ├── last = float(_t.last or _t.close or _t.bid or 0)
    │       ├── IF last <= 0 → return
    │       ├── reason = check(ticker, last, _t)
    │       │   └── IF last <= brackets[ticker][0]: return "hard_stop_breach"
    │       │   └── ELSE return None
    │       │
    │       ├── IF reason:
    │       │   └── signal_queue.put({type:"exit", ticker, reason})
    │       │       └── Enqueue to thread-safe Queue (non-blocking)
    │       │
    │       └── NEXT CYCLE (within 1s):
    │           └── _drain_event_exits() picks up the signal
    │               └── executor.close_position(ticker, streamer)
    │                   └── Market order sent to IB
    │
    └── IB pushes PnL update:
        └── reqPnLSingle callback
            └── state.update({ticker: unrealized_pnl})
```

---

## STEP 7: Reconnection Flow

```
IB Gateway drops TCP connection
│
└── _supervise_gateway() detects:
    │
    ├── ib.isConnected() → False
    ├── delay = min(30.0, 1.0 * 2^attempts)  # 1s, 2s, 4s, 8s, 16s, 30s
    ├── time.sleep(delay)
    │
    ├── _reconnect()
    │   └── self.connect(host, port, clientId)
    │       └── ib.connect(host, port, clientId=cid)
    │           └── Retries up to MAX_RECONNECT (5) times
    │
    ├── IF ok:
    │   ├── _seeded_subs.clear() → force re-seed everything
    │   ├── streamer.touch(open_positions) → positions seeded first
    │   └── log("GATEWAY: reconnected — re-seeding position history")
    │
    ├── _resubscribe_all()
    │   └── FOR EACH tracked ticker:
    │       └── streamer.subscribe(t) → ib.reqMktData
    │
    └── NEXT CYCLE:
        └── _sync_subs() → seed_history ONE per cycle
            └── Positions seeded first, then scanner
```

---

## STEP 8: Ironclad Gate (Learning Protection)

```
brain.on_trade_close(ticker, won, pnl_pct, direction, source, exit_triggers)
│
├── IRONCLADE CHECK:
│   ├── source == "ib_fill"? → ALLOW
│   ├── source == "real_trade"? → ALLOW
│   ├── source == "ib_paper"? → ALLOW
│   ├── source == "reconciled_exit"? → ALLOW
│   └── ANY OTHER SOURCE? → BLOCK
│       └── log("LEARN BLOCKED %s src=%s") → return
│
├── IF ALLOWED:
│   ├── episodic.add(alpha, pnl_pct)
│   │   └── k-NN memory: stores indicator vector + outcome
│   │
│   ├── nash.record_pattern(alpha, won, score)
│   │   └── Pattern memory: alpha pattern → win/loss
│   │
│   ├── realized.record_outcome(pnl_pct, score, won)
│   │   └── Updates: win_rate, avg_pnl, prediction_error
│   │
│   ├── reflector.adapt(won, score)
│   │   └── Adjusts indicator weights based on outcome
│   │
│   ├── meta_label.record(...)
│   ├── horizon_bandit.record(...)
│   └── regime_weights.record(...)
│
└── dynamics.adapt_threshold(pred_error)
    ├── pred_error > 0.6? → threshold += 0.01 (raise bar)
    └── pred_error < 0.3? → threshold -= 0.005 (lower bar)
```

---

## STEP 9: Telemetry POST Actions

```
POST /safety-net
│
├── {"action": "enable"}
│   └── hippocampus.safety_enabled = True
│       └── Halts now ACTIVE (bot blocks new entries on violation)
│
├── {"action": "disable"}
│   └── hippocampus.safety_enabled = False
│       └── Halts OFF (bot trades without safety checks)
│
├── {"action": "halt"}
│   └── bot._halt("manual")
│       ├── self._halted = True
│       └── telegram.hafety_halt("manual")
│           └── Bot keeps running, blocks new entries
│
├── {"action": "resume"}
│   └── bot._halted = False
│       └── New entries allowed again
│
└── Bot NEVER stops on any halt — only blocks new entries
```
