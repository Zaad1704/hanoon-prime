"""Live-wiring smoke test — real Gateway, false trades, zero orders.

Starts the production bot stack against the real IB Gateway (paper
port 4002), seeds real historical bars, drives real cycles, then gives
the bot a FALSE SENSE of trading: scripted closed trades are injected
into the executor's close queue (the exact dict shape _record_close
produces on a real IB fill). _reflect_closed() therefore runs the
identical fan-out it would on live fills. No order is ever sent: the
sleep manager reports the market closed (weekend), which gates
_exec_decision, and no decisions occur anyway without live ticks.

Verifies after injection: reflect fan-out (7 learners), episodic
growth, realized-EV stats, memory WR, threshold adaptation, bandit /
meta-label / regime-weight persistence, learned-exit adaptation, and
telemetry serving live brain state.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanoon_prime.ib_adapter import IBStreamingBot  # noqa: E402
from hanoon_prime.immune import TELEMETRY_PORT  # noqa: E402
from hanoon_prime.telemetry import TelemetryAPI  # noqa: E402

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
LOG = logging.getLogger("smoke")
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record and print one smoke check."""
    CHECKS.append((name, ok, detail))
    LOG.info("%s %s %s", "PASS" if ok else "FAIL", name, detail)


def http_get(path: str) -> dict:
    """GET a telemetry endpoint (real HTTP, like the webapp)."""
    with urllib.request.urlopen(  # noqa: S310
        f"http://127.0.0.1:{TELEMETRY_PORT}{path}", timeout=5
    ) as r:
        return json.loads(r.read().decode())


def scripted_trades() -> list[dict]:
    """False trade lifecycle: 3 wins / 2 losses across horizons+directions."""
    return [
        {
            "ticker": "SMOKE_A",
            "pnl": 0.012,
            "return_pct": 0.012,
            "direction": 1,
            "entry_price": 100.0,
            "shares": 50,
            "horizon": "scalp",
            "exit": "trail",
        },
        {
            "ticker": "SMOKE_B",
            "pnl": -0.008,
            "return_pct": -0.008,
            "direction": -1,
            "entry_price": 200.0,
            "shares": 25,
            "horizon": "scalp",
            "exit": "brain_exit",
        },
        {
            "ticker": "SMOKE_C",
            "pnl": 0.020,
            "return_pct": 0.020,
            "direction": 1,
            "entry_price": 300.0,
            "shares": 10,
            "horizon": "swing",
            "exit": "smoke_bracket",
        },
        {
            "ticker": "SMOKE_D",
            "pnl": -0.005,
            "return_pct": -0.005,
            "direction": -1,
            "entry_price": 150.0,
            "shares": 40,
            "horizon": "multihour",
            "exit": "manual_or_bracket",
        },
        {
            "ticker": "SMOKE_E",
            "pnl": 0.003,
            "return_pct": 0.003,
            "direction": 1,
            "entry_price": 90.0,
            "shares": 80,
            "horizon": "scalp",
            "exit": "trail",
        },
    ]


def phase_connect(bot: IBStreamingBot) -> None:
    """Real connect, real telemetry, real historical bar seeding."""
    bot.connect()  # IB_PAPER_PORT 4002, retry logic — production path
    LOG.info("CONNECTED to Gateway, account=%s", bot.account)
    telemetry = TelemetryAPI(bot, RUNTIME / "journal_live.jsonl")
    telemetry.start()
    time.sleep(0.3)
    health = http_get("/health")
    check("telemetry /health", bool(health), json.dumps(health)[:120])
    for t in ("AAPL", "MSFT", "NVDA"):
        try:
            bot.streamer.subscribe(t)
            bot.streamer.seed_history(t)  # 2 D of 1-min bars — works closed
        except Exception as e:  # noqa: BLE001
            LOG.warning("seed %s failed: %s", t, e)
    bars = sum(
        len(getattr(bot.streamer.buffers.get(t), "close", []) or [])
        for t in ("AAPL", "MSFT", "NVDA")
    )
    check("real historical bars seeded", bars > 0, f"{bars} bars in streamer")


def phase_false_trades(bot: IBStreamingBot) -> None:
    """Register brain-side entries, inject scripted closes, run real cycles."""
    trades = scripted_trades()
    brain = bot.juli.brain
    for tr in trades:
        brain.register_position(tr["ticker"], tr["entry_price"], horizon=tr["horizon"])
    LOG.info("Registered %d brain-side positions", len(trades))
    for i, tr in enumerate(trades):
        bot.executor._closed_trades.append(
            {
                k: tr[k]
                for k in (
                    "ticker",
                    "pnl",
                    "return_pct",
                    "direction",
                    "entry_price",
                    "shares",
                )
            }
        )
        bot._exit_reasons[tr["ticker"]] = tr["exit"]
        bot._cycle(0.1, None)  # real cycle → _reflect_closed → full fan-out
        time.sleep(0.2)
        snap = brain.snapshot()
        LOG.info(
            "close %d/%d reflected: episodic=%s realized_n=%s",
            i + 1,
            len(trades),
            snap.get("episodic_size"),
            snap.get("realized", {}).get("n"),
        )


def phase_replay(bot: IBStreamingBot) -> None:
    """Live-like replay: real bars through the real tick path, no orders.

    Calls juli.tick directly — decisions and exit signals are computed
    from the seeded real historical bars, but _exec_decision is never
    invoked, so nothing is sent to IB.
    """
    brain = bot.juli.brain
    sel_before = (brain.snapshot().get("horizon_bandit", {}) or {}).get("selects", 0)
    brain.register_position("AAPL", 100.0, horizon="scalp")
    # Positions drive both exit evaluation and entry evaluation
    # (_evaluate_entries iterates tracked | positions).
    watch = {"AAPL", "MSFT", "NVDA"}
    # False live quotes: _snapshot returns None without a bid/ask (correct
    # pipeline behavior), so the replay fakes quotes the way the false
    # closes faked fills — real bars + synthetic quote, zero orders.
    for sym in watch:
        tk = bot.streamer.ticker_subs.get(sym)
        if tk is not None:
            tk.bid, tk.ask, tk.last, tk.hasBidAsk = 100.0, 100.02, 100.01, True
    think_before = int(getattr(brain, "_decision_count", 0))
    dec_total, exit_total = 0, 0
    for _ in range(3):
        # S2's slow cycles stamp a 2s refractory that suppresses the fast
        # path — clear it so the replay exercises real scoring, not the
        # refractory stub.
        bot.juli._state.update(refractory_until=0.0)
        exits, decisions = bot.juli.tick(watch, bot._snapshot, bot.streamer, set())
        dec_total += len(decisions)
        exit_total += len(exits)
        for d in decisions:
            assert (
                "sizing" in d and "regime_canon" in d
            ), f"bad decision keys: {list(d)}"
    sel_after = (brain.snapshot().get("horizon_bandit", {}) or {}).get("selects", 0)
    thought_after = int(getattr(brain, "_decision_count", 0))
    check(
        "replay: brain scored 9 ticker-ticks",
        thought_after - think_before >= 9,
        f"decisions={dec_total} exits={exit_total}"
        f" thinks=+{thought_after - think_before}",
    )
    check(
        "replay: bandit selection ran",
        sel_after > sel_before,
        f"selects {sel_before} -> {sel_after}",
    )
    check(
        "replay: regime context present",
        bool(brain._last_regime),
        str(dict(list(brain._last_regime.items())[:3]))[:100],
    )
    check(
        "replay: zero orders sent",
        not bot.executor._closed_trades and not bot._order_placed_ts,
        "close queue and order log untouched",
    )


def phase_verify(bot: IBStreamingBot, before: dict) -> None:
    """Assert every mechanism fired, on live state + persisted files."""
    brain = bot.juli.brain
    snap = brain.snapshot()
    realized = snap.get("realized", {}) or {}
    memory = snap.get("memory", {}) or {}
    d = lambda k, sub: (snap.get(sub, {}) or {}).get(k, 0) - (
        before.get(sub, {}) or {}
    ).get(
        k, 0
    )  # noqa: E731
    check(
        "reflect fan-out ran",
        d("episodes", "memory") >= 5,
        f"memory episodes +{d('episodes', 'memory')}",
    )
    check(
        "realized-EV ate 5 trades",
        d("total", "realized") == 5,
        f"realized total +{d('total', 'realized')}",
    )
    check(
        "memory WR tracking",
        d("total_trades", "memory") >= 5,
        f"total_trades +{d('total_trades', 'memory')} win_rate={memory.get('win_rate')}",
    )
    t = float(snap.get("threshold", 0.0))
    check("threshold sane (adapt bounded)", 0.40 <= t <= 0.80, f"threshold={t}")

    def _bandit_total(s: dict) -> int:
        """Sum all arm trial counts across regime cells."""
        arms = (s.get("horizon_bandit", {}) or {}).get("arms", {}) or {}
        return sum(int(a.get("n", 0)) for v in arms.values() for a in v)

    check(
        "bandit trained (+5 rewards)",
        _bandit_total(snap) - _bandit_total(before) >= 5,
        f"arm trials +{_bandit_total(snap) - _bandit_total(before)}",
    )
    check(
        "meta-label fed",
        bool(snap.get("meta_label", {})),
        str((snap.get("meta_label", {}) or {}))[:100],
    )
    check(
        "learned exits adapted",
        brain._learned_exit.count >= 5,
        f"recorded={brain._learned_exit.count}",
    )
    genome = brain.genome.get_genome()
    check(
        "genome read-model live",
        bool(genome.get("weights")),
        f"v{genome.get('version')} advisor={genome.get('advisor_delta')}",
    )
    trend = [100 * (1 + 0.002 * i) for i in range(40)]
    bot.juli._state.update(latest_prices=trend)
    bot.juli.feed.fallback_regime()
    label = bot.juli._state.get("regime_label", "unknown")
    check(
        "regime fallback publishes",
        label != "unknown",
        f"label={label} src={bot.juli._state.get('regime_source', '?')}",
    )
    brain_http = http_get("/brain")
    check(
        "telemetry /brain live",
        bool(brain_http),
        f"regime={brain_http.get('brain_state', {}).get('regime_label', '?')}",
    )
    pipe = http_get("/pipeline")
    check(
        "telemetry /pipeline live",
        bool(pipe.get("vitals")),
        f"healthy={pipe.get('healthy')}",
    )
    for fname in (
        "juli_horizon_bandit.json",
        "juli_meta_label.json",
        "juli_regime_weights.json",
        "juli_state.json",
        "juli_realized.json",
    ):
        f = RUNTIME / fname
        ok = f.exists() and f.stat().st_size > 2
        check(
            f"persistence {fname}",
            ok,
            f"{f.stat().st_size}B" if ok else "missing/empty",
        )


def main() -> None:
    """Run the full live smoke sequence and report a checklist."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logging.getLogger("ib_insync").setLevel(logging.WARNING)
    bot = IBStreamingBot(account="PAPER")
    pnl = None
    try:
        phase_connect(bot)
        bot._cycle(0.1, None)  # one real cycle pre-injection: tick path sanity
        check("real cycle ran clean (market closed)", True, "no exception")
        before = bot.juli.brain.snapshot()
        phase_false_trades(bot)
        phase_replay(bot)
        phase_verify(bot, before)
    except Exception as e:  # noqa: BLE001
        LOG.exception("SMOKE ABORT: %s", e)
        check("sequence completed", False, str(e))
    finally:
        try:
            pnl = bot._start_pnl()
            bot._cleanup(pnl)
        except Exception:  # noqa: BLE001
            LOG.exception("cleanup issue")
    LOG.info("=" * 60)
    failed = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        LOG.info("%s %-42s %s", "PASS" if ok else "FAIL", name, detail)
    LOG.info("SMOKE RESULT: %d/%d passed", len(CHECKS) - len(failed), len(CHECKS))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
