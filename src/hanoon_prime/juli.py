"""hanoon_prime.juli — Thin scanner router for the Neuromorphic Brain.

Brain-first: every evaluated ticker yields a Verdict from
``NeuromorphicBrain.decide_entry`` — never a silent omission. Tickers not
in the rotating eval window are scheduling, not decisions: they produce no
Verdict this cycle and age out of ``/verdicts`` naturally.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Any

from .brain.orchestrator import NeuromorphicBrain
from .brain.policy.verdict import Verdict
from .brain.shared_state import BrainState
from .data.budget import DataBudget
from .data.scanner import IBScanner, ScanResult
from .juli_feed import JuliFeed

log = logging.getLogger(__name__)
MAX_CANDIDATES: int = 20
# Rotating EVAL_WINDOW keeps flow continuous (no THINK burst, then silence).
EVAL_WINDOW: int = 4


def _fmt_verdict(v: Verdict) -> str:
    """One verdict as a compact, greppable log token."""
    where = f"{v.stage}:{v.reason}" if v.stage else (v.reason or v.action)
    return f"{v.ticker}:{v.action}({v.score:.3f})[{where}]"


class JuliBrain:
    """Scanner + router. NeuromorphicBrain makes all decisions."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self.scanner = IBScanner(ib_client)
        self.budget = DataBudget()
        self._state = BrainState()
        self.brain = NeuromorphicBrain(brain_state=self._state)
        self._candidates: list[ScanResult] = []
        self._last_alloc: float = 0.0
        self.feed = JuliFeed(self._state)
        self._recent_verdicts: collections.deque[Verdict] = collections.deque(
            maxlen=200
        )
        self._lock_held: bool = False
        self.brain.start()

    def tick(
        self,
        watch: set[str] | dict[str, Any],
        snapshot: dict[str, Any] | Any,
        streamer: Any,
        held_positions: set[str] | list[str],
        closing: set[str] | None = None,
        pos_info: dict[str, dict[str, Any]] | None = None,
        session: str = "rth",
    ) -> tuple[list[dict[str, Any]], list[Verdict]]:
        """Brain-first decision loop: returns (exits, verdicts) with lock."""
        if self._lock_held:
            return [], []
        self._lock_held = True
        self.brain.begin_entry_cycle()
        try:
            self._data_preamble(streamer, snapshot, held_positions)
            universe = sorted(set(watch) | set(held_positions or ()))
            if not universe:
                self._eval_off = 0
                return [], []
            verdicts = self._eval_window(universe, snapshot, held_positions, session)
            exits = self._evaluate_exits(
                set(held_positions or ()), snapshot, closing or set(), pos_info or {}
            )
            policy_exits = self.brain.state.get("policy_exits")  # array-safe list
            if policy_exits:
                exits += list(policy_exits)
            return exits, verdicts
        finally:
            self._lock_held = False

    def _data_preamble(
        self, streamer: Any, snapshot: Any, held_positions: set[str] | list[str]
    ) -> None:
        """Scanner + reference-feed upkeep (idempotent, tolerant)."""
        self.feed.ensure_refs(streamer)
        self._sync_and_scan()
        if callable(snapshot):
            self._maybe_screen(snapshot)
        self.feed.fallback_regime()
        self._maybe_allocate(set(held_positions or ()))

    def _eval_window(
        self,
        universe: list[str],
        snapshot: Any,
        held_positions: set[str] | list[str],
        session: str,
    ) -> list[Verdict]:
        """Score the rotating EVAL_WINDOW slice (scheduling, no decisions)."""
        off = int(getattr(self, "_eval_off", 0)) % len(universe)
        window = universe[off : off + EVAL_WINDOW]
        self._eval_off = (off + EVAL_WINDOW) % len(universe)
        verdicts = [
            self.brain.decide_entry(
                ticker,
                self._snap_for(snapshot, ticker),
                set(held_positions or ()),
                session,
            )
            for ticker in window
        ]
        self._recent_verdicts.extend(verdicts)
        if verdicts:
            log.info("EVAL %s", " ".join(_fmt_verdict(v) for v in verdicts))
        return verdicts

    @staticmethod
    def _snap_for(snapshot: Any, ticker: str) -> dict[str, Any] | None:
        """Resolve a snapshot from a callable or a dict (never raises)."""
        try:
            if callable(snapshot):
                snap = snapshot(ticker)
            elif isinstance(snapshot, dict):
                snap = snapshot.get(ticker)
            else:
                snap = None
        except Exception:
            return None
        return snap if isinstance(snap, dict) else None

    def _sync_and_scan(self) -> None:
        """Sync scanner results and start new scan if due."""
        if self.scanner.should_scan():
            try:
                self.scanner.scan("most_active")
            except Exception as e:
                log.warning("Scan start failed: %s", e)
        try:
            if results := self.scanner.collect():
                self._candidates = results
        except Exception as e:
            log.debug("Scan collect error: %s", e)

    def _maybe_screen(self, get_snapshot: Any) -> None:
        """Screen candidates + publish cross-asset ref prices."""
        if not self._candidates:
            return
        snaps = [get_snapshot(c.symbol) for c in self._candidates[:MAX_CANDIDATES]]
        log.info(
            "SCREEN: %d/%d passed",
            sum(1 for s in snaps if s and s.get("last", 0) > 0),
            len(self._candidates),
        )
        self.feed.publish_ref_prices(get_snapshot)

    def _maybe_allocate(self, positions: set[str]) -> None:
        """Allocate data budget periodically."""
        if time.time() - self._last_alloc < 5.0:
            return
        self._last_alloc = time.time()
        self.budget.allocate(
            positions, [c.symbol for c in self._candidates[:MAX_CANDIDATES]]
        )

    def _evaluate_exits(
        self,
        positions: set[str],
        snapshot: Any,
        closing: set[str],
        pos_info: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate open positions for exit signals (direction-aware)."""
        exits = []
        watched = []
        for t in positions:
            if t in closing:
                continue
            snap = self._snap_for(snapshot, t)
            if snap is None or snap.get("last", 0) <= 0:
                continue
            info = pos_info.get(t, {})
            direction = int(info.get("direction", 1)) or 1
            entry = info.get("entry_price") or snap["last"]
            if entry > 0 and not self.brain.exits.is_registered(t):
                self.brain.register_position(t, entry)
            sig = self.brain.check_exit(
                t,
                snap["last"],
                direction=direction,
                stop_price=info.get("stop_price") or None,
            )
            if sig.should_exit:
                exits.append({"ticker": t, "reason": sig.reason, "type": sig.exit_type})
                log.info("EXIT SIGNAL %s: %s", t, sig.reason)
            else:
                watched.append(
                    f"{t}@{snap['last']:.2f} {'LONG' if direction > 0 else 'SHORT'}"
                )
        if watched:
            log.info("WATCH %s", " ".join(watched))
        return exits

    def on_trade_close(
        self, ticker: str, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """Route trade close to NeuromorphicBrain — all learning here."""
        self.brain.on_trade_close(ticker, won, pnl_pct, direction)
