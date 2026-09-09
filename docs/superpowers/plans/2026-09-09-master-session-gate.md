# Master Session Gate + Pre-Market Activation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate pre-market (04:00–09:30 ET) as a trading session and make the session gate control the *whole* system — when a session is deactivated, the bot stops all brain cycles and HALIM stops inference; only telemetry stays up.

**Architecture:** Unify session ids (`pre_market`/`rth`/`post_market`/`overnight`) in `SleepManager`, add `effective_state(enabled)` as the single gate (clock window AND TradingConfig enabled-set). The bot's `_cycle` goes to a deep-sleep keepalive when inactive. Equity resolves via a fallback chain so pre-market ENTERs can size. Telemetry serves `GET /session`; HALIM (separate process) polls it and sleeps on any failure.

**Tech Stack:** Python 3.12, `zoneinfo`, stdlib `http.server`, `threading`, `urllib`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-09-master-session-gate-design.md`

## Global Constraints
- **Never run black/isort on `src/hanoon_prime/ib_cycle.py` or `src/hanoon_prime/telemetry.py`** (both black-excluded in `.pre-commit-config.yaml:${`; ib_cycle also excluded from isort). Match each file's existing formatting by hand.
- `halim/**` is excluded from ruff/black/isort/mypy — do not rely on hooks there, keep it clean anyway.
- pre-commit mypy hook runs `mypy --strict` over `src/hanoon_prime` (package config: pyproject.toml `[tool.mypy]`, excludes only `ib_adapter|ib_streamer|ib_executor|ib_compat|ib_cycle`). All edited files EXCEPT ib_cycle must be strict-typed (annotations on every def).
- R3: functions ≤40 lines, nesting ≤3. R3b: files ≤200 lines — `sleep_manager.py` and `account_equity.py` are NOT on the length-exempt list; keep them under 200.
- TDD on every task: write failing test → run to see it fail → implement → run to see it pass → commit.
- Commit messages lowercase `type(scope): summary` matching repo history.
- Full suite must stay green: `.venv/bin/python -m pytest -q` (currently 573 pass / 3 skip).

---

### Task 1: Unified session ids + `effective_state` in SleepManager

**Files:**
- Modify: `src/hanoon_prime/monitor/sleep_manager.py` (whole file rewrite, keep `minutes_to_close`/`is_eod_window`/`force_active`)
- Modify: `tests/test_bugfixes.py:227-228` (RTH expectation — see below)
- Create: `tests/test_session_gate.py`

**Interfaces:**
- Produces:
  - `SleepState(active: bool = True, session: str = "rth", reason: str = "")` (unchanged dataclass)
  - `SleepManager.get_state(ib_connected: bool = True, now: datetime | None = None) -> SleepState`
  - `SleepManager.effective_state(enabled, ib_connected: bool = True, now: datetime | None = None) -> SleepState` — `active = clock_window_active AND enabled.is_session_active(session)`
  - module const `SESSION_IDS: frozenset[str]`
  - `Protocol` type `SessionEnabled` (duck: `is_session_active(session: str) -> bool`)
- Session windows: `pre_market` 04:00–09:30, `rth` 09:30–16:00, `post_market` 16:00–20:00, `overnight` 20:00–04:00. `_CLOCK_ACTIVE` = pre_market/rth True, post_market/overnight False. Weekend always inactive.

- [ ] **Step 1: Write the failing tests**

`tests/test_session_gate.py`:
```python
"""Master session gate: unified ids, window classification, effective_state."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanoon_prime.brain.policy.trading_policy import TradingConfig  # noqa: E402
from hanoon_prime.monitor.sleep_manager import (  # noqa: E402
    SESSION_IDS,
    SleepManager,
)

ET = ZoneInfo("America/New_York")


def _dt(day: int, h: int, mi: int) -> datetime:
    return datetime(2026, 9, 14, h, mi, tzinfo=ET)  # Monday 2026-09-14


class TestWindowClassification:
    def test_pre_market_boundary(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 4, 0))
        assert st.active is True
        assert st.session == "pre_market"

    def test_pre_market_just_before_rth(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 9, 29))
        assert st.session == "pre_market"

    def test_rth_starts_at_0930(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 9, 30))
        assert st.active is True
        assert st.session == "rth"

    def test_rth_ends_at_1600(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 15, 59))
        assert st.session == "rth"

    def test_post_market_inactive(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 16, 0))
        assert st.active is False
        assert st.session == "post_market"

    def test_post_market_ends_at_2000(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 19, 59))
        assert st.session == "post_market"

    def test_overnight_inactive(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 2, 0))
        assert st.active is False
        assert st.session == "overnight"

    def test_weekend_inactive(self) -> None:
        st = SleepManager().get_state(now=datetime(2026, 9, 12, 12, 0, tzinfo=ET))
        assert st.active is False
        assert st.session == "weekend"

    def test_ib_disconnected_inactive(self) -> None:
        st = SleepManager().get_state(ib_connected=False, now=_dt(14, 10, 0))
        assert st.active is False

    def test_session_ids_match_config_suffixes(self) -> None:
        cfg = TradingConfig()
        suffixes = {a[len("session_"):] for a in dir(cfg) if a.startswith("session_")}
        assert SESSION_IDS == suffixes


class TestEffectiveState:
    def test_config_disable_turns_system_off(self) -> None:
        cfg = TradingConfig()
        cfg.session_pre_market = False
        st = SleepManager().effective_state(cfg, now=_dt(14, 5, 0))
        assert st.active is False
        assert st.reason == "session_disabled"

    def test_enabled_pre_market_stays_active(self) -> None:
        st = SleepManager().effective_state(TradingConfig(), now=_dt(14, 5, 0))
        assert st.active is True
        assert st.session == "pre_market"

    def test_post_market_off_even_with_config(self) -> None:
        st = SleepManager().effective_state(TradingConfig(), now=_dt(14, 17, 0))
        assert st.active is False
```

In `tests/test_bugfixes.py:214-228` change `test_rth_is_active`'s assertion `assert state.session == "RTH"` to `assert state.session == "rth"` and update the docstring comment to "Monday 10:00 AM ET — RTH (unified lowercase id)".

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_session_gate.py -q --no-header --no-cov`
Expected: FAIL — `session == "overnight"`/`"pre_market"`/etc because the current `get_state` returns `RTH`/`pre_post`/`overnight`.

- [ ] **Step 3: Rewrite `src/hanoon_prime/monitor/sleep_manager.py`**

```python
"""monitor.sleep_manager — Market session awareness.

Decides whether the bot should be ACTIVE (trading) or SLEEPING (idle).
Honors market sessions (pre_market/RTH/post_market/overnight), weekends,
and holidays.

Session ids MATCH TradingConfig keys exactly — pre_market / rth /
post_market / overnight. No capitalized "RTH", no legacy "pre_post":
an id mismatch silently short-circuits the gate (is_session_active
falls back to True), so the contract lives in SESSION_IDS below.

Uses zoneinfo for proper US/Eastern timezone (handles DST automatically).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Canonical session ids — must equal TradingConfig.session_* suffixes.
SESSION_IDS = frozenset({"pre_market", "rth", "post_market", "overnight"})

# Clock-level default activity: post_market/overnight remain OFF even
# with config enabled; only pre_market + rth trade by default.
_CLOCK_ACTIVE = {
    "pre_market": True,
    "rth": True,
    "post_market": False,
    "overnight": False,
}

_PRE_START = (4, 0)
_RTH_START = (9, 30)
_RTH_END = (16, 0)
_POST_END = (20, 0)


class SessionEnabled(Protocol):
    """Anything exposing the TradingConfig session policy."""

    def is_session_active(self, session: str) -> bool: ...


def _classify_session(now_et: datetime) -> str:
    """Map an ET-aware datetime to a canonical session id."""
    minute_of_day = now_et.hour * 60 + now_et.minute

    def inside(start: tuple[int, int], end: tuple[int, int]) -> bool:
        return (start[0] * 60 + start[1]) <= minute_of_day < (end[0] * 60 + end[1])

    if inside(_PRE_START, _RTH_START):
        return "pre_market"
    if inside(_RTH_START, _RTH_END):
        return "rth"
    if inside(_RTH_END, _POST_END):
        return "post_market"
    return "overnight"


@dataclass
class SleepState:
    active: bool = True
    session: str = "rth"
    reason: str = ""


class SleepManager:
    """Market session awareness."""

    def __init__(self) -> None:
        """Initialize with no forced override."""
        self._force_active: bool = False

    def _et_now(self, now: datetime | None) -> datetime:
        """Resolve the reference clock (injectable for tests)."""
        if now is not None:
            return now
        return datetime.now(timezone.utc).astimezone(_ET)

    def minutes_to_close(self) -> float:
        """Minutes until RTH close (4:00 PM ET). Negative if past close."""
        now = datetime.now(timezone.utc).astimezone(_ET)
        close = now.replace(
            hour=_RTH_END[0], minute=_RTH_END[1], second=0, microsecond=0
        )
        return (close - now).total_seconds() / 60.0

    def is_eod_window(self, minutes: float = 5.0) -> bool:
        """True if within `minutes` of RTH close on a weekday."""
        now = datetime.now(timezone.utc).astimezone(_ET)
        if now.weekday() >= 5:
            return False
        remaining = self.minutes_to_close()
        return 0 < remaining <= minutes

    def get_state(
        self, ib_connected: bool = True, now: datetime | None = None
    ) -> SleepState:
        """Clock + connectivity gate (session id is always classified)."""
        now_et = self._et_now(now)
        if self._force_active:
            return SleepState(
                active=True, session=_classify_session(now_et), reason="manual"
            )
        if now_et.weekday() >= 5:
            return SleepState(active=False, session="weekend", reason="Weekend")
        session = _classify_session(now_et)
        if not ib_connected:
            return SleepState(active=False, session=session, reason="IB disconnected")
        base = _CLOCK_ACTIVE.get(session, False)
        return SleepState(
            active=base, session=session, reason="Market open" if base else "Off hours"
        )

    def effective_state(
        self,
        enabled: SessionEnabled,
        ib_connected: bool = True,
        now: datetime | None = None,
    ) -> SleepState:
        """System gate: clock window AND the enabled-set must agree.

        This is the single gate the whole stack (bot, telemetry, HALIM)
        reads. Disabling a session via TradingConfig puts the whole
        system to sleep.
        """
        state = self.get_state(ib_connected=ib_connected, now=now)
        if state.active and not enabled.is_session_active(state.session):
            return SleepState(
                active=False, session=state.session, reason="session_disabled"
            )
        return state

    def force_active(self, active: bool) -> None:
        """Override market hours check."""
        self._force_active = active
```

The existing `tests/test_bugfixes.py` weekend/overnight/RTH tests patch `hanoon_prime.monitor.sleep_manager.datetime` — `_et_now` still calls `datetime.now(timezone.utc).astimezone(_ET)` when `now is None`, so the patches keep working.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_session_gate.py tests/test_bugfixes.py -q --no-header --no-cov`
Expected: PASS (all new + existing Bug#3 tests).

- [ ] **Step 5: Format + commit**

```bash
.venv/bin/black src/hanoon_prime/monitor/sleep_manager.py tests/test_session_gate.py
.venv/bin/isort src/hanoon_prime/monitor/sleep_manager.py tests/test_session_gate.py tests/test_bugfixes.py
git add -A
git commit -m "feat(session): unified session ids + effective_state gate"
```

---

### Task 2: Session-aware cycle loop — deep sleep

**Files:**
- Modify: `src/hanoon_prime/ib_cycle.py:312-355` (`_cycle`, `_run_brain_cycle`), add `_sleep_tick`
- Modify: `src/hanoon_prime/monitor/pipeline.py:49-73` (`record_cycle` session param)
- Create: tests in `tests/test_session_gate.py` (loop gate)

**Interfaces:**
- Consumes: `SleepManager.effective_state` + `SessionEnabled` (Task 1). `TRADING_CONFIG` already imported in ib_cycle.
- Produces:
  - `BotCycleMixin._sleep_tick(state: SleepState, poll: float) -> None` — deep-sleep keepalive
  - `PipelineMonitor.record_cycle(market_open: bool, session: str = "rth") -> None` (vitals gain `session` + `session_active`)
  - `_finish_cycle(..., session: str = "rth")` param carried to `monitor.record_cycle`

- [ ] **Step 1: Write the failing tests (loop gate)**

Append to `tests/test_session_gate.py`:
```python
from types import SimpleNamespace
from unittest.mock import MagicMock  # noqa: E402

from hanoon_prime import ib_cycle as _ibc  # noqa: E402
from hanoon_prime.ib_cycle import BotCycleMixin  # noqa: E402
from hanoon_prime.monitor.sleep_manager import SleepState  # noqa: E402


def _mixin() -> BotCycleMixin:
    mixin = BotCycleMixin.__new__(BotCycleMixin)
    mixin.streamer = MagicMock()
    mixin.streamer.ticker_subs = {}
    mixin.streamer.last_seen = {}
    mixin.streamer.drain_signals.return_value = []
    mixin.streamer.update_bar.return_value = False
    mixin.hippocampus = MagicMock()
    mixin.hippocampus._open_positions = {}
    mixin.hippocampus._consecutive_losses = 0
    mixin.executor = MagicMock()
    mixin.executor.get_newly_closed_trades.return_value = []
    juli = MagicMock()
    juli._candidates = []
    juli._state = {}
    juli._recent_verdicts = []
    juli.tick.return_value = ([], [])
    juli.budget.get_all_tracked.return_value = set()
    mixin.juli = juli
    juli.brain = MagicMock()
    juli.brain._consolidation = None
    mixin.monitor = MagicMock()
    mixin._closing = set()
    mixin._last_beat = 0.0
    mixin._last_policy_sync = 0.0
    mixin._sleeping = False
    mixin.ib = MagicMock()
    mixin.ib.pendingTickers.return_value = []
    mixin._exit_reasons = {}
    mixin.journal = MagicMock()
    return mixin


class TestDeepSleepLoop:
    def test_asleep_cycle_skips_brain(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _ibc._SLEEP_MGR,
            "effective_state",
            lambda enabled, ib_connected=True, now=None: SleepState(
                active=False, session="overnight", reason="Off hours"
            ),
        )
        mixin = _mixin()
        mixin._cycle(0.0, None)
        mixin.juli.tick.assert_not_called()
        mixin.monitor.record_cycle.assert_called()

    def test_awake_cycle_runs_brain(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _ibc._SLEEP_MGR,
            "effective_state",
            lambda enabled, ib_connected=True, now=None: SleepState(
                active=True, session="rth", reason="Market open"
            ),
        )
        mixin = _mixin()
        mixin._cycle(0.0, None)
        mixin.juli.tick.assert_called_once()

    def test_asleep_cycle_still_honors_flatten(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _ibc._SLEEP_MGR,
            "effective_state",
            lambda enabled, ib_connected=True, now=None: SleepState(
                active=False, session="overnight", reason="Off hours"
            ),
        )
        mixin = _mixin()
        mixin._check_manual_flatten = lambda: True  # type: ignore[method-assign]
        mixin._cycle(0.0, None)
        mixin.juli.tick.assert_not_called()
        mock = mixin.monitor.record_cycle
        assert len(mock.call_args_list) >= 1
```

Also extend `tests/test_pipeline_monitor.py` near `test_record_cycle_publishes_vitals`:
```python
def test_record_cycle_carries_session(tmp_path):
    bot = make_bot(tmp_path)
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True, session="pre_market")
    v = mon.snapshot()["vitals"]
    assert v["session"] == "pre_market"
    assert v["session_active"] is True
    mon.record_cycle(market_open=False, session="overnight")
    v = mon.snapshot()["vitals"]
    assert v["session_active"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_session_gate.py tests/test_pipeline_monitor.py -q --no-header --no-cov`
Expected: FAIL — `mixin.juli.tick` still called when asleep; `record_cycle` has no `session` kwarg.

- [ ] **Step 3: Implement the deep-sleep loop**

In `src/hanoon_prime/monitor/pipeline.py:49` change the signature and vitals:
```python
    def record_cycle(self, market_open: bool, session: str = "rth") -> None:
        """Publish vitals from the cycle thread (cheap, lock-light)."""
        ...
        with self._lock:
            self._vitals = {
                "ts": time.time(),
                "market_open": market_open,
                "session": session,
                "session_active": market_open,
                "ib_connected": connected,
                "bar_sizes": sizes,
                "decision_count": decisions,
                "journal_bytes": self._journal_path_size(),
            }
            ...
```
(Only the two added keys; rest of the method unchanged.)

In `src/hanoon_prime/ib_cycle.py` modify `_cycle` (around line 312). Insert the gate at the very top of the `try`:
```python
        try:
            state = _SLEEP_MGR.effective_state(TRADING_CONFIG)
            if not state.active:
                self._sleep_tick(state, poll)
                return
            if getattr(self, "_sleeping", False):
                self._sleeping = False
                log.info("SESSION WAKE: %s active", state.session)
            self._supervise_gateway()
            ...
```
Modify `_run_brain_cycle` (line 334):
```python
        mkt_state = _SLEEP_MGR.effective_state(TRADING_CONFIG)
```
and its `_finish_cycle(...)` call to add `session=mkt_state.session`.

Modify `_finish_cycle` signature (line 357) to add `session: str = "rth"` after `meta`:
```python
        meta: CycleMeta,
        session: str = "rth",
    ) -> None:
```
and its final `self.monitor.record_cycle(market_open)` (line 388) → `self.monitor.record_cycle(market_open, session=session)`.

Add `_sleep_tick` after `_cycle`:
```python
    def _sleep_tick(self, state: SleepState, poll: float) -> None:
        """Keepalive for a deactivated session: no brain, no orders, no HALIM.

        Gateway supervision and on-demand flatten stay live so risk
        controls still work while the system is fully asleep.
        """
        if not getattr(self, "_sleeping", False):
            self._sleeping = True
            log.info("SESSION SLEEP: %s inactive — whole system idle", state.session)
        try:
            if self._check_manual_flatten() or self._check_eod_flatten():
                self._finish_cycle(
                    [], [], None,
                    CycleMeta(poll, time.monotonic(), False),
                    session=state.session,
                )
                self._sleeping = True
                return
        except Exception as exc:
            log.error("Sleep flatten failed: %s", exc)
        try:
            self._supervise_gateway()
        except Exception as exc:
            log.debug("Gateway supervise during sleep: %s", exc)
        self._heartbeat()
        self.monitor.record_cycle(False, session=state.session)
        time.sleep(max(CYCLE_FLOOR, poll))
```
Requires imports in ib_cycle.py — `SleepState` from `.monitor.sleep_manager` (extend the existing import at line 20).

Note: ib_cycle.py is not black/isort/mypy-checked by pre-commit (excluded). Match the file's existing 4-space mixed-quote style. `R3 contract` uses a 40-line cap — `_sleep_tick` is ~28 lines, fine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_session_gate.py tests/test_pipeline_monitor.py tests/test_bugfixes.py -q --no-header --no-cov`
Expected: PASS. (Existing `test_finish_cycle_skips_entries_when_market_closed` still uses `CycleMeta(..., market_open=False)` and passes untouched.)

- [ ] **Step 5: Sanity — quick subset of the full suite**

Run: `.venv/bin/python -m pytest tests/test_contract.py tests/test_architecture.py tests/test_trading_policy.py -q --no-header --no-cov`
Expected: PASS (contract R-checks unaffected by ib_cycle edits since ib_cycle is already exempt).

- [ ] **Step 6: Commit**

```bash
.venv/bin/black src/hanoon_prime/monitor/pipeline.py
.venv/bin/isort src/hanoon_prime/monitor/pipeline.py tests/test_session_gate.py tests/test_pipeline_monitor.py
git add -A
git commit -m "feat(system): deep-sleep gate for deactivated sessions"
```

---

### Task 3: Pre-market equity fallback chain

**Files:**
- Create: `src/hanoon_prime/account_equity.py`
- Modify: `src/hanoon_prime/ib_cycle.py:396-417` (`_publish_account_feed`)
- Modify: `src/hanoon_prime/brain/policy/portfolio_risk.py:81-94` (`update_equity` gains `synced` param)
- Modify: `src/hanoon_prime/brain/consolidation.py:117-119` (pass `synced` through)
- Create: `tests/test_account_equity.py`

**Interfaces:**
- Produces:
  - `resolve_account_equity(ib: Any, account: str | None) -> tuple[float | None, bool]` — `(equity, synced)`; chain: live NetLiq → local cash+portfolio → cache → `(None, False)`. Always persists a live value to cache.
  - `save_equity_cache(equity: float, path: Path | None = None) -> None` (atomic tmp+replace into `runtime/equity_cache.json` by default)
  - `load_equity_cache(path: Path | None = None) -> float | None`
  - cache path override via env `HANOON_EQUITY_CACHE`
- `PortfolioRisk.update_equity(equity: float, synced: bool = True) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_account_equity.py`:
```python
"""Equity fallback chain (pre-market sizing)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanoon_prime.account_equity import (  # noqa: E402
    load_equity_cache,
    resolve_account_equity,
    save_equity_cache,
)


def _ib(net: str | None = None, cash: str = "0.0", positions: tuple = ()) -> MagicMock:
    ib = MagicMock()
    items = []
    if net is not None:
        items.append(SimpleNamespace(tag="NetLiquidation", value=net))
    items.append(SimpleNamespace(tag="CashBalance", value=cash))
    ib.accountSummary.return_value = items

    def _porfolio() -> tuple:
        return positions

    ib.portfolio.return_value = positions  # type: ignore[attr-defined]
    return ib


def _pos(sym: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=sym), marketValue=value,
        position=10, unrealizedPNL=1.0,
    )


class TestEquityFallback:
    def test_net_liq_is_used_and_cached(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(tmp_path / "eq.json"))
        ib = _ib(net="100000")
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq == 100000.0
        assert synced is True
        assert load_equity_cache() == 100000.0

    def test_local_cash_plus_portfolio(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(tmp_path / "eq.json"))
        ib = _ib(cash="500", positions=(_pos("AAPL", 100.0),))
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq == 600.0
        assert synced is True

    def test_cache_fallback(self, tmp_path, monkeypatch) -> None:
        p = tmp_path / "eq.json"
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(p))
        save_equity_cache(1234.5, p)
        ib = _ib()  # empty summary, empty portfolio
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq == 1234.5
        assert synced is True

    def test_unknown_stays_unsynced(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(tmp_path / "eq.json"))
        ib = _ib()
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq is None
        assert synced is False

    def test_bad_cache_file_ignored(self, tmp_path, monkeypatch) -> None:
        p = tmp_path / "eq.json"
        p.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv("HANOON_EQUITY_CACHE", str(p))
        ib = _ib()
        eq, synced = resolve_account_equity(ib, "DU123")
        assert eq is None
        assert synced is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_account_equity.py -q --no-header --no-cov`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.account_equity`.

- [ ] **Step 3: Implement `src/hanoon_prime/account_equity.py`**

```python
"""Account equity resolution for pre-market sizing.

IB's accountSummary reports no NetLiquidation before 09:30 ET. Pre-market
entries still need equity to size, so resolve in order: live NetLiq →
local cash + portfolio mark → last-good cache. Unknown stays unknown,
and the sizing veto owns that case (no fabricated equity).

Cache path is overridable via HANOON_EQUITY_CACHE (used by tests).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from ._ib_sync import read_portfolio


def _cache_path() -> Path:
    env = os.getenv("HANOON_EQUITY_CACHE", "")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "runtime" / "equity_cache.json"


def _summary_tags(ib: Any, account: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        items = list(ib.accountSummary(account))
    except Exception:
        items = []
    for item in items:
        try:
            out[str(item.tag)] = float(item.value)
        except (AttributeError, TypeError, ValueError):
            pass
    return out


def save_equity_cache(equity: float, path: Path | None = None) -> None:
    """Persist the last known equity (atomic tmp + replace, best-effort)."""
    p = path or _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"ts": time.time(), "equity": float(equity)}),
            encoding="utf-8",
        )
        tmp.replace(p)
    except Exception:
        pass


def load_equity_cache(path: Path | None = None) -> float | None:
    """Return the last cached equity, or None when missing/corrupt/<=0."""
    p = path or _cache_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        eq = float(data["equity"])
        return eq if math.isfinite(eq) and eq > 0 else None
    except Exception:
        return None


def _live_or_local(ib: Any, account: str | None) -> float | None:
    """Live NetLiq first; otherwise cash + portfolio mark (pre-market)."""
    tags = _summary_tags(ib, account)
    net = tags.get("NetLiquidation", 0.0)
    if math.isfinite(net) and net > 0:
        return net
    try:
        portfolio = read_portfolio(ib)
    except Exception:
        portfolio = {}
    cash = tags.get("CashBalance", 0.0)
    total = cash + sum(float(p.get("value", 0.0) or 0.0) for p in portfolio.values())
    return total if math.isfinite(total) and total > 0 else None


def resolve_account_equity(
    ib: Any, account: str | None
) -> tuple[float | None, bool]:
    """(equity, synced). synced=True for any real value (live/local/cache)."""
    live = _live_or_local(ib, account)
    if live is not None:
        save_equity_cache(live)
        return live, True
    cached = load_equity_cache()
    if cached is not None:
        return cached, True
    return None, False
```

- [ ] **Step 4: Wire into the account feed**

In `src/hanoon_prime/ib_cycle.py` add import near the top (after the existing `from ._ib_sync import read_portfolio` at line 16):
```python
from .account_equity import resolve_account_equity
```
Replace the NetLiquidation block inside `_publish_account_feed` (lines 404-413):
```python
            try:
                equity, synced = resolve_account_equity(self.ib, self.account)
                if equity is not None:
                    feed["equity"] = equity
                    feed["equity_synced"] = synced
                feed["positions"] = read_portfolio(self.ib)
            except Exception as exc:
                log.debug("Account sync skipped: %s", exc)
```

In `src/hanoon_prime/brain/policy/portfolio_risk.py:81-86` change `update_equity`:
```python
    def update_equity(self, equity: float, synced: bool = True) -> None:
        """Update equity; synced=False keeps the account flagged unsynced."""
        if equity <= 0:
            return  # never fabricate equity from a failed read
        self._equity = float(equity)
        if synced:
            self._equity_synced = True
```
(Rest of the method unchanged.)

In `src/hanoon_prime/brain/consolidation.py:117-119` change:
```python
        equity = feed.get("equity")
        if equity is not None:
            self.portfolio_risk.update_equity(
                float(equity), synced=bool(feed.get("equity_synced", True))
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_account_equity.py tests/test_brain_policy_cycle.py tests/test_brain_pipeline.py -q --no-header --no-cov`
Expected: PASS (existing policy tests call `update_equity(x)` with default `synced=True`).

- [ ] **Step 6: Format + commit**

```bash
.venv/bin/black src/hanoon_prime/account_equity.py src/hanoon_prime/brain/policy/portfolio_risk.py src/hanoon_prime/brain/consolidation.py tests/test_account_equity.py
.venv/bin/isort src/hanoon_prime/account_equity.py src/hanoon_prime/brain/policy/portfolio_risk.py src/hanoon_prime/brain/consolidation.py tests/test_account_equity.py
git add -A
git commit -m "feat(equity): fallback chain so pre-market entries can size"
```
(Do NOT run black/isort on ib_cycle.py — it is excluded.)

---

### Task 4: Telemetry `/session` + health fields

**Files:**
- Modify: `src/hanoon_prime/telemetry.py` (`ROUTES_GET` line 21-34, `_health` line 179, add `_session`)
- Modify: `tests/test_telemetry.py` (add session tests)

**Interfaces:**
- Produces: `GET /session` → `{"session", "active", "enabled", "reason", "ts"}` — the source of truth HALIM polls. `/health` gains `"session"` + `"session_active"`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_telemetry.py` add (import `_H` from `hanoon_prime.telemetry` and `Object.__new__`):
```python
    def test_session_endpoint_shape(self) -> None:
        from hanoon_prime.telemetry import _H

        handler = _H.__new__(_H)
        handler.bot = None
        body = handler._session()
        assert body["session"] in {"pre_market", "rth", "post_market", "overnight"}
        assert isinstance(body["active"], bool)
        assert set(body["enabled"]) == {
            "pre_market", "rth", "post_market", "overnight",
        }
        assert isinstance(body["ts"], float)

    def test_health_reports_session(self) -> None:
        from hanoon_prime.telemetry import _H

        handler = _H.__new__(_H)
        bot = MagicMock()
        bot.streamer.ticker_subs = {}
        bot.journal = MagicMock()
        bot.journal.count.return_value = 0
        bot.hippocampus = MagicMock()
        bot.hippocampus.safety_enabled = True
        bot._halted = False
        bot._last_beat = 0.0
        handler.bot = bot
        handler.ib = MagicMock()
        handler.ib.isConnected.return_value = True
        handler.ib.positions.return_value = []
        h = handler._health()
        assert "session" in h
        assert "session_active" in h
        assert isinstance(h["session_active"], bool)
```
(Existing `test_telemetry.py` already imports `MagicMock` — confirm and reuse; otherwise add the import.)

Also add a route-table assertion that `/session` is registered:
```python
    def test_session_route_registered(self) -> None:
        from hanoon_prime.telemetry import ROUTES_GET

        assert ROUTES_GET["/session"] == "_session"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_telemetry.py -q --no-header --no-cov`
Expected: FAIL — `/session` not in ROUTES_GET; `_session`/health fields missing.

- [ ] **Step 3: Implement**

In `src/hanoon_prime/telemetry.py`:
- Add to `ROUTES_GET` (after `"/verdicts"`):
```python
    "/session": "_session",
```
- Add method after `_config` (line ~352):
```python
    def _session(self) -> dict[str, Any]:
        """Current session-gate state (source of truth for HALIM + ops)."""
        from .monitor.sleep_manager import SleepManager

        st = SleepManager().effective_state(TRADING_CONFIG)
        return {
            "session": st.session,
            "active": bool(st.active),
            "enabled": TRADING_CONFIG.to_dict().get("sessions", {}),
            "reason": st.reason,
            "ts": time.time(),
        }
```
- In `_health` (line ~192) add two keys (and compute the gate once at the top of the return-expressions area, after `policy`):
```python
        from .monitor.sleep_manager import SleepManager

        _st = SleepManager().effective_state(TRADING_CONFIG)
```
then add to the returned dict (after `"last_beat"`):
```python
            "session": _st.session,
            "session_active": bool(_st.active),
```
Match telemetry.py's existing style (4-space, double quotes) — it is black-excluded, keep formatting consistent.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_telemetry.py -q --no-header --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(telemetry): GET /session + session fields on /health"
```

---

### Task 5: HALIM session gate (separate process)

**Files:**
- Create: `halim/halim/session_gate.py` (stdlib only)
- Modify: `halim/halim/serve.py` (import, `/health`, `_do_post` 503 gate, start poller in `main`)
- Create: `tests/test_halim_session.py`

**Interfaces:**
- Produces (in `halim/halim/session_gate.py`):
  - `poll_once(url: str | None = None) -> dict` — fetch gate once; ANY error → `{"active": False, "session": "unknown"}` (fail-safe asleep)
  - `snapshot() -> dict`, `asleep() -> bool`
  - `start() -> threading.Thread` (daemon poll loop, `HALIM_SESSION_POLL_SEC` default 20)
  - URL from env `HALIM_SESSION_URL` default `http://127.0.0.1:8080/session`
- `halim/halim/serve.py` exposes `asleep` as `_system_asleep` and `snapshot` as `_session_snapshot`.

- [ ] **Step 1: Write the failing tests**

`tests/test_halim_session.py`:
```python
"""HALIM session gate — stdlib import-only, fail-safe asleep."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "halim") not in sys.path:
    sys.path.insert(0, str(ROOT / "halim"))

from halim import session_gate  # noqa: E402


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _patch_urlopen(monkeypatch, result: MagicMock) -> None:
    monkeypatch.setattr(session_gate.urllib.request, "urlopen", result)


class TestSessionPoller:
    def test_ok_active(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, MagicMock(return_value=_Resp(b'{"active": true, "session": "rth"}')))
        session_gate.poll_once()
        assert session_gate.asleep() is False
        assert session_gate.snapshot() == {"active": True, "session": "rth"}

    def test_ok_deactivated(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, MagicMock(return_value=_Resp(b'{"active": false, "session": "overnight"}')))
        session_gate.poll_once()
        assert session_gate.asleep() is True

    def test_timeout_falls_back_asleep(self, monkeypatch) -> None:
        urlopen = MagicMock(side_effect=TimeoutError("gate down"))
        _patch_urlopen(monkeypatch, urlopen)
        session_gate.poll_once()
        assert session_gate.asleep() is True
        assert session_gate.snapshot()["session"] == "unknown"

    def test_garbage_falls_back_asleep(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, MagicMock(return_value=_Resp(b"not json{")))
        session_gate.poll_once()
        assert session_gate.asleep() is True

    def test_url_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("HALIM_SESSION_URL", "http://127.0.0.1:9999/gate")
        urlopen = MagicMock(return_value=_Resp(b'{"active": true, "session": "rth"}'))
        _patch_urlopen(monkeypatch, urlopen)
        session_gate.poll_once()
        last_url = urlopen.call_args.args[0]
        assert last_url == "http://127.0.0.1:9999/gate"

    def test_wake_after_sleep(self, monkeypatch) -> None:
        session_gate.poll_once()
        urlopen = MagicMock(return_value=_Resp(b'{"active": true, "session": "rth"}'))
        _patch_urlopen(monkeypatch, urlopen)
        session_gate.poll_once()
        assert session_gate.asleep() is False
```
(The kernel of the gate — the serve.py handler is a thin call into `asleep()`, validated at runtime in Task 6.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_halim_session.py -q --no-header --no-cov`
Expected: FAIL — `ModuleNotFoundError: halim.session_gate`.

- [ ] **Step 3: Implement `halim/halim/session_gate.py`**

```python
"""Halim's half of the whole-system session gate.

The bot owns the gate (telemetry GET /session). Halim polls it and falls
back to ASLEEP on any error, so a deactivated session silences inference.
Stdlib only — keep this importable in minimal environments (it is what
the repo test suite imports, not the full serve module).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any, Dict

DEFAULT_URL = "http://127.0.0.1:8080/session"

_lock = threading.Lock()
_state: Dict[str, Any] = {"active": True, "session": "unknown"}


def session_url() -> str:
    """Gate URL — HALIM_SESSION_URL or the default telemetry endpoint."""
    return os.getenv("HALIM_SESSION_URL", DEFAULT_URL)


def poll_interval() -> float:
    """Seconds between polls — HALIM_SESSION_POLL_SEC (default 20)."""
    return float(os.getenv("HALIM_SESSION_POLL_SEC", "20.0"))


def snapshot() -> Dict[str, Any]:
    """Thread-safe copy of the last observed gate state."""
    with _lock:
        return dict(_state)


def asleep() -> bool:
    """True when the whole system is asleep (or unknown → fail-safe)."""
    return not bool(snapshot().get("active", True))


def poll_once(url: str | None = None) -> Dict[str, Any]:
    """Fetch the gate once; any error transitions to asleep (fail-safe)."""
    global _state
    try:
        with urllib.request.urlopen(url or session_url(), timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        snap: Dict[str, Any] = {
            "active": bool(data.get("active")),
            "session": str(data.get("session", "unknown")),
        }
    except Exception:
        snap = {"active": False, "session": "unknown"}
    with _lock:
        _state = snap
    return dict(snap)


def _loop() -> None:
    while True:
        poll_once()
        time.sleep(poll_interval())


def start() -> threading.Thread:
    """Start the daemon poller; first poll fires immediately."""
    thread = threading.Thread(target=_loop, daemon=True, name="halim-session-poller")
    thread.start()
    return thread
```

- [ ] **Step 4: Wire into `halim/halim/serve.py`**

- Add imports near the top (after line 26's `from halim.protocol import ...`):
```python
from halim.session_gate import asleep as _system_asleep
from halim.session_gate import snapshot as _session_snapshot
from halim.session_gate import start as _start_session_poller
```
- Add a module constant (near the priority levels):
```python
# Inference routes silenced while the whole system sleeps (session gate).
_INFERENCE_ROUTES = frozenset(
    {
        "/v1/complete",
        "/v1/complete-thinking",
        "/v1/record",
        "/v1/evolve",
        "/v1/chat",
        "/v1/generate",
    }
)
```
- `do_GET` `/health` (line 511-512) → add the sleep fields:
```python
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "model": MODEL_NAME,
                    "protocol": PROTOCOL_VERSION,
                    "asleep": _system_asleep(),
                    "session": _session_snapshot().get("session", "unknown"),
                },
            )
```
- `_do_post` — right after the body parse / invalid_json branch (after line 546), insert:
```python
        if _system_asleep() and self.path in _INFERENCE_ROUTES:
            self._json(503, {"ok": False, "reason": "system_asleep"})
            return
```
- `main()` — after `_get_worker().start()` (line 654) add:
```python
    _start_session_poller()
    print("   Session poller started (whole-system gate)", flush=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_halim_session.py -q --no-header --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(halim): session-gate poller — whole-system sleep"
```

---

### Task 6: Docs, spec parity, full validation, deploy

**Files:**
- Modify: `docs/superpowers/specs/2026-09-09-master-session-gate-design.md` (equity §4 wording parity)
- Modify: `docs/ARCHITECTURE.md` (session-gate paragraph)

- [ ] **Step 1: Spec parity edit**

In the spec §4, replace the sentence `equity_synced=True only when a real source produced the value this cycle (chain 1 or 2).` with:
> `equity_synced=True` for any real value (chains 1–3: live NetLiq, local cash+portfolio mark, or last-good cache). Only chain 4 (truly unknown) stays unsynced, so the sizing veto still owns the no-data case.

- [ ] **Step 2: ARCHITECTURE.md section**

Append a short subsection to the session/ops area of `docs/ARCHITECTURE.md`:
```markdown
### Whole-system session gate

`SleepManager.effective_state()` is the single gate (clock window AND
TradingConfig enabled-set; ids: pre_market/rth/post_market/overnight).
Enabled sessions (pre_market + rth) run the full brain-first pipeline,
including pre-market ENTER execution sized from the equity fallback chain
(runtime/equity_cache.json). Disabled sessions put the WHOLE system to
sleep: the bot runs only a keepalive (gateway supervision, on-demand
flatten, telemetry heartbeat), and HALIM polls `GET /session` and 503s
inference while asleep. Telemetry stays up in both states.
```

- [ ] **Step 3: Full-suite validation**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (previously 573 pass / 3 skip; expect ~590+ pass, same skips).

Run: `.venv/bin/mypy --strict src/hanoon_prime` (whole package under the hook's config)
Expected: only the known baseline `halim_recommendations.py:97` errors remain — no new errors.

Run: `.venv/bin/pre-commit run --all-files`
Expected: all hooks pass. (Note: the mypy hook excludes `ib_cycle.py`; the `check_file_length` hook exempts telemetry/orchestrator/ib_cycle; all new/edited files pass R3/R3b.)

- [ ] **Step 4: Commit + push both remotes**

```bash
git add -A
git commit -m "docs: session gate architecture + spec parity for equity fallback"
git push origin main
git push origin-sajib main
```

- [ ] **Step 5: Live deploy + runtime verification (paper)**

1. Restart the bot so the new loop/equity/telemetry code is live:
   - Kill the current `python -m hanoon_prime.cli` process (find via `ps aux | rg "hanoon_prime.cli"`).
   - Relaunch from the repo: `python3 -m hanoon_prime.cli` (same as the running command).
2. Restart HALIM serve so it picks up `session_gate.py`:
   - Kill the current `halim/halim/serve.py` process (PID from `ps aux | rg "halim/halim/serve.py"`).
   - Relaunch exactly as it runs now: `python3 halim/halim/serve.py --host 127.0.0.1 --port 8765`.
3. Verify while in pre-market (04:00–09:30 ET == 14:00–19:30 +06):
   - `curl -s http://127.0.0.1:8080/session` → `"session": "pre_market", "active": true`.
   - `/verdicts` shows ENTER actions and `/journal` shows sized brackets in pre-market: `rg '"event": "verdict"' runtime/journal_live.jsonl | tail` plus `rg "place_bracket|ENTER|SKIP" logs/hanoon_prime.log`.
   - `curl -s http://127.0.0.1:8080/risk` → `equity` non-zero via fallback chain.
   - `curl -s http://127.0.0.1:8765/health` → `"asleep": false, "session": "pre_market"`.
   - Toggle sleep via webapp/`/config` POST `{"sessions": {"pre_market": false}}` → within ~40s `curl http://127.0.0.1:8765/health` reports `"asleep": true` and `POST /v1/complete` returns 503 `system_asleep`; re-enable and confirm wake.
4. If any check fails, stop, fix via the normal TDD loop, and re-run Step 3 before pushing.