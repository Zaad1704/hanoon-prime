"""live-fidelity joint checks — marks truth, consent, and closed bars."""
import datetime

from hanoon_prime.inspection import live
from hanoon_prime.inspection.checks import FAIL, OK, UNVERIFIABLE, WARN
from hanoon_prime.inspection.ctx import InspectionContext


def _now_t() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _health(ctx: InspectionContext, *, active: bool, count: int) -> InspectionContext:
    ctx.memo["health"] = {
        "status": "ok",
        "connected": True,
        "session_active": active,
        "position_count": count,
        "positions": [],
    }
    return ctx


def test_positions_marked_live_unreachable(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"_unreachable": "offline"}
    assert live.positions_marked_live(ctx).status == UNVERIFIABLE


def test_positions_marked_live_flat_when_no_holdings(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=0)
    ctx.memo["positions"] = {"positions": [], "total_pnl": 0.0, "count": 0}
    ctx.memo["account"] = {"account_summary": {}, "positions_open": 0}
    assert live.positions_marked_live(ctx).status == OK


def test_positions_marked_live_fails_cross_source_zero(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=2)
    ctx.memo["positions"] = {
        "positions": [
            {
                "ticker": "TQQQ",
                "entry_price": 70.0,
                "market_price": 70.0,
                "unrealized_pnl": 0.0,
            },
            {
                "ticker": "SOXL",
                "entry_price": 30.0,
                "market_price": 30.0,
                "unrealized_pnl": 0.0,
            },
        ],
        "total_pnl": 0.0,
        "count": 2,
    }
    ctx.memo["account"] = {
        "account_summary": {"UnrealizedPnL": "-128.11"},
        "positions_open": 2,
    }
    result = live.positions_marked_live(ctx)
    assert result.status == FAIL
    assert "0.00 vs IB -128.11" in result.detail


def test_positions_marked_live_warns_stale_marks(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=2)
    ctx.memo["positions"] = {
        "positions": [
            {"ticker": "A", "entry_price": 10.0, "market_price": 10.0},
            {"ticker": "B", "entry_price": 20.0, "market_price": 21.5},
        ],
        "total_pnl": 1.5,
        "count": 2,
    }
    ctx.memo["account"] = {"account_summary": {"unrealized_pnl": "42.00"}}
    assert live.positions_marked_live(ctx).status == WARN


def test_positions_marked_live_fails_systemic_stale(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=2)
    ctx.memo["positions"] = {
        "positions": [
            {"ticker": "A", "entry_price": 10.0, "market_price": 10.0},
            {"ticker": "B", "entry_price": 20.0, "market_price": 20.0},
        ],
        "total_pnl": 0.0,
        "count": 2,
    }
    ctx.memo["account"] = {"account_summary": {"unrealized_pnl": "0.00"}}
    assert live.positions_marked_live(ctx).status == FAIL


def test_positions_marked_live_ok_when_marked(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=1)
    ctx.memo["positions"] = {
        "positions": [{"ticker": "TQQQ", "entry_price": 70.0, "market_price": 71.13}],
        "total_pnl": 113.0,
        "count": 1,
    }
    ctx.memo["account"] = {"account_summary": {"UnrealizedPnL": "113.00"}}
    assert live.positions_marked_live(ctx).status == OK


def test_bars_advance_idle_when_inactive(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=False, count=0)
    assert live.bars_advance_when_active(ctx).status == OK


def test_bars_advance_fails_stuck_zero(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=14)
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    lines = "\n".join(
        f"{_now_t()}.000 INFO ib_cycle CYCLE bars=0 open=14 d=0 x=0" for _ in range(10)
    )
    log.write_text(lines + "\n")
    fresh = _health(InspectionContext(base_dir=tmp_path), active=True, count=14)
    result = live.bars_advance_when_active(fresh)
    assert result.status == FAIL
    assert "0/10" in result.detail


def test_bars_advance_ok_when_closing(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=14)
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            [
                f"{_now_t()}.000 INFO ib_cycle CYCLE bars=0 open=14 d=0 x=0",
                f"{_now_t()}.000 INFO ib_cycle CYCLE bars=3 open=14 d=0 x=0",
                f"{_now_t()}.000 INFO ib_cycle CYCLE bars=2 open=14 d=0 x=0",
            ]
        )
        + "\n"
    )
    assert live.bars_advance_when_active(ctx).status == OK


def test_bars_advance_warns_intermittent(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=14)
    # 2/50 = 4% — above the 1% FAIL floor but below the 5% WARN ceiling
    # for a 1s-cycle / 1-min-bar configuration.
    lines = [
        (
            f"{_now_t()}.000 INFO ib_cycle CYCLE bars={3 if i % 25 == 0 else 0} "
            f"open=14 d=0 x=0"
        )
        for i in range(50)
    ]
    ctx.log_path.parent.mkdir(parents=True)
    ctx.log_path.write_text("\n".join(lines) + "\n")
    assert live.bars_advance_when_active(ctx).status == WARN


def test_bars_advance_no_cycle_fails_when_active(tmp_path) -> None:
    ctx = _health(InspectionContext(base_dir=tmp_path), active=True, count=0)
    ctx.log_path.parent.mkdir(parents=True)
    ctx.log_path.write_text(f"{_now_t()}.000 INFO ib_cycle HEARTBEAT open=0\n")
    assert live.bars_advance_when_active(ctx).status == FAIL
