"""brain.pillar_awareness — the win/loss pillar Juli must keep upright.

Edge-vs-break-even geometry: the pillar stands at 90 degrees (success) when
the realized win rate clears the break-even rate implied by the realized
payoff ratio ``avg_win / avg_loss``. Any negative edge is a lean, and a lean
is failure regardless of which side (long or short) produced it — the same
objective as the RL cart-pole task, where staying upright is rewarded and any
tilt past the threshold is terminal.

The brain reads this from :meth:`RealizedStats.win_loss_record`; the Inside
Man and webapp read the identical shape from the persisted ``brain_state``.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..immune import PILLAR_EDGE_FALL, PILLAR_MIN_TRADES

STATE_UPRIGHT = "upright"
STATE_TIPPING = "tipping"
STATE_FALLEN = "fallen"
STATE_WARMING = "warming"

__all__ = [
    "STATE_FALLEN",
    "STATE_TIPPING",
    "STATE_UPRIGHT",
    "STATE_WARMING",
    "compute_pillar_awareness",
    "pillar_fields",
    "resolve_pillar",
    "win_loss_record",
]


def _tally(
    samples: Iterable[tuple[int, float, int]],
) -> tuple[list[float], list[float]]:
    """Split realized ``(won, abs_pnl, direction)`` rows into win/loss values."""
    wins: list[float] = []
    losses: list[float] = []
    for row in samples or []:
        try:
            won, pnl, _direction = row
            value = abs(float(pnl))
        except (TypeError, ValueError):
            continue
        (wins if won else losses).append(value)
    return wins, losses


def win_loss_record(samples: Iterable[tuple[int, float, int]]) -> dict[str, Any]:
    """Fold realized rows into win/loss stats (cold-safe on empty/malformed).

    The break-even win rate is ``1/(1 + avg_win/avg_loss)``; with losses but
    zero wins it is 1.0 (no win rate clears it), and with no losses it is 0.0.
    """
    wins, losses = _tally(samples)
    n = len(wins) + len(losses)
    gross_win = sum(wins)
    gross_loss = sum(losses)
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    r_r = avg_win / avg_loss if avg_loss > 0 else 0.0
    break_even = 1.0 / (1.0 + r_r) if avg_loss > 0 else 0.0
    win_rate = len(wins) / n if n else 0.0
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "record": f"{len(wins)}W-{len(losses)}L",
        "win_rate": round(win_rate, 4),
        "break_even_wr": round(break_even, 4),
        "edge": round(win_rate - break_even, 4),
        "r_r": round(r_r, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "gross_win": round(gross_win, 6),
        "gross_loss": round(gross_loss, 6),
        "net": round(gross_win - gross_loss, 6),
    }


def _defaults() -> dict[str, Any]:
    """Warming/empty pillar awareness shape."""
    return {
        "state": STATE_WARMING,
        "upright": True,
        "tilt": 0.0,
        "edge": 0.0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "record": "0W-0L",
        "win_rate": 0.0,
        "break_even_wr": 0.0,
        "r_r": 0.0,
        "net": 0.0,
    }


def _merge(base: dict[str, Any], record: dict[str, Any]) -> None:
    """Copy the record's numeric/record fields into ``base``."""
    for key in (
        "trades",
        "wins",
        "losses",
        "win_rate",
        "break_even_wr",
        "edge",
        "r_r",
        "net",
    ):
        base[key] = record.get(key, base[key])
    base["record"] = record.get("record", base["record"])


def _apply_state(base: dict[str, Any]) -> None:
    """Classify edge -> state/tilt on an already-merged record."""
    edge = float(base["edge"] or 0.0)
    if edge >= 0.0:
        state = STATE_UPRIGHT
    elif edge > PILLAR_EDGE_FALL:
        state = STATE_TIPPING
    else:
        state = STATE_FALLEN
    span = abs(PILLAR_EDGE_FALL) or 1.0
    base["state"] = state
    base["upright"] = state == STATE_UPRIGHT
    base["tilt"] = round(min(1.0, max(0.0, -edge / span)), 4)


def compute_pillar_awareness(record: dict[str, Any] | None) -> dict[str, Any]:
    """Map a win/loss record to the pillar state, tilt, and edge.

    ``tilt`` is 0.0 at the upright (break-even-or-better) state and grows to
    1.0 as the edge falls to ``PILLAR_EDGE_FALL``. ``state`` is one of
    ``upright`` / ``tipping`` / ``fallen`` / ``warming``.
    """
    base = _defaults()
    if not isinstance(record, dict):
        return base
    _merge(base, record)
    trades = int(base["trades"] or 0)
    if trades <= 0 or trades < PILLAR_MIN_TRADES:
        return base
    _apply_state(base)
    return base


def resolve_pillar(
    pillar: dict[str, Any] | None, snapshot: dict[str, Any] | None
) -> dict[str, Any]:
    """Use the live pillar awareness, else derive it from the realized store."""
    if isinstance(pillar, dict) and pillar:
        return compute_pillar_awareness(pillar)
    samples = snapshot.get("rr_samples", []) if isinstance(snapshot, dict) else []
    return compute_pillar_awareness(win_loss_record(samples))


def pillar_fields(
    pillar: dict[str, Any] | None, snapshot: dict[str, Any] | None
) -> dict[str, Any]:
    """Flatten pillar awareness into HALIM-evidence key/value pairs."""
    p = resolve_pillar(pillar, snapshot)
    return {
        "pillar_state": p["state"],
        "pillar_upright": p["upright"],
        "pillar_edge": p["edge"],
        "pillar_tilt": p["tilt"],
        "win_loss_record": p["record"],
        "win_loss_net": p["net"],
    }
