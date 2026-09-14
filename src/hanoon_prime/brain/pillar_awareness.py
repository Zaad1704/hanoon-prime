"""brain.pillar_awareness — the win/loss pillar Juli must keep upright.

Realized edge-vs-break-even state (upright/tipping/fallen/warming) with
dynamic allostatic setpoint threshold, RPE mood, and HALIM evidence keys.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..immune import PILLAR_EDGE_FALL, PILLAR_MIN_TRADES

STATE_UPRIGHT = "upright"
STATE_TIPPING = "tipping"
STATE_FALLEN = "fallen"
STATE_WARMING = "warming"


def _tally(
    samples: Iterable[tuple[int, float, int]],
) -> tuple[list[float], list[float]]:
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


def _apply_state(base: dict[str, Any], setpoint_edge: float | None = None) -> None:
    """Classify edge -> state/tilt; fallen line matches the learned norm.

    When a regime has learned a negative setpoint (a stress norm), the
    fallen line tightens to the setpoint (never looser than
    ``PILLAR_EDGE_FALL``); a positive norm keeps the structural floor.
    """
    edge = float(base["edge"] or 0.0)
    norm = PILLAR_EDGE_FALL
    if setpoint_edge is not None and float(setpoint_edge) < 0.0:
        norm = max(PILLAR_EDGE_FALL, float(setpoint_edge))
    if edge >= 0.0:
        state = STATE_UPRIGHT
    elif edge > norm:
        state = STATE_TIPPING
    else:
        state = STATE_FALLEN
    base["state"] = state
    base["upright"] = state == STATE_UPRIGHT
    base["tilt"] = round(min(1.0, max(0.0, -edge / (abs(norm) or 1.0))), 4)


def compute_pillar_awareness(
    record: dict[str, Any] | None,
    rpe: dict[str, Any] | None = None,
    setpoint_edge: float | None = None,
) -> dict[str, Any]:
    """Map a win/loss record to the pillar state, tilt, and edge.

    ``tilt`` is 0.0 at the upright state and grows to 1.0 as the edge
    falls to the fallen threshold (structural or allostatic setpoint).
    ``state`` is one of ``upright`` / ``tipping`` / ``fallen`` / ``warming``.
    ``rpe`` (dopamine channels) and ``setpoint_edge`` (allostatic norm)
    are surfaced alongside the geometry so the mood + norm are visible.
    """
    base = _defaults()
    if isinstance(record, dict):
        _merge(base, record)
        trades = int(base["trades"] or 0)
        if trades > 0 and trades >= PILLAR_MIN_TRADES:
            _apply_state(base, setpoint_edge)
    return _decorate(base, rpe, setpoint_edge)


def _decorate(
    base: dict[str, Any],
    rpe: dict[str, Any] | None,
    setpoint_edge: float | None = None,
) -> dict[str, Any]:
    """Attach the dopamine channels and allostatic setpoint to the shape."""
    if isinstance(rpe, dict):
        base["rpe_phasic"] = _num(rpe, "phasic")
        base["rpe_tonic"] = _num(rpe, "tonic")
        meta = rpe.get("meta") or rpe.get("v_meta")
        base["rpe_meta"] = dict(meta) if isinstance(meta, dict) else {}
    edge = float(base.get("edge", 0.0) or 0.0)
    setpoint = float(setpoint_edge or 0.0)
    base["setpoint"] = round(setpoint, 4)
    base["setpoint_deviation"] = round(edge - setpoint, 4)
    base["below_setpoint"] = bool(setpoint_edge is not None and edge < setpoint)
    return base


def _num(rpe: dict[str, Any], key: str) -> float:
    raw = rpe.get(key)
    if not isinstance(raw, (int, float)):
        raw = 0.0
    return round(float(raw), 4)


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
        "pillar_setpoint": p.get("setpoint", 0.0),
        "pillar_deviation": p.get("setpoint_deviation", 0.0),
        "win_loss_record": p["record"],
        "win_loss_net": p["net"],
    }
