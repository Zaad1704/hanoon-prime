"""hanoon_prime.labels — triple-barrier labels + sample uniqueness (AFML ch. 3-4).

Each entry labelled by first barrier: profit target (+1), stop loss (-1),
or vertical time expiry (0). Overlapping spans down-weighted by concurrency.
R1: emits labels only — live exit path untouched (TIMEOUT_BARS=999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .immune import (
    ATR_STOP_MULT,
    ATR_TARGET_MULT,
    LABEL_MIN_WEIGHT,
    LABEL_VERTICAL_BARS,
)

__all__ = [
    "BarrierLabel",
    "LabelSet",
    "barrier_bracket",
    "triple_barrier_label",
    "triple_barrier_labels",
    "label_concurrency",
    "uniqueness_weights",
    "build_label_set",
]


@dataclass(frozen=True)
class BarrierLabel:
    """One labelled entry event: label is +1/-1/0, price is the fill."""

    entry_idx: int
    exit_idx: int
    direction: int
    label: int
    price: float
    reason: str


@dataclass(frozen=True)
class LabelSet:
    """A batch of labels plus their concurrency-adjusted sample weights."""

    labels: tuple[BarrierLabel, ...]
    entry_t0: np.ndarray
    entry_t1: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    concurrency: np.ndarray


def barrier_bracket(direction: int, entry: float, atr: float) -> tuple[float, float]:
    """Return ``(stop, target)`` for ``direction`` — same geometry as hands."""
    if direction > 0:
        return entry - atr * ATR_STOP_MULT, entry + atr * ATR_TARGET_MULT
    return entry + atr * ATR_STOP_MULT, entry - atr * ATR_TARGET_MULT


def _touch(
    direction: int,
    stop: float,
    target: float,
    high_i: float,
    low_i: float,
) -> tuple[int, float] | None:
    """Which barrier this bar touches first, or None. Stop checked before target."""
    if direction > 0:
        if low_i <= stop:
            return -1, stop
        if high_i >= target:
            return 1, target
        return None
    if high_i >= stop:
        return -1, stop
    if low_i <= target:
        return 1, target
    return None


def _first_touch(
    direction: int,
    entry_idx: int,
    entry: float,
    atr: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    vertical_bars: int,
) -> BarrierLabel:
    """Walk forward from ``entry_idx + 1`` to the first barrier touch.

    Entry is only *known* at the close of ``entry_idx``, so scanning starts on
    the next bar — this is the causality guarantee that keeps the label free
    of same-bar look-ahead.
    """
    stop, target = barrier_bracket(direction, entry, atr)
    last = close.size - 1
    end = min(entry_idx + vertical_bars, last)
    for i in range(entry_idx + 1, end + 1):
        hit = _touch(direction, stop, target, float(high[i]), float(low[i]))
        if hit is not None:
            label, price = hit
            return BarrierLabel(
                entry_idx, i, direction, label, price, "target" if label > 0 else "stop"
            )
    return BarrierLabel(entry_idx, end, direction, 0, float(close[end]), "vertical")


def triple_barrier_label(
    direction: int,
    entry_idx: int,
    entry: float,
    atr: float,
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    vertical_bars: int = LABEL_VERTICAL_BARS,
) -> BarrierLabel:
    """Label a single entry event against the three barriers."""
    return _first_touch(
        direction,
        entry_idx,
        entry,
        atr,
        np.asarray(high, dtype=float),
        np.asarray(low, dtype=float),
        np.asarray(close, dtype=float),
        int(vertical_bars),
    )


def triple_barrier_labels(
    direction: Iterable[int],
    entry_idx: Iterable[int],
    entry: Iterable[float],
    atr: Iterable[float],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    vertical_bars: int = LABEL_VERTICAL_BARS,
) -> tuple[BarrierLabel, ...]:
    """Label many entry events against a shared price path.

    Inputs are aligned positionally; the price path is materialised once.
    """
    hi = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    cl = np.asarray(close, dtype=float)
    return tuple(
        _first_touch(int(d), int(i), float(e), float(a), hi, lo, cl, int(vertical_bars))
        for d, i, e, a in zip(direction, entry_idx, entry, atr)
    )


def label_concurrency(t0: Sequence[int], t1: Sequence[int], n_bars: int) -> np.ndarray:
    """Bar-level count of how many label spans are open at each bar."""
    counts = np.zeros(int(n_bars), dtype=float)
    for a, b in zip(t0, t1):
        counts[int(a) : int(b) + 1] += 1.0
    return counts


def uniqueness_weights(t0: Sequence[int], t1: Sequence[int], n_bars: int) -> np.ndarray:
    """Average-uniqueness weight per sample, floored at LABEL_MIN_WEIGHT."""
    counts = label_concurrency(t0, t1, int(n_bars))
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = np.empty(len(t0), dtype=float)
    for k, (a, b) in enumerate(zip(t0, t1)):
        weights[k] = max(float(np.mean(inv[int(a) : int(b) + 1])), LABEL_MIN_WEIGHT)
    return weights


def build_label_set(
    labels: Sequence[BarrierLabel],
    n_bars: int,
    vertical_bars: int = LABEL_VERTICAL_BARS,
) -> LabelSet:
    """Package labels with their uniqueness weights and concurrency profile."""
    t0 = np.asarray([lb.entry_idx for lb in labels], dtype=int)
    t1 = np.asarray([lb.exit_idx for lb in labels], dtype=int)
    y = np.asarray([lb.label for lb in labels], dtype=int)
    return LabelSet(
        labels=tuple(labels),
        entry_t0=t0,
        entry_t1=t1,
        y=y,
        weights=uniqueness_weights(t0.tolist(), t1.tolist(), n_bars),
        concurrency=label_concurrency(t0.tolist(), t1.tolist(), n_bars),
    )
