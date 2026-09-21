"""tests/test_labels.py — unit tests for triple-barrier labels + uniqueness.

Guarantees under test:
  - barrier geometry mirrors hands._make_position for both directions
  - the first-touched barrier decides the label (+1 / -1 / 0)
  - a vertical time-out yields label 0 at the horizon bar
  - labels are causal: no bar at or before entry is scanned, and later bars
    cannot rewrite an already-resolved label
  - concurrency counts overlapping spans; uniqueness down-weights them
  - build_label_set aligns y, weights and concurrency
"""

from __future__ import annotations

import numpy as np
import pytest

from hanoon_prime.hands import _make_position
from hanoon_prime.immune import LABEL_MIN_WEIGHT, LABEL_VERTICAL_BARS
from hanoon_prime.labels import (
    BarrierLabel,
    barrier_bracket,
    build_label_set,
    label_concurrency,
    triple_barrier_label,
    triple_barrier_labels,
    uniqueness_weights,
)


def _flat(n: int, price: float = 100.0):
    hi = np.full(n, price, dtype=float)
    lo = np.full(n, price, dtype=float)
    cl = np.full(n, price, dtype=float)
    return hi, lo, cl


class TestBracketGeometry:
    @pytest.mark.parametrize("direction", [1, -1])
    def test_matches_hands(self, direction):
        assert barrier_bracket(direction, 100.0, 2.0) == _make_position(
            direction, 100.0, 2.0
        )

    def test_long_stop_below_target_above(self):
        stop, target = barrier_bracket(1, 100.0, 2.0)
        assert stop < 100.0 < target

    def test_short_stop_above_target_below(self):
        stop, target = barrier_bracket(-1, 100.0, 2.0)
        assert target < 100.0 < stop


class TestFirstTouch:
    def test_long_target_hit(self):
        hi, lo, cl = _flat(10)
        hi[3] = 130.0  # target is 100 + 6*2 = 112
        label = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == 1
        assert label.exit_idx == 3
        assert label.reason == "target"
        assert label.price == pytest.approx(112.0)

    def test_long_stop_hit(self):
        hi, lo, cl = _flat(10)
        lo[2] = 90.0  # stop is 100 - 2*2 = 96
        label = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == -1
        assert label.exit_idx == 2
        assert label.reason == "stop"
        assert label.price == pytest.approx(96.0)

    def test_short_target_hit(self):
        hi, lo, cl = _flat(10)
        lo[4] = 80.0  # short target is 100 - 6*2 = 88
        label = triple_barrier_label(-1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == 1
        assert label.exit_idx == 4
        assert label.price == pytest.approx(88.0)

    def test_short_stop_hit(self):
        hi, lo, cl = _flat(10)
        hi[1] = 110.0  # short stop is 100 + 2*2 = 104
        label = triple_barrier_label(-1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == -1
        assert label.exit_idx == 1
        assert label.price == pytest.approx(104.0)

    def test_stop_wins_when_bar_touches_both(self):
        """Pessimistic tie-break: stop before target within one bar."""
        hi, lo, cl = _flat(10)
        hi[2] = 130.0
        lo[2] = 80.0
        label = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == -1
        assert label.exit_idx == 2

    def test_vertical_barrier_is_zero(self):
        hi, lo, cl = _flat(100, 100.0)
        cl[LABEL_VERTICAL_BARS] = 101.0
        label = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == 0
        assert label.exit_idx == LABEL_VERTICAL_BARS
        assert label.reason == "vertical"
        assert label.price == pytest.approx(101.0)


class TestCausality:
    def test_entry_bar_is_not_scanned(self):
        """A spike on the entry bar itself must be ignored."""
        hi, lo, cl = _flat(10)
        hi[0] = 999.0
        lo[0] = 1.0
        label = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == 0

    def test_resolved_label_immune_to_later_bars(self):
        hi, lo, cl = _flat(20)
        lo[2] = 90.0
        base = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        hi[15] = 500.0  # huge move long after resolution
        after = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert base == after

    def test_truncated_path_closes_vertically(self):
        hi, lo, cl = _flat(3)
        label = triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert label.label == 0
        assert label.exit_idx == 2


class TestBatch:
    def test_aligns_with_single_calls(self):
        n = 60
        hi, lo, cl = _flat(n)
        lo[5] = 90.0
        hi[20] = 130.0
        batch = triple_barrier_labels(
            [1, 1], [0, 10], [100.0, 100.0], [2.0, 2.0], hi, lo, cl
        )
        assert batch[0] == triple_barrier_label(1, 0, 100.0, 2.0, hi, lo, cl)
        assert batch[1] == triple_barrier_label(1, 10, 100.0, 2.0, hi, lo, cl)

    def test_returns_barrier_labels(self):
        hi, lo, cl = _flat(10)
        batch = triple_barrier_labels([1], [0], [100.0], [2.0], hi, lo, cl)
        assert isinstance(batch[0], BarrierLabel)


class TestUniqueness:
    def test_non_overlapping_spans_have_unit_weight(self):
        w = uniqueness_weights([0, 5, 10], [2, 7, 12], n_bars=20)
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0])

    def test_concurrency_counts_overlap(self):
        counts = label_concurrency([0, 1], [3, 2], n_bars=6)
        np.testing.assert_array_equal(counts, [1, 2, 2, 1, 0, 0])

    def test_overlap_damps_weight(self):
        w = uniqueness_weights([0, 0], [4, 4], n_bars=10)
        assert np.all(w < 1.0)
        assert np.all(w >= LABEL_MIN_WEIGHT)

    def test_weight_is_mean_inverse_concurrency(self):
        t0, t1 = [0], [2]
        counts = label_concurrency(t0, t1, 5)
        expected = float(np.mean(1.0 / counts[0:3]))
        np.testing.assert_allclose(uniqueness_weights(t0, t1, 5), [expected])


class TestBuildLabelSet:
    def test_aligns_outputs(self):
        n = 60
        hi, lo, cl = _flat(n)
        lo[5] = 90.0
        cl[40] = 101.0
        labels = triple_barrier_labels(
            [1, 1], [0, 20], [100.0, 100.0], [2.0, 2.0], hi, lo, cl
        )
        ls = build_label_set(labels, n_bars=n)
        assert ls.y.shape == (2,)
        assert ls.weights.shape == (2,)
        assert ls.concurrency.shape == (n,)
        np.testing.assert_array_equal(ls.entry_t0, [0, 20])
        np.testing.assert_array_equal(ls.y, [lb.label for lb in labels])

    def test_shapes_hold_for_empty_input(self):
        ls = build_label_set((), n_bars=10)
        assert ls.y.shape == (0,)
        assert ls.weights.shape == (0,)
        assert ls.concurrency.shape == (10,)
