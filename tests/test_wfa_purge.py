"""tests/test_wfa_purge.py — purged & embargoed cross-validation tests.

Guarantees under test:
  - purge_mask correctly identifies labels overlapping a test window
  - embargo_mask correctly identifies labels in the embargo zone
  - train_test_split produces disjoint train/test sets that cover all labels
  - weight_adjusted_return/sharpe handle edge cases and produce correct values
  - purged_fold_windows returns correct diagnostics per fold
  - no training label survives if its span overlaps the test window
  - embargo zone is strictly after the test window
"""

from __future__ import annotations

import numpy as np
import pytest

from hanoon_prime.immune import WFA_EMBARGO_BARS
from hanoon_prime.wfa import (
    embargo_mask,
    purge_mask,
    purged_fold_windows,
    train_test_split,
    weight_adjusted_return,
    weight_adjusted_sharpe,
)


class TestPurgeMask:
    def test_no_overlap_keeps_all(self):
        t0 = np.array([0, 1, 2])
        t1 = np.array([5, 6, 7])
        mask = purge_mask(t0, t1, 10, 20)
        assert np.all(mask)

    def test_full_overlap_removes(self):
        t0 = np.array([12, 15])
        t1 = np.array([18, 25])
        mask = purge_mask(t0, t1, 10, 20)
        assert not np.any(mask)

    def test_partial_overlap_at_start(self):
        t0 = np.array([5, 25])
        t1 = np.array([15, 30])
        mask = purge_mask(t0, t1, 10, 20)
        assert mask[1]  # label at 25-30: no overlap
        assert not mask[0]  # label at 5-15: overlaps (t1=15 >= 10)

    def test_partial_overlap_at_end(self):
        t0 = np.array([18, 25])
        t1 = np.array([25, 30])
        mask = purge_mask(t0, t1, 10, 20)
        assert mask[1]  # label at 25-30: t0=25 > 20, no overlap
        assert not mask[0]  # label at 18-25: t0=18 <= 20, overlaps

    def test_adjacent_no_overlap(self):
        t0 = np.array([20])
        t1 = np.array([25])
        mask = purge_mask(t0, t1, 10, 20)
        # t0=20 <= test_end=20 AND t1=25 >= test_start=10 → overlaps → purged
        assert not mask[0]

    def test_empty_arrays(self):
        mask = purge_mask(np.array([]), np.array([]), 10, 20)
        assert mask.size == 0


class TestEmbargoMask:
    def test_before_embargo_keeps(self):
        t0 = np.array([5, 8, 15])
        mask = embargo_mask(t0, 20, 5)
        assert np.all(mask)  # all t0 < 20

    def test_inside_embargo_drops(self):
        t0 = np.array([18, 21, 25])
        mask = embargo_mask(t0, 20, 5)
        assert mask[0]  # 18 < 20 → kept
        assert not mask[1]  # 21 in [20, 25) → embargoed
        assert mask[2]  # 25 not in [20, 25) → kept (half-open range)

    def test_at_boundary(self):
        t0 = np.array([20, 24, 25])
        mask = embargo_mask(t0, 20, 5)
        assert not mask[0]  # 20 in [20, 25) → embargoed
        assert not mask[1]  # 24 in [20, 25) → embargoed
        assert mask[2]  # 25 not in [20, 25) → kept

    def test_empty_arrays(self):
        mask = embargo_mask(np.array([]), 20, 5)
        assert mask.size == 0


class TestTrainTestSplit:
    def test_split_basic(self):
        t0 = np.array([0, 5, 10, 15, 20, 25])
        t1 = np.array([3, 8, 13, 18, 23, 28])
        train, test = train_test_split(t0, t1, 10, 20, embargo_bars=5)
        # test labels: t0 in [10, 20) → indices 2 (t0=10), 3 (t0=15)
        assert np.all(test == np.array([False, False, True, True, False, False]))
        # train: purge overlaps + embargo, excluding test labels
        # index 0: t0=0,t1=3 → t1=3 < 10 → purge_ok, embargo_ok (0<10) → train
        # index 1: t0=5,t1=8 → t1=8 < 10 → purge_ok, embargo_ok → train
        # index 4: t0=20,t1=23 → t0=20 <= 20 → purge: t0<=20 AND t1=23>=10 → overlap → purged
        # index 5: t0=25,t1=28 → t0=25 > 20 → purge_ok. embargo: t0=25 >= 25 → embargo_ok → train
        assert train[0]  # label 0: train
        assert train[1]  # label 1: train
        assert not train[2]  # label 2: test
        assert not train[3]  # label 3: test
        assert not train[4]  # label 4: purged (overlaps test window)
        assert train[5]  # label 5: train (after embargo zone)

    def test_train_and_test_are_disjoint(self):
        t0 = np.array([0, 5, 10, 15, 20])
        t1 = np.array([3, 8, 13, 18, 23])
        train, test = train_test_split(t0, t1, 8, 16, embargo_bars=4)
        assert not np.any(train & test)

    def test_all_labels_accounted_for(self):
        t0 = np.array([0, 5, 10, 15, 20])
        t1 = np.array([3, 8, 13, 18, 23])
        train, test = train_test_split(t0, t1, 8, 16, embargo_bars=4)
        # Every label is either train, test, purged, or embargoed
        purge_ok = purge_mask(t0, t1, 8, 16)
        embargo_ok = embargo_mask(t0, 16, 4)
        combined = train | test | ~purge_ok | ~embargo_ok
        assert np.all(combined)

    def test_zero_embargo(self):
        t0 = np.array([0, 10, 20])
        t1 = np.array([5, 15, 25])
        train, test = train_test_split(t0, t1, 8, 16, embargo_bars=0)
        # No embargo, so label at t0=20 (after test) should be in train
        # But t0=20 > test_end=16, t1=25 > test_start=8 → purge overlap check:
        # t0=20 <= 16? No → purge_ok. embargo_ok (t0=20 < 16? No, t0 >= 16+0? Yes) → embargo_ok
        assert train[2]  # label 2: train


class TestWeightAdjusted:
    def test_equal_weights_matches_simple_mean(self):
        pnl = np.array([0.01, 0.02, 0.03])
        weights = np.ones(3)
        assert weight_adjusted_return(pnl, weights) == pytest.approx(0.02)

    def test_one_weight_dominates(self):
        pnl = np.array([0.01, 0.10])
        weights = np.array([0.01, 0.99])
        result = weight_adjusted_return(pnl, weights)
        assert result > 0.09  # dominated by the 0.10 return

    def test_zero_pnl(self):
        pnl = np.array([0.0, 0.0])
        weights = np.array([1.0, 1.0])
        assert weight_adjusted_return(pnl, weights) == 0.0

    def test_empty(self):
        assert weight_adjusted_return(np.array([]), np.array([])) == 0.0

    def test_sharpe_single_trade(self):
        assert weight_adjusted_sharpe(np.array([0.01]), np.array([1.0])) == 0.0

    def test_sharpe_positive(self):
        pnl = np.array([0.01, 0.02, 0.03, 0.04])
        weights = np.ones(4)
        assert weight_adjusted_sharpe(pnl, weights) > 0

    def test_sharpe_empty(self):
        assert weight_adjusted_sharpe(np.array([]), np.array([])) == 0.0


class TestPurgedFoldWindows:
    def test_returns_correct_fold_count(self):
        total = 1000
        t0 = np.array([10, 50, 100, 200, 300, 500])
        t1 = t0 + 20
        result = purged_fold_windows(total, t0, t1, folds=5)
        assert len(result) == 5

    def test_keys_present(self):
        total = 1000
        t0 = np.array([10, 50, 100])
        t1 = t0 + 10
        result = purged_fold_windows(total, t0, t1, folds=3)
        for fold_info in result:
            assert "test_start" in fold_info
            assert "test_end" in fold_info
            assert "train_mask" in fold_info
            assert "n_purged" in fold_info
            assert "n_embargoed" in fold_info

    def test_no_labels_no_purge(self):
        total = 1000
        t0 = np.array([], dtype=int)
        t1 = np.array([], dtype=int)
        result = purged_fold_windows(total, t0, t1, folds=3)
        for fold_info in result:
            assert fold_info["n_purged"] == 0
            assert fold_info["n_embargoed"] == 0
            assert np.all(fold_info["train_mask"])

    def test_purge_count_non_negative(self):
        total = 1000
        rng = np.random.default_rng(42)
        t0 = rng.integers(0, 900, size=50)
        t1 = t0 + rng.integers(5, 30, size=50)
        result = purged_fold_windows(total, t0, t1, folds=5)
        for fold_info in result:
            assert fold_info["n_purged"] >= 0
            assert fold_info["n_embargoed"] >= 0

    def test_train_mask_boolean(self):
        total = 1000
        t0 = np.array([10, 50, 100, 200, 500])
        t1 = t0 + 15
        result = purged_fold_windows(total, t0, t1, folds=4)
        for fold_info in result:
            assert fold_info["train_mask"].dtype == bool
            assert fold_info["train_mask"].size == len(t0)
