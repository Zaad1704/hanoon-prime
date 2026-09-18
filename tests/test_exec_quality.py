"""tests/test_exec_quality.py — execution-quality gauges."""

from __future__ import annotations

from hanoon_prime.monitor.exec_quality import ExecQuality


def test_record_long_fill_positive_slippage() -> None:
    q = ExecQuality(maxlen=10)
    q.record(
        direction=1,
        expected=100.00,
        actual=100.10,
        submitted=0.0,
        confirmed=0.5,
    )
    s = q.summary()
    assert s["fills"] == 1
    assert 9.0 < s["slippage_bps_mean"] < 11.0  # 10bps bad fill
    assert s["ack_ms_p50"] == 500.0


def test_record_short_fill_negative_slippage() -> None:
    q = ExecQuality(maxlen=10)
    # short sold at 100.10 vs expected 100.00 → favorable for the seller
    q.record(
        direction=-1,
        expected=100.00,
        actual=100.10,
        submitted=0.0,
        confirmed=0.1,
    )
    s = q.summary()
    assert -11.0 < s["slippage_bps_mean"] < -9.0


def test_slippage_positive_for_long_bad_short_bad() -> None:
    q = ExecQuality(maxlen=10)
    q.record(direction=1, expected=100.0, actual=100.05, submitted=0.0, confirmed=0.2)
    q.record(direction=-1, expected=100.0, actual=99.95, submitted=0.0, confirmed=0.4)
    s = q.summary()
    assert s["slippage_bps_mean"] >= 0.0  # both adverse to the taker
    assert s["fills"] == 2


def test_bad_inputs_ignored() -> None:
    q = ExecQuality(maxlen=5)
    q.record(direction=1, expected=0.0, actual=100.0, submitted=0.0, confirmed=0.1)
    q.record(
        direction=1,
        expected=float("nan"),
        actual=100.0,
        submitted=0.0,
        confirmed=0.1,
    )
    assert q.summary()["fills"] == 0


def test_ack_latency_clamps_negative() -> None:
    q = ExecQuality(maxlen=5)
    q.record(direction=1, expected=100.0, actual=100.0, submitted=5.0, confirmed=0.0)
    assert q.summary()["ack_ms_p50"] == 0.0


def test_percentile_helpers_empty() -> None:
    from hanoon_prime.monitor.exec_quality import _pctile, _pmean

    assert _pctile([], 50) == 0.0
    assert _pmean([]) == 0.0
