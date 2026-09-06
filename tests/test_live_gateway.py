"""Live, read-only integration tests against a running IB Gateway.

These are marked ``live`` and EXCLUDED from the ``-m "not live"`` gate, so the
default unit run never touches a broker.  They are intentionally **read-only**:
``reqMktData`` (market data), ``reqHistoricalData`` (bars), ``qualifyContracts``
and ``accountSummary``.  They NEVER call ``placeOrder``, never mutate positions,
and never touch the live (4001) port — only the paper Gateway port 4002
(``IB_PAPER_PORT``).

env notes:
  * pytest 9 strict ``filterwarnings = error`` turns ``ib_insync``'s eventkit
    ``DeprecationWarning`` (and its unfinalised ``connectAsync`` coroutine
    ``RuntimeWarning``) into hard errors.  We suppress warnings for these live
    tests only.
  * py3.14 requires a current event loop in the main thread; we install one and
    keep it alive for the whole module (do NOT null it between tests, which
    deadlocks the module-scoped client).
  * ``ib_compat`` leaves ``ib = None`` under the pytest collector (the warning
    above), so we import ``ib_insync`` raw and pin it onto the
    ``ib_streamer`` MODULE global that ``subscribe`` reads.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

import hanoon_prime.ib_streamer as ibs_module
from hanoon_prime.ib_streamer import IBStreamer
from hanoon_prime.immune import IB_CLIENT_ID, IB_HOST, IB_PAPER_PORT

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _live_warnings():
    """Silence ib_insync's py3.14 DeprecationWarnings per-test.

    NOTE: this must NOT touch the event loop — the module-scoped ``paper_ib``
    client owns the loop; resetting it between tests deadlocks teardown.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _import_real_ib():
    """Import the real ib_insync module, tolerating the py3.14 eventkit warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import ib_insync
    return ib_insync


@pytest.fixture(scope="module")
def paper_ib():
    """A read-only IB client connected to the paper Gateway (one per module)."""
    try:
        ib_mod = _import_real_ib()
    except Exception as exc:  # pragma: no cover - env guard
        pytest.skip(f"ib_insync unavailable: {exc}")
    saved = ibs_module.ib
    ibs_module.ib = ib_mod  # module global that IBStreamer.subscribe reads
    # py3.14 needs a current event loop in the main thread for ib_insync's
    # synchronous connect/reqHistoricalData to drive the asyncio loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = ib_mod.IB()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ok = client.connect(
                IB_HOST, IB_PAPER_PORT, clientId=IB_CLIENT_ID + 99, timeout=8
            )
        except Exception as exc:  # pragma: no cover - gateway absent
            ibs_module.ib = saved
            pytest.skip(f"IB Gateway unreachable: {exc}")
    if not ok or not client.isConnected():
        ibs_module.ib = saved
        pytest.skip("IB Gateway paper port not connected")
    yield client
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            for ticker in list(client.ticker_subs):
                client.cancelMktData(client.ticker_subs[ticker])
            if client.isConnected():
                client.disconnect()
        except Exception:  # pragma: no cover - teardown best-effort
            pass
    ibs_module.ib = saved


def test_gateway_accepts_api_client(paper_ib):
    """Read-only connectivity: account summary."""
    assert paper_ib.isConnected()
    summary = paper_ib.accountSummary()
    accounts = {a.account for a in summary}
    assert accounts, "expected at least one account from accountSummary"


@pytest.mark.parametrize("ticker", ["AAPL", "SPY"])
def test_streamer_seed_and_atr(paper_ib, ticker):
    """Drive the previously-uncaptured subscribe / seed_history / ATR path."""
    streamer = IBStreamer(paper_ib)
    streamer.subscribe(ticker)
    assert ticker in streamer.buffers and ticker in streamer.ticker_subs

    streamer.seed_history(ticker)  # reqHistoricalData — works without a data sub
    buf = streamer.buffers[ticker]
    assert len(buf.close) > 0

    # ready() True-branch after seeding >= EDGE_LOOKBACK bars.
    assert streamer.ready(ticker) is True

    atr = streamer.buffer_atr(ticker)
    assert atr > 0.0

    arrays = streamer.get_arrays(ticker)
    assert "close" in arrays and "high" in arrays

    assert isinstance(streamer.get_last_price(ticker), float)

    # update_bar() must not raise even with no live BBO feed (bid/ask may be -1).
    assert isinstance(streamer.update_bar(ticker), bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paper_ib.cancelMktData(streamer.ticker_subs[ticker])


def test_est_buy_vol_helper():
    """Pure helper, exercised through the live wrapper as well."""
    assert IBStreamer._est_buy_vol(100.0, 105.0, 95.0, 1000.0) == 500.0
    # flat OHLC -> 0.0 buy-volume fraction.
    assert IBStreamer._est_buy_vol(100.0, 100.0, 100.0, 1000.0) == 0.0
