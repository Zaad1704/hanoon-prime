"""hanoon_prime.brain.neurons.adaptive_thresholds_wiring — D1 gate helpers.

Keeps NeuromorphicBridge under the R3 200-line cap by housing the
adaptive-firing-threshold feed here. The feed is gated by
NEURO_ADAPTIVE_THRESHOLD_ENABLED: with the flag OFF it only records
price history in the (inert) adapter and re-asserts build-time neuron
thresholds — scoring stays byte-identical.
"""

from __future__ import annotations

from .network import LIFNetwork
from .threshold_adapter import DynamicThresholdAdapter


def snapshot_base_thresholds(network: LIFNetwork) -> dict[str, float]:
    """Build-time neuron thresholds (input/hidden 0.7, decision 0.6)."""
    return {nid: neuron.threshold for nid, neuron in network._neurons.items()}


def _bounded_scale(dynamic: float, base: float) -> float:
    """Map the adapter's base-scaled output onto a [0.5x, 5.0x] multiplier."""
    return max(0.5, min(5.0, dynamic / max(base, 1e-9)))


def feed_market_env(
    bridge: object,
    ticker: str,
    prices: list[float],
    vix: float | None = None,
) -> None:
    """Feed per-ticker prices/VIX and rescale neuron thresholds when gated on.

    ``bridge`` is any object exposing ``_threshold_adapter``,
    ``_network`` and ``_base_neuron_thresholds`` (the NeuromorphicBridge).
    """
    from ...immune import NEURO_ADAPTIVE_THRESHOLD_ENABLED

    adapter: DynamicThresholdAdapter = bridge._threshold_adapter  # type: ignore[attr-defined]
    if vix is not None:
        adapter.update_vix(vix)
    asset_idx = int(adapter.create_ticker_key(ticker), 16) % len(adapter.price_history)
    for price in prices:
        adapter.record_price(asset_idx, price)
    if not NEURO_ADAPTIVE_THRESHOLD_ENABLED:
        restore_thresholds(bridge)
        return
    dynamic = adapter.compute_dynamic_threshold(asset_idx)
    scale = _bounded_scale(dynamic, adapter.base_threshold)
    for nid, base in bridge._base_neuron_thresholds.items():  # type: ignore[attr-defined]
        bridge._network._neurons[nid].threshold = base * scale  # type: ignore[attr-defined]


def restore_thresholds(bridge: object) -> None:
    """Put build-time thresholds back (0.6/0.7) after adaptive re-scaling."""
    for nid, base in bridge._base_neuron_thresholds.items():  # type: ignore[attr-defined]
        bridge._network._neurons[nid].threshold = base  # type: ignore[attr-defined]


__all__ = ["feed_market_env", "restore_thresholds", "snapshot_base_thresholds"]
