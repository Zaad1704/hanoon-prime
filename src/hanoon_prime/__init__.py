"""hanoon_prime — the minimal, profit-first trading system.

Architecture: IB → JULI → Entry/Hold/Exit + Learning
"""
from .brain import Brain, Thought
from .alpha import compute_alpha, INDICATOR_NAMES
from .scoring import compute_score
from .edge import compute_ev, score_to_win_prob, kelly_fraction
from .thinker import deliberate
from .journal import Journal
from .data import load_ohlcv, compute_buy_volume, estimate_bid_ask
from ._calibrate import calibrate, Calibration

__version__ = "1.0.0"
__all__ = [
    "Brain",
    "Thought",
    "compute_alpha",
    "INDICATOR_NAMES",
    "compute_score",
    "compute_ev",
    "score_to_win_prob",
    "kelly_fraction",
    "deliberate",
    "Journal",
    "load_ohlcv",
    "compute_buy_volume",
    "estimate_bid_ask",
    "calibrate",
    "Calibration",
]
