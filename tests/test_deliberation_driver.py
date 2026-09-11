"""tests/test_deliberation_driver — the bounded Deliberator bridge."""
from __future__ import annotations

from hanoon_prime.brain.config import (
    AFFECTIVE_MOD_BOUND,
    EPISODIC_MOD_BOUND,
    HALIM_MOD_BOUND,
)
from hanoon_prime.brain.deliberation_driver import DeliberationDriver
from hanoon_prime.cortex import Thought


def _thought(score: float = 0.5, confidence: float = 0.6) -> Thought:
    return Thought(verdict="HOLD", score=score, direction=0, confidence=confidence)


def test_compose_passes_raw_mods_to_trace():
    """The trace records raw contributor values; clamping happens in score math."""
    r = DeliberationDriver().compose(
        _thought(0.5), 1.0, halim=0.5, episodic=0.5, thinker_mod=0.5, news_bias=0.0
    )
    assert r.trace["halim_mod"] == 0.5
    assert r.trace["episodic_mod"] == 0.5
    assert r.trace["affective_mod"] == 0.5
    assert r.trace["salience_atten"] == 1.0
    assert r.trace["regime_mult"] == 1.0
    assert "final_score" in r.trace


def test_compose_regime_scales_score():
    r = DeliberationDriver().compose(
        _thought(0.3), 2.0, halim=0.0, episodic=0.0, thinker_mod=0.0, news_bias=0.0
    )
    assert r.score == 0.6  # 0.3 * 2.0


def test_compose_final_score_is_bounded():
    """Oversized mods are clamped to their bounds before summation."""
    r = DeliberationDriver().compose(
        _thought(0.9),
        1.5,
        halim=5.0,  # clamped to HALIM_MOD_BOUND (0.03) in the score math
        episodic=5.0,  # clamped to EPISODIC_MOD_BOUND (0.10)
        thinker_mod=5.0,  # clamped to AFFECTIVE_MOD_BOUND (0.05)
        news_bias=0.03,
    )
    assert r.score == 1.0  # saturated
    assert r.trace["halim_mod"] == 5.0  # raw value still in the CoT trace


def test_compose_salience_scales_confidence():
    r = DeliberationDriver().compose(
        _thought(0.5, confidence=0.8),
        1.0,
        halim=0.0,
        episodic=0.0,
        thinker_mod=0.0,
        news_bias=0.03,
    )
    assert r.trace["salience_atten"] == 1.03
    assert r.confidence == 0.824  # 0.8 * 1.03


def test_modifier_bounds_match_fast_path():
    """Driver's Deliberator bounds must equal the live path's mod bounds."""
    assert HALIM_MOD_BOUND == 0.03
    assert EPISODIC_MOD_BOUND == 0.10
    assert AFFECTIVE_MOD_BOUND == 0.05
