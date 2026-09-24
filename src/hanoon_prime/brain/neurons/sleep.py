"""hanoon_prime.brain.neurons.sleep — Offline memory consolidation.

During market closure, replays stored attractor patterns to stabilize
synaptic weights and prevent catastrophic forgetting.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from . import replay as _replay
from .attractor import Attractor, AttractorMemory
from .network import LIFNetwork
from .stdp import STDPLearner


@dataclass
class SleepResult:
    """Result of sleep consolidation cycle."""

    patterns_replayed: int = 0
    spikes_generated: int = 0
    weights_updated: int = 0
    mean_weight_change: float = 0.0
    duration_ms: float = 0.0


class SleepReplayEngine:
    """Offline replay engine for continual learning during non-trading hours.

    Injects Poisson noise + replays attractor patterns through the SNN
    to consolidate learned behaviors without market data.
    """

    POISSON_RATE: float = 3.0
    REPLAY_BATCH: int = 10
    REPLAY_STEPS: int = 5
    WIN_BIAS: float = 1.0
    LOSS_BIAS: float = 3.0
    MAX_PATTERNS: int = 100

    def __init__(
        self,
        network: LIFNetwork,
        stdp: STDPLearner,
        memory: AttractorMemory,
    ) -> None:
        self._network = network
        self._stdp = stdp
        self._memory = memory
        self._cycle_count: int = 0
        self.last_replay: SleepResult | None = None

    def should_run(self, is_market_open: bool) -> bool:
        """Should sleep consolidation run? Only when market is closed."""
        return not is_market_open

    def encode_pattern(self, center: List[float]) -> Dict[str, float]:
        """Map an attractor center (EPISODIC_KEYS order) to real input neurons."""
        return _replay.encode_pattern(self._network, center)

    def select_patterns(
        self,
        replay_list: List[Tuple[Dict[str, float], float, bool]] | None = None,
    ) -> List[Tuple[Dict[str, float], float, bool]]:
        """Select patterns for replay (3× loser drive, or an override list).

        Every entry is (pattern, drive, won): the win/loss polarity is
        explicit so _replay_pattern can punish losers and reward winners
        without inferring sign from drive magnitude (FIX-2026-09-23-08).
        """
        if replay_list is not None:
            patterns = list(replay_list)
            if len(patterns) > self.MAX_PATTERNS:
                random.shuffle(patterns)
                patterns = patterns[: self.MAX_PATTERNS]
            return patterns

        attractors: list[Attractor] = list(self._memory)
        if not attractors:
            return []

        patterns = []
        for att in attractors:
            if att.trade_count < 2:
                continue

            won = att.wins > att.losses
            weight = self.LOSS_BIAS if not won else self.WIN_BIAS
            patterns.append((self.encode_pattern(att.center), weight, won))

        if len(patterns) > self.MAX_PATTERNS:
            random.shuffle(patterns)
            patterns = patterns[: self.MAX_PATTERNS]

        return patterns

    def run_cycle(
        self,
        duration_sec: float = 60.0,
        replay_list: List[Tuple[Dict[str, float], float, bool]] | None = None,
    ) -> SleepResult:
        """Run one sleep consolidation cycle."""
        start = time.time()

        patterns = self.select_patterns(replay_list)
        if not patterns:
            result = SleepResult(duration_ms=(time.time() - start) * 1000)
            self.last_replay = result
            return result

        spikes_generated = 0
        weights_updated = 0
        total_change = 0.0
        deadline = start + duration_sec

        for pattern, weight, won in patterns[: self.REPLAY_BATCH]:
            if time.time() >= deadline:
                break
            count, change, updated = self._replay_pattern(pattern, weight, won)
            spikes_generated += count
            total_change += change
            weights_updated += updated

        self._cycle_count += 1

        result = SleepResult(
            patterns_replayed=len(patterns[: self.REPLAY_BATCH]),
            spikes_generated=spikes_generated,
            weights_updated=weights_updated,
            mean_weight_change=total_change
            / max(1, len(patterns[: self.REPLAY_BATCH])),
            duration_ms=(time.time() - start) * 1000,
        )
        self.last_replay = result
        return result

    def snapshot(self) -> dict[str, Any]:
        """Telemetry summary of the sleep engine and its last cycle."""
        last = self.last_replay
        return {
            "initialized": True,
            "cycle_count": self._cycle_count,
            "last_replay": None
            if last is None
            else {
                "patterns_replayed": last.patterns_replayed,
                "spikes_generated": last.spikes_generated,
                "weights_updated": last.weights_updated,
                "mean_weight_change": last.mean_weight_change,
                "duration_ms": last.duration_ms,
            },
        }

    def _replay_pattern(
        self,
        pattern: Dict[str, float],
        weight: float,
        won: bool,
    ) -> Tuple[int, float, int]:
        """Replay one pattern: drive real inputs, let spikes consolidate.

        ``won`` is the explicit trade-outcome polarity: winners consolidate
        with positive reward, losers are punished with negative reward.
        The sign is NEVER inferred from ``weight`` (drive magnitude) —
        losers replay at 3× drive (SLEEP_LOSS_WEIGHT) precisely so the
        punishment lands harder, making the scheme punishment-dominant
        per CONTRACT (FIX-2026-09-23-08).

        Returns (spikes fired, total abs synapse change, synapses modified).
        """
        before = _replay.snapshot_synapses(self._stdp)
        _replay.apply_pattern_input(self._network, pattern, weight)
        spikes = _replay.drive_steps(
            self._network,
            self._stdp,
            steps=self.REPLAY_STEPS,
            poisson_rate=self.POISSON_RATE,
        )
        self._stdp.apply_reward(1.0 if won else -1.0)
        updated, change = _replay.synapse_delta_stats(before, self._stdp)
        return spikes, change, updated


__all__ = ["SleepReplayEngine", "SleepResult"]
