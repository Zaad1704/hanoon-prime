"""brain.arm_stats — Beta-arm math shared by bandit + learning telemetry."""

from __future__ import annotations

from typing import Any


def beta_variance(alpha: float, beta: float) -> float:
    """Variance of Beta(α, β): ~0 = committed, up to 1/12 = unshaped."""
    if alpha <= 0.0 or beta <= 0.0:
        return 0.0
    s = alpha + beta
    if s <= 0.0:
        return 0.0
    return (alpha * beta) / (s * s * (s + 1.0))


def cell_variance(arm: dict[str, Any]) -> float:
    """Beta variance from an arm row's [alpha, beta] pair."""
    ab = arm.get("ab")
    if isinstance(ab, list) and len(ab) == 2:
        return beta_variance(float(ab[0]), float(ab[1]))
    return 0.0


def arms_list(arms: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten per-regime arm rows into one list of arm entries."""
    out: list[dict[str, Any]] = []
    for row in arms.values():
        for arm in row:
            if isinstance(arm, dict):
                out.append(arm)
    return out


def posterior_stats(
    arms: dict[str, list[dict[str, Any]]],
    commit_var: float,
    min_samples: int,
) -> tuple[int, int, int]:
    """(committed, thin, total) over all bandit cells."""
    committed = 0
    thin = 0
    entries = arms_list(arms)
    for arm in entries:
        n = float(arm.get("n", 0))
        if cell_variance(arm) < commit_var and n >= min_samples:
            committed += 1
        elif n < min_samples:
            thin += 1
    return committed, thin, len(entries)


__all__ = ["beta_variance", "cell_variance", "arms_list", "posterior_stats"]
