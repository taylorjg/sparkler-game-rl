"""Evaluate heuristic tuning parameters."""

from __future__ import annotations

import argparse

from sparkler.simulator import SparklerSimulator
from sparkler.heuristic import expert_action


def run_episode(
    seed: int,
    *,
    margin: float,
    lookahead: int,
    max_steps: int = 20_000,
) -> int:
    sim = SparklerSimulator(seed=seed)
    for _ in range(max_steps):
        result = sim.step(flap=expert_action(sim, margin=margin, lookahead=lookahead))
        if result.terminated:
            return result.score
    return sim.score


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune heuristic expert parameters")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    best_mean = -1.0
    best_params = (0.0, 8)

    for margin in (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0):
        for lookahead in (4, 8, 12, 16):
            scores = [
                run_episode(args.seed + i, margin=margin, lookahead=lookahead)
                for i in range(args.episodes)
            ]
            mean = sum(scores) / len(scores)
            print(f"margin={margin:5.1f} lookahead={lookahead:2d}  mean={mean:.2f}  max={max(scores)}")
            if mean > best_mean:
                best_mean = mean
                best_params = (margin, lookahead)

    print(f"\nBest: margin={best_params[0]}, lookahead={best_params[1]}, mean={best_mean:.2f}")


if __name__ == "__main__":
    main()
