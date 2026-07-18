"""Evaluate a simple gap-centering heuristic."""

from __future__ import annotations

import argparse

from sparkler.heuristic import heuristic_action
from sparkler.simulator import SparklerSimulator


def run_episode(seed: int | None = None, max_steps: int = 20_000) -> int:
    sim = SparklerSimulator(seed=seed)
    for _ in range(max_steps):
        result = sim.step(flap=heuristic_action(sim))
        if result.terminated:
            return result.score
    return sim.score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristic sparkler agent")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scores = [run_episode(seed=args.seed + i) for i in range(args.episodes)]
    print(f"Heuristic episodes: {args.episodes}")
    print(f"Mean score: {sum(scores) / len(scores):.2f}")
    print(f"Max score: {max(scores)}")
    print(f"Scores: {scores}")


if __name__ == "__main__":
    main()
