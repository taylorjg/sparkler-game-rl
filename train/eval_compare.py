"""Compare heuristic, BC-only, and BC+PPO agents on full difficulty."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from sparkler.curriculum import FULL
from train.bc_utils import train_bc_checkpoint
from train.eval_heuristic import run_episode
from train.training_utils import evaluate_episode_scores, make_env


@dataclass
class EvalResult:
    name: str
    scores: list[int]

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores))

    @property
    def max(self) -> int:
        return int(max(self.scores))


def evaluate_heuristic(n_episodes: int, seed: int) -> EvalResult:
    scores = [run_episode(seed=seed + i) for i in range(n_episodes)]
    return EvalResult("Heuristic (full game)", scores)


def evaluate_saved_model(name: str, model_path: Path, n_episodes: int, seed: int) -> EvalResult:
    env = make_env(1, seed, FULL)
    model = PPO.load(model_path, env=env)
    scores = evaluate_episode_scores(model, env, n_episodes)
    env.close()
    return EvalResult(name, scores)


def print_comparison(results: list[EvalResult]) -> None:
    print()
    print(f"{'Agent':<28} {'Mean':>8} {'Std':>8} {'Max':>6}")
    print("-" * 54)
    for result in results:
        print(
            f"{result.name:<28} "
            f"{result.mean:>8.2f} "
            f"{result.std:>8.2f} "
            f"{result.max:>6}"
        )
    print()


def model_path(base: Path) -> Path | None:
    if base.exists():
        return base
    zip_path = base.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare agents on full difficulty")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-episodes", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument(
        "--rebuild-bc",
        action="store_true",
        help="Retrain and overwrite the BC-only checkpoint",
    )
    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    bc_path = models_dir / "sparkler_bc"
    ppo_path = model_path(models_dir / "sparkler_ppo")

    results: list[EvalResult] = []

    print(f"Evaluating on full difficulty ({args.episodes} episodes)...")
    results.append(evaluate_heuristic(args.episodes, args.seed))

    if not model_path(bc_path) or args.rebuild_bc:
        train_bc_checkpoint(
            bc_path,
            demo_episodes=args.demo_episodes,
            bc_epochs=args.bc_epochs,
            seed=args.seed,
        )
    bc_model_path = model_path(bc_path)
    assert bc_model_path is not None
    results.append(
        evaluate_saved_model("BC only (full game)", bc_model_path, args.episodes, args.seed + 100)
    )

    if ppo_path is not None:
        results.append(
            evaluate_saved_model("BC + PPO (full game)", ppo_path, args.episodes, args.seed + 200)
        )
    else:
        print("Skipping BC+PPO: models/sparkler_ppo not found")

    print_comparison(results)


if __name__ == "__main__":
    main()
