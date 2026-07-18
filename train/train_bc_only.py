"""Train a BC-only agent from full-difficulty expert demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from sparkler.curriculum import FULL
from train.bc_utils import evaluate_bc_checkpoint, train_bc_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC from full-game expert demos")
    parser.add_argument("--demo-episodes", type=int, default=500)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=100)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / "sparkler_bc"

    train_bc_checkpoint(
        output_path,
        curriculum=FULL,
        demo_episodes=args.demo_episodes,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes=args.dagger_episodes,
        bc_epochs=args.bc_epochs,
        seed=args.seed,
    )
    evaluate_bc_checkpoint(output_path, seed=args.seed + 1)


if __name__ == "__main__":
    main()
