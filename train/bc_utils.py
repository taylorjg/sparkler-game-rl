"""Train and save a behavioral cloning checkpoint."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from sparkler.curriculum import CurriculumSettings, FULL
from sparkler.demos import (
    collect_dagger_demonstrations,
    collect_demonstrations,
    merge_demonstrations,
    pretrain_with_behavioral_cloning,
)
from train.training_utils import evaluate_episode_scores, evaluate_model, make_env


def train_bc_checkpoint(
    output_path: Path,
    *,
    curriculum: CurriculumSettings = FULL,
    demo_episodes: int = 500,
    dagger_rounds: int = 3,
    dagger_episodes: int = 100,
    bc_epochs: int = 30,
    seed: int = 42,
) -> PPO:
    curriculum_label = "full difficulty" if curriculum.enable_speed_ramp else "easy mode"
    print(f"Collecting {demo_episodes} expert episodes ({curriculum_label})...")
    dataset = collect_demonstrations(
        demo_episodes,
        curriculum=curriculum,
        seed=seed,
    )
    print(f"Collected {len(dataset[0])} expert transitions")

    env = make_env(1, seed, curriculum)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)

    for round_index in range(dagger_rounds + 1):
        if round_index > 0:
            print(
                f"DAgger round {round_index}/{dagger_rounds}: "
                f"collecting {dagger_episodes} policy rollouts..."
            )
            dagger_data = collect_dagger_demonstrations(
                model,
                dagger_episodes,
                curriculum=curriculum,
                seed=seed + 10_000 + round_index * 1_000,
            )
            print(f"Collected {len(dagger_data[0])} DAgger transitions")
            dataset = merge_demonstrations(dataset, dagger_data)

        print(
            f"Behavioral cloning on {len(dataset[0])} transitions "
            f"(round {round_index}, {bc_epochs} epochs)..."
        )
        pretrain_with_behavioral_cloning(
            model,
            dataset[0],
            dataset[1],
            n_epochs=bc_epochs,
        )

    model.save(output_path)
    print(f"Saved BC model to {output_path}")
    env.close()
    return model


def evaluate_bc_checkpoint(
    model_path: Path,
    *,
    seed: int = 42,
    n_episodes: int = 20,
) -> None:
    eval_env = make_env(1, seed, FULL)
    model = PPO.load(model_path, env=eval_env)
    evaluate_model(model, eval_env, "full difficulty after BC")
    scores = evaluate_episode_scores(model, eval_env, n_episodes)
    eval_env.close()
    print(f"Episode scores: {scores}")
