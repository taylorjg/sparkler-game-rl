"""Behavioral cloning pre-training followed by PPO fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from sparkler.curriculum import CurriculumSettings, EASY, FULL
from sparkler.demos import collect_demonstrations, pretrain_with_behavioral_cloning
from train.training_utils import evaluate_model, make_env, train_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="BC pre-train + PPO fine-tune")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-episodes", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=30)
    args = parser.parse_args()

    models_dir = Path("models")
    logs_dir = Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    print(f"Collecting {args.demo_episodes} expert episodes (easy mode)...")
    observations, actions = collect_demonstrations(
        args.demo_episodes,
        curriculum=EASY,
        seed=args.seed,
    )
    print(f"Collected {len(observations)} transitions")

    env = make_env(args.n_envs, args.seed, EASY)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(logs_dir),
    )

    print(f"Behavioral cloning for {args.bc_epochs} epochs...")
    pretrain_with_behavioral_cloning(
        model,
        observations,
        actions,
        n_epochs=args.bc_epochs,
    )

    bc_path = models_dir / "sparkler_bc"
    model.save(bc_path)
    print(f"Saved BC model to {bc_path}")

    bc_eval_env = make_env(1, args.seed + 1, EASY)
    evaluate_model(model, bc_eval_env, "easy mode after BC")
    bc_eval_env.close()

    easy_steps = args.timesteps // 2
    full_steps = args.timesteps - easy_steps
    print(f"PPO fine-tune phase 1/2: easy mode ({easy_steps} steps)")
    train_phase(model, env, easy_steps, reset_timesteps=True)

    env.close()
    env = make_env(args.n_envs, args.seed + 2, FULL)
    print(f"PPO fine-tune phase 2/2: full difficulty ({full_steps} steps)")
    train_phase(model, env, full_steps, reset_timesteps=False)

    eval_env = make_env(1, args.seed + 3, FULL)
    evaluate_model(model, eval_env, "full difficulty after BC+PPO")
    eval_env.close()
    env.close()

    model_path = models_dir / "sparkler_ppo"
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
