"""Train a PPO agent for the sparkler game."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from sparkler.curriculum import EASY, FULL
from train.training_utils import evaluate_model, make_env, train_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on SparklerEnv")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train easy mode first, then ramp to full difficulty",
    )
    args = parser.parse_args()

    models_dir = Path("models")
    logs_dir = Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    if args.curriculum:
        easy_steps = args.timesteps // 2
        full_steps = args.timesteps - easy_steps
        env = make_env(args.n_envs, args.seed, EASY)
        print(f"Curriculum phase 1/2: easy mode ({easy_steps} steps)")
    else:
        easy_steps = 0
        full_steps = args.timesteps
        env = make_env(args.n_envs, args.seed, FULL)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(logs_dir),
    )

    if args.curriculum:
        train_phase(model, env, easy_steps, reset_timesteps=True)
        env.close()
        env = make_env(args.n_envs, args.seed + 1, FULL)
        print(f"Curriculum phase 2/2: full difficulty ({full_steps} steps)")
        train_phase(model, env, full_steps, reset_timesteps=False)
    else:
        train_phase(model, env, full_steps, reset_timesteps=True)

    eval_env = make_env(1, args.seed + 2, FULL)
    evaluate_model(model, eval_env, "full difficulty")
    eval_env.close()
    env.close()

    model_path = models_dir / "sparkler_ppo"
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
