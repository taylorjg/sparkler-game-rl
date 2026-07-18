"""Train a PPO agent for the sparkler game."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from sparkler.env import SparklerEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on SparklerEnv")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_dir = Path("models")
    logs_dir = Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    env = make_vec_env(
        SparklerEnv,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(logs_dir),
    )
    model.learn(total_timesteps=args.timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
    )
    print(f"Eval mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    model_path = models_dir / "sparkler_ppo"
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
