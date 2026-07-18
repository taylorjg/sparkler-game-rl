"""Shared helpers for PPO training scripts."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from sparkler.curriculum import CurriculumSettings
from sparkler.env import SparklerEnv


def make_env(n_envs: int, seed: int, curriculum: CurriculumSettings) -> VecEnv:
    return make_vec_env(
        SparklerEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"curriculum": curriculum},
    )


def evaluate_mean_score(model: PPO, env: VecEnv, n_episodes: int = 20) -> float:
    return float(np.mean(evaluate_episode_scores(model, env, n_episodes)))


def evaluate_episode_scores(model: PPO, env: VecEnv, n_episodes: int = 20) -> list[int]:
    scores: list[int] = []
    episode_count = 0
    obs = env.reset()
    episode_scores = np.zeros(env.num_envs, dtype=int)

    while episode_count < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, _rewards, dones, infos = env.step(actions)
        for index, done in enumerate(dones):
            if done:
                scores.append(int(infos[index].get("score", episode_scores[index])))
                episode_scores[index] = 0
                episode_count += 1
                if episode_count >= n_episodes:
                    break
            elif "score" in infos[index]:
                episode_scores[index] = int(infos[index]["score"])

    return scores


def train_phase(
    model: PPO,
    env: VecEnv,
    timesteps: int,
    *,
    reset_timesteps: bool,
) -> None:
    model.set_env(env)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=reset_timesteps)


def evaluate_model(model: PPO, env: VecEnv, label: str) -> tuple[float, float, float]:
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
    )
    mean_score = evaluate_mean_score(model, env, n_episodes=20)
    print(f"Eval mean reward ({label}): {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Eval mean score ({label}): {mean_score:.2f} obstacles")
    return mean_reward, std_reward, mean_score
