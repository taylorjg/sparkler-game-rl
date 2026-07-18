"""Demonstration collection and behavioral cloning."""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO

from sparkler.curriculum import CurriculumSettings, FULL
from sparkler.env import SparklerEnv
from sparkler.heuristic import expert_action


def collect_demonstrations(
    n_episodes: int,
    *,
    curriculum: CurriculumSettings = FULL,
    seed: int = 42,
    max_steps: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[int] = []

    for episode in range(n_episodes):
        env = SparklerEnv(curriculum=curriculum, seed=seed + episode)
        obs, _info = env.reset()
        for _ in range(max_steps):
            assert env.sim is not None
            action = int(expert_action(env.sim))
            observations.append(obs)
            actions.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def collect_dagger_demonstrations(
    model: PPO,
    n_episodes: int,
    *,
    curriculum: CurriculumSettings = FULL,
    seed: int = 10_000,
    max_steps: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the current policy and label visited states with the expert."""
    observations: list[np.ndarray] = []
    actions: list[int] = []

    for episode in range(n_episodes):
        env = SparklerEnv(curriculum=curriculum, seed=seed + episode)
        obs, _info = env.reset()
        for _ in range(max_steps):
            assert env.sim is not None
            observations.append(obs)
            actions.append(int(expert_action(env.sim)))
            policy_action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(int(policy_action))
            if terminated or truncated:
                break

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def merge_demonstrations(
    *datasets: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.concatenate([dataset[0] for dataset in datasets], axis=0)
    actions = np.concatenate([dataset[1] for dataset in datasets], axis=0)
    return observations, actions


def pretrain_with_behavioral_cloning(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    n_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> float:
    """Supervised pre-training to mimic expert flap decisions."""
    model.policy.set_training_mode(True)
    optimizer = th.optim.Adam(model.policy.parameters(), lr=learning_rate)

    obs_tensor = th.as_tensor(observations, device=model.device, dtype=th.float32)
    act_tensor = th.as_tensor(actions, device=model.device, dtype=th.long)
    n_samples = len(observations)

    for epoch in range(n_epochs):
        permutation = th.randperm(n_samples, device=model.device)
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, n_samples, batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_obs = obs_tensor[batch_indices]
            batch_actions = act_tensor[batch_indices]

            features = model.policy.extract_features(batch_obs)
            latent_pi, _latent_vf = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)
            loss = th.nn.functional.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        mean_loss = epoch_loss / max(batch_count, 1)
        print(f"BC epoch {epoch + 1}/{n_epochs} loss={mean_loss:.4f}")

    model.policy.set_training_mode(False)

    with th.no_grad():
        features = model.policy.extract_features(obs_tensor)
        latent_pi, _latent_vf = model.policy.mlp_extractor(features)
        logits = model.policy.action_net(latent_pi)
        predictions = th.argmax(logits, dim=1)
        accuracy = (predictions == act_tensor).float().mean().item()

    print(f"BC action accuracy on demonstrations: {accuracy * 100:.1f}%")
    return accuracy
