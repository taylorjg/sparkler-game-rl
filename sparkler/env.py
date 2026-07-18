"""Gymnasium environment wrapper for the sparkler simulator."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sparkler.constants import DEFAULT_HEIGHT, DEFAULT_WIDTH
from sparkler.curriculum import CurriculumSettings, FULL
from sparkler.simulator import SparklerSimulator


class SparklerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_episode_steps: int = 10_000,
        seed: int | None = None,
        curriculum: CurriculumSettings | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.max_episode_steps = max_episode_steps
        self.seed_value = seed
        self.curriculum = curriculum or FULL

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -5.0, 0.0, 0.0, -2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 5.0, 1.0, 1.0, 2.0, 2.0, 1.0], dtype=np.float32),
        )

        self.sim: SparklerSimulator | None = None
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self.seed_value
        self.sim = SparklerSimulator(
            self.width,
            self.height,
            seed=episode_seed,
            curriculum=self.curriculum,
        )
        self.steps = 0
        return self._obs_vector(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Environment must be reset before stepping.")

        flap = action == 1
        result = self.sim.step(flap)
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        return self._obs_vector(), result.reward, result.terminated, truncated, self._info()

    def _obs_vector(self) -> np.ndarray:
        assert self.sim is not None
        obs = self.sim.get_observation()
        return np.array(
            [
                obs["ship_y_norm"],
                obs["velocity_y_norm"],
                obs["gap_center_norm"],
                obs["gap_half_height_norm"],
                obs["distance_to_obstacle_norm"],
                obs["scroll_speed_norm"],
                obs["thrust_active"],
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        assert self.sim is not None
        return {
            "score": self.sim.score,
            "gap_percent": self.sim.gap_percent,
            "scroll_x": self.sim.scroll_x,
        }
