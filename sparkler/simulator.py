"""Headless sparkler game simulator."""

from __future__ import annotations

from dataclasses import dataclass
import random

from sparkler import constants as C


@dataclass
class ObstaclePair:
    x: float
    width: float
    gap_top: float
    gap_bottom: float
    gap_percent: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def gap_center(self) -> float:
        return (self.gap_top + self.gap_bottom) / 2

    @property
    def gap_half_height(self) -> float:
        return (self.gap_bottom - self.gap_top) / 2


@dataclass
class StepResult:
    reward: float
    terminated: bool
    obstacle_cleared: bool
    score: int


class SparklerSimulator:
    """Deterministic headless port of the Phaser game loop."""

    def __init__(
        self,
        width: int = C.DEFAULT_WIDTH,
        height: int = C.DEFAULT_HEIGHT,
        seed: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.scroll_x = 0.0
        self.ship_y = self.height * C.SHIP_Y_RATIO
        self.velocity_y = 0.0
        self.running_elapsed_ms = 0.0
        self.gap_percent = float(C.INITIAL_GAP_PERCENT)
        self.obstacle_pair_cleared = False
        self.score = 0
        self.thrust_remaining_ms = 0.0
        self.alive = True
        self.obstacle = self._make_obstacle_pair(
            self.width * C.INITIAL_OBSTACLE_X_RATIO,
            self.gap_percent,
        )

    @property
    def ship_screen_x(self) -> float:
        return self.width * C.SHIP_X_RATIO

    @property
    def ship_world_x(self) -> float:
        return self.scroll_x + self.ship_screen_x

    @property
    def thrust_active(self) -> bool:
        return self.thrust_remaining_ms > 0

    def flap(self) -> None:
        if not self.thrust_active:
            self.thrust_remaining_ms = C.STIMULUS_DURATION_MS

    def step(self, flap: bool, delta_ms: float = C.REFERENCE_FRAME_MS) -> StepResult:
        if not self.alive:
            raise RuntimeError("Cannot step after terminal state; call reset() first.")

        delta_ms = min(delta_ms, C.MAX_DELTA_MS)
        reward = C.REWARD_SURVIVAL
        obstacle_cleared = False
        terminated = False

        if flap:
            self.flap()

        self.running_elapsed_ms += delta_ms

        acceleration_y = C.UP_THRUST if self.thrust_active else 0.0
        total_accel_y = C.GRAVITY_Y + acceleration_y
        dt = delta_ms / 1000.0
        self.velocity_y += total_accel_y * dt
        self.ship_y += self.velocity_y * dt
        self._clamp_ship_to_world()

        scroll_distance = self._get_scroll_distance(delta_ms)
        self.scroll_x += scroll_distance

        if self._collides():
            self.alive = False
            terminated = True
            reward += C.REWARD_DEATH
            return StepResult(reward, terminated, obstacle_cleared, self.score)

        if not self.obstacle_pair_cleared and self.ship_world_x >= self.obstacle.right:
            self.obstacle_pair_cleared = True
            self.score += 1
            obstacle_cleared = True
            reward += C.REWARD_OBSTACLE_CLEARED

        if self.obstacle.right < self.scroll_x:
            if self.gap_percent > C.MIN_GAP_PERCENT:
                self.gap_percent -= C.GAP_SHRINK_STEP
            obstacle_x = (
                self.scroll_x
                + self.width
                + self._get_obstacle_width()
                + C.OBSTACLE_LINE_WIDTH
            )
            self.obstacle = self._make_obstacle_pair(obstacle_x, self.gap_percent)
            self.obstacle_pair_cleared = False

        if self.thrust_active:
            self.thrust_remaining_ms -= delta_ms
            if self.thrust_remaining_ms <= 0:
                self.thrust_remaining_ms = 0.0

        return StepResult(reward, terminated, obstacle_cleared, self.score)

    def get_observation(self) -> dict[str, float]:
        distance_to_obstacle = self.obstacle.x - self.ship_world_x
        max_speed = self._get_speed(at_elapsed_ms=C.SPEED_RAMP_DURATION_MS)
        return {
            "ship_y_norm": self.ship_y / self.height,
            "velocity_y_norm": self.velocity_y / 1000.0,
            "gap_center_norm": self.obstacle.gap_center / self.height,
            "gap_half_height_norm": self.obstacle.gap_half_height / self.height,
            "distance_to_obstacle_norm": distance_to_obstacle / self.width,
            "scroll_speed_norm": self._get_speed() / max(max_speed, 1.0),
            "thrust_active": 1.0 if self.thrust_active else 0.0,
        }

    def _clamp_ship_to_world(self) -> None:
        if self.ship_y <= 0:
            self.ship_y = 0.0
            self.velocity_y = max(self.velocity_y, 0.0)
        if self.ship_y >= self.height:
            self.ship_y = float(self.height)
            self.velocity_y = min(self.velocity_y, 0.0)

    def _collides(self) -> bool:
        ship_x = self.ship_world_x
        obstacle = self.obstacle
        if ship_x < obstacle.x or ship_x > obstacle.right:
            return False
        return self.ship_y < obstacle.gap_top or self.ship_y > obstacle.gap_bottom

    def _make_obstacle_pair(self, x: float, gap_percent: float) -> ObstaclePair:
        obstacle_width = self._get_obstacle_width()
        gap_height = self.height * gap_percent / 100.0
        half_remaining_height = (self.height - gap_height) / 2.0
        centre_offset_ratio = self.rng.uniform(-0.5, 0.5)
        upper_height = (1.0 + centre_offset_ratio) * half_remaining_height
        lower_height = (1.0 - centre_offset_ratio) * half_remaining_height
        gap_top = upper_height
        gap_bottom = self.height - lower_height
        return ObstaclePair(
            x=x,
            width=float(obstacle_width),
            gap_top=gap_top,
            gap_bottom=gap_bottom,
            gap_percent=gap_percent,
        )

    def _get_obstacle_width(self) -> int:
        max_dimension = max(self.width, self.height)
        return round(max_dimension / 20)

    def _get_speed(self, at_elapsed_ms: float | None = None) -> int:
        elapsed = self.running_elapsed_ms if at_elapsed_ms is None else at_elapsed_ms
        max_dimension = max(self.width, self.height)
        base_speed = max_dimension / 200.0
        ramp_progress = min(1.0, elapsed / C.SPEED_RAMP_DURATION_MS)
        multiplier = 1.0 + (C.MAX_SPEED_MULTIPLIER - 1.0) * ramp_progress
        return round(base_speed * multiplier)

    def _get_scroll_distance(self, delta_ms: float) -> float:
        return self._get_speed() * (delta_ms / C.REFERENCE_FRAME_MS)
